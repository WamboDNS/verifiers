"""Multi-agent environment for training multiple agents with independent rollouts."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import abstractmethod
from typing import Any, Literal

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.agents.agent import Agent, AgentConfig
from verifiers.collectors.trace_collector import TraceCollector
from verifiers.types import (
    Messages,
    ModelResponse,
    RolloutInput,
    RolloutTiming,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.response_utils import (
    parse_is_truncated,
    parse_response_messages,
    parse_response_tokens,
)

logger = logging.getLogger(__name__)


class MultiAgentMonitorRubric(vf.Rubric):
    """Monitor rubric that tracks per-agent metrics."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_metric(self.num_turns)
        self.add_metric(self.num_agents)

    async def num_turns(self, state: State) -> int:
        """Total number of turns across all agents."""
        collector = state.get("collector")
        if collector is not None:
            return len(collector)
        return len(state.get("trajectory", []))

    async def num_agents(self, state: State) -> int:
        """Number of agents that participated."""
        collector = state.get("collector")
        if collector is not None:
            return len(collector.get_agent_ids())
        return 1


class MultiAgentEnv(vf.MultiTurnEnv):
    """
    Environment for multi-agent rollouts with per-agent training support.

    MultiAgentEnv extends MultiTurnEnv to support multiple agents, each with:
    - Independent conversation contexts
    - Per-agent trajectory collection
    - LoRA adapter grouping for training
    - Optional frozen (non-trainable) agents

    By default, agents run independently (no interaction). Override
    `get_agent_observation` to implement agent-to-agent communication.

    Example:
        ```python
        class TicTacToeEnv(MultiAgentEnv):
            def __init__(self):
                agents = {
                    "player_x": AgentConfig(
                        agent_id="player_x",
                        system_prompt="You are playing X in tic-tac-toe.",
                        trainable=True,
                        lora_id=0,
                    ),
                    "player_o": AgentConfig(
                        agent_id="player_o",
                        system_prompt="You are playing O in tic-tac-toe.",
                        trainable=True,
                        lora_id=1,
                    ),
                }
                super().__init__(
                    agents=agents,
                    turn_order="sequential",
                    max_turns=9,
                    dataset=my_dataset,
                )

            async def get_initial_observation(
                self, agent_id: str, state: State
            ) -> str | None:
                return f"Game started. You are {agent_id.split('_')[1].upper()}."

            async def get_agent_observation(
                self, agent_id: str, response: Messages, state: State
            ) -> str | None:
                # Show opponent's move to each agent
                board = state.get("board", "")
                return f"Current board:\\n{board}"
        ```
    """

    def __init__(
        self,
        agents: dict[str, Agent | AgentConfig],
        turn_order: Literal["sequential", "parallel"] = "parallel",
        max_turns: int = 10,
        **kwargs: Any,
    ):
        """
        Initialize a multi-agent environment.

        Args:
            agents: Dict mapping agent_id to Agent or AgentConfig.
                AgentConfigs are automatically converted to Agents.
            turn_order: How agents take turns:
                - "parallel": All agents act simultaneously each turn
                - "sequential": Agents act in order, one per turn
            max_turns: Maximum number of environment turns.
                For sequential, this is total turns across all agents.
                For parallel, this is total rounds (each round = all agents act).
            **kwargs: Additional arguments passed to MultiTurnEnv.
        """
        if not agents:
            raise ValueError("agents dict cannot be empty")

        super().__init__(max_turns=max_turns, **kwargs)

        # Convert AgentConfigs to Agents
        self.agents: dict[str, Agent] = {}
        for agent_id, agent_or_config in agents.items():
            if isinstance(agent_or_config, AgentConfig):
                if agent_or_config.agent_id != agent_id:
                    raise ValueError(
                        f"AgentConfig.agent_id ({agent_or_config.agent_id!r}) "
                        f"must match dict key ({agent_id!r})"
                    )
                self.agents[agent_id] = Agent.from_config(agent_or_config)
            else:
                if agent_or_config.agent_id != agent_id:
                    raise ValueError(
                        f"Agent.agent_id ({agent_or_config.agent_id!r}) "
                        f"must match dict key ({agent_id!r})"
                    )
                self.agents[agent_id] = agent_or_config

        self.turn_order = turn_order
        self.agent_ids = list(self.agents.keys())

        # Add monitor rubric
        self.add_rubric(MultiAgentMonitorRubric())

        self.logger.debug(
            f"Initialized MultiAgentEnv with {len(self.agents)} agents, "
            f"turn_order={turn_order}, max_turns={max_turns}"
        )

    # -------------------------------------------------------------------------
    # Agent access helpers
    # -------------------------------------------------------------------------

    def get_trainable_agents(self) -> list[str]:
        """
        Get list of trainable agent IDs.

        Returns:
            List of agent_ids for agents with trainable=True.
        """
        return [aid for aid, agent in self.agents.items() if agent.trainable]

    def get_lora_groups(self) -> dict[int | None, list[str]]:
        """
        Group trainable agents by their LoRA adapter ID.

        Returns:
            Dict mapping lora_id to list of agent_ids.
            Agents with lora_id=None are grouped under None.
        """
        groups: dict[int | None, list[str]] = {}
        for aid, agent in self.agents.items():
            if agent.trainable:
                lora = agent.lora_id
                if lora not in groups:
                    groups[lora] = []
                groups[lora].append(aid)
        return groups

    # -------------------------------------------------------------------------
    # Hooks for subclasses to override
    # -------------------------------------------------------------------------

    async def get_active_agents(self, state: State) -> list[str]:
        """
        Determine which agents act this turn.

        Override this for custom turn-taking logic (e.g., dynamic ordering).

        Args:
            state: Current rollout state.

        Returns:
            List of agent_ids that should act this turn.
        """
        if self.turn_order == "parallel":
            return list(self.agent_ids)
        else:
            # Sequential: rotate through agents
            turn_idx = state.get("current_turn", 0) % len(self.agent_ids)
            return [self.agent_ids[turn_idx]]

    async def get_initial_observation(
        self, agent_id: str, state: State
    ) -> str | None:
        """
        Get initial observation for an agent at episode start.

        Override this to provide agent-specific initial observations.
        By default returns None (agent sees only their system prompt).

        Args:
            agent_id: The agent to get observation for.
            state: Current rollout state.

        Returns:
            Initial observation string, or None for no observation.
        """
        return None

    async def get_agent_observation(
        self, agent_id: str, response: Messages, state: State
    ) -> str | None:
        """
        Get observation for an agent after another agent acts.

        Override this to implement agent-to-agent communication.
        By default returns None (agents are independent).

        Args:
            agent_id: The observing agent.
            response: The response from the acting agent(s).
            state: Current rollout state.

        Returns:
            Observation string, or None for no observation.
        """
        return None

    async def check_episode_done(self, state: State) -> bool:
        """
        Check if the episode should end (beyond standard stop conditions).

        Override this to implement custom termination logic.
        By default returns False (rely on standard stop conditions).

        Args:
            state: Current rollout state.

        Returns:
            True if episode should end, False otherwise.
        """
        return False

    # -------------------------------------------------------------------------
    # Stop conditions
    # -------------------------------------------------------------------------

    @vf.stop
    async def episode_done(self, state: State) -> bool:
        """Check custom episode termination."""
        return await self.check_episode_done(state)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _get_agent_response(
        self,
        agent: Agent,
        state: State,
    ) -> tuple[Messages, ModelResponse]:
        """
        Get model response for an agent.

        Uses agent's client/model/sampling_args if set, otherwise falls back
        to state defaults. If the agent has a lora_id, includes it in the
        request for vLLM multi-LoRA routing.
        """
        prompt = agent.get_prompt()

        # Resolve client/model/sampling_args
        client = agent.client or state["client"]
        model = agent.model or state["model"]
        sampling_args = {**state.get("sampling_args", {}), **agent.sampling_args}

        # Add LoRA request for multi-LoRA vLLM serving
        if agent.lora_id is not None:
            extra_body = sampling_args.get("extra_body", {})
            extra_body["lora_request"] = {
                "lora_name": f"lora_{agent.lora_id}",
                "lora_int_id": agent.lora_id,
            }
            sampling_args["extra_body"] = extra_body

        response = await self.get_model_response(
            state=state,
            prompt=prompt,
            client=client,
            model=model,
            sampling_args=sampling_args,
        )

        completion = await parse_response_messages(response, self.message_type)
        return completion, response

    async def _agent_turn(
        self,
        agent_id: str,
        state: State,
    ) -> TrajectoryStep:
        """
        Execute a single turn for an agent.

        Returns the trajectory step for this turn.
        """
        agent = self.agents[agent_id]
        prompt = agent.get_prompt()

        # Get model response
        completion, response = await self._get_agent_response(agent, state)

        # Record response in agent's context
        agent.record_response(completion)

        # Parse tokens if needed
        tokens = await parse_response_tokens(
            response, self.message_type, self.max_seq_len
        )
        is_truncated = await parse_is_truncated(response, self.message_type)
        if tokens is not None and tokens.get("is_truncated"):
            is_truncated = True

        # Create trajectory step
        step = TrajectoryStep(
            prompt=prompt,
            completion=completion,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=is_truncated,
            trajectory_id=state["trajectory_id"],
            extras={"agent_id": agent_id},
        )

        return step

    # -------------------------------------------------------------------------
    # State and rollout management
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """
        Initialize multi-agent specific state.

        Sets up:
        - Per-agent state dicts
        - TraceCollector for trajectory collection
        - Turn counter
        """
        state = await super().setup_state(state)

        # Initialize per-agent state
        state["agents"] = {}
        for agent_id, agent in self.agents.items():
            state["agents"][agent_id] = {
                "agent_id": agent_id,
                "reward": None,
                "metrics": {},
                "trajectory": [],
            }

        # Create trace collector with global metadata
        state["collector"] = TraceCollector(
            example_id=state.get("example_id"),
            task=state.get("task"),
        )

        # Register agents in collector
        for agent_id, agent in self.agents.items():
            state["collector"].register_agent(
                agent_id=agent_id,
                name=agent.name,
                trainable=agent.trainable,
                lora_id=agent.lora_id,
            )

        # Turn counter
        state["current_turn"] = 0

        # Reset all agents
        for agent_id, agent in self.agents.items():
            agent.reset()
            obs = await self.get_initial_observation(agent_id, state)
            if obs is not None:
                agent.on_env_reset(obs, state.get("info", {}))

        return state

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Messages:
        """
        Not used directly in multi-agent mode.

        MultiAgentEnv handles agent turns differently - this method exists
        for compatibility with the base class interface.
        """
        return []

    async def render_completion(self, state: State) -> None:
        """
        Render final completion from all agents' trajectories.

        Concatenates all agent completions into state["completion"].
        """
        collector: TraceCollector = state["collector"]
        all_completions: list[Any] = []

        for agent_id in self.agent_ids:
            steps = collector.get_steps(agent_id)
            for step in steps:
                step_completion = step.get("completion", [])
                if isinstance(step_completion, list):
                    all_completions.extend(step_completion)
                elif step_completion:
                    all_completions.append(step_completion)

        state["completion"] = all_completions

    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Execute a multi-agent rollout.

        This is the main rollout loop for multi-agent environments.
        It orchestrates turn-taking, trajectory collection, and state management.
        """
        state = await self.init_state(input, client, model, sampling_args)

        try:
            try:
                state = await self.setup_state(state)
            except vf.Error as e:
                state["error"] = e

            collector: TraceCollector = state["collector"]

            # Main rollout loop
            while not await self.is_completed(state):
                try:
                    # Get agents that act this turn
                    active_agents = await self.get_active_agents(state)

                    if self.turn_order == "parallel":
                        # All active agents act simultaneously
                        turn_tasks = [
                            self._agent_turn(aid, state) for aid in active_agents
                        ]
                        steps = await asyncio.gather(*turn_tasks)

                        # Record steps and notify agents of observations
                        for aid, step in zip(active_agents, steps):
                            collector.add(aid, step)
                            state["trajectory"].append(step)
                            state["agents"][aid]["trajectory"].append(step)

                        # Notify all agents of observations (for inter-agent comms)
                        for aid in self.agent_ids:
                            for acting_aid, step in zip(active_agents, steps):
                                if acting_aid != aid:
                                    obs = await self.get_agent_observation(
                                        aid, step["completion"], state
                                    )
                                    if obs is not None:
                                        self.agents[aid].on_after_step(obs, {})

                    else:
                        # Sequential: one agent per turn
                        for aid in active_agents:
                            step = await self._agent_turn(aid, state)
                            collector.add(aid, step)
                            state["trajectory"].append(step)
                            state["agents"][aid]["trajectory"].append(step)

                            # Notify other agents
                            for other_aid in self.agent_ids:
                                if other_aid != aid:
                                    obs = await self.get_agent_observation(
                                        other_aid, step["completion"], state
                                    )
                                    if obs is not None:
                                        self.agents[other_aid].on_after_step(obs, {})

                    state["current_turn"] += 1

                except vf.Error as e:
                    if isinstance(e, vf.OverlongPromptError):
                        state["prompt_too_long"] = True
                        state["is_truncated"] = True
                    else:
                        state["error"] = e

            # Render final completion
            await self.render_completion(state)

            # Store per-agent rollouts for training access
            state["agent_rollouts"] = collector.extract_rollouts()

            return state

        except asyncio.CancelledError:
            await self._cleanup(state)
            raise

    # -------------------------------------------------------------------------
    # Training data access
    # -------------------------------------------------------------------------

    def get_agent_rollouts(self, state: State) -> list[dict[str, Any]]:
        """
        Extract per-agent rollouts from a completed state.

        Args:
            state: Completed rollout state.

        Returns:
            List of per-agent rollout dicts from the TraceCollector.
        """
        return state.get("agent_rollouts", [])

    def get_trainable_rollouts(self, state: State) -> list[dict[str, Any]]:
        """
        Extract rollouts only for trainable agents.

        Args:
            state: Completed rollout state.

        Returns:
            List of rollout dicts for trainable agents only.
        """
        all_rollouts = self.get_agent_rollouts(state)
        return [r for r in all_rollouts if r.get("meta", {}).get("trainable", True)]
