"""Multi-agent rubric for per-agent scoring in multi-agent environments."""

from __future__ import annotations

from typing import Any

from verifiers.rubrics.rubric import Rubric
from verifiers.types import State


class MultiAgentRubric(Rubric):
    """
    Rubric that scores each agent independently with per-agent rubrics.

    MultiAgentRubric allows assigning different rubrics to different agents,
    enabling agent-specific reward functions and metrics. Each agent is
    scored using its own rubric, and rewards/metrics are stored per-agent.

    Example:
        ```python
        # Create rubrics for different agents
        player_rubric = vf.Rubric(funcs=[win_reward])
        referee_rubric = vf.Rubric(funcs=[fairness_reward])

        # Create multi-agent rubric with per-agent scoring
        multi_rubric = MultiAgentRubric(
            agent_rubrics={
                "player1": player_rubric,
                "player2": player_rubric,  # Can share rubrics
                "referee": referee_rubric,
            }
        )

        # Use in environment
        env = MyMultiAgentEnv(
            agents=agents,
            rubric=multi_rubric,
        )
        ```

    Scoring behavior:
    - If state has "agents" dict (multi-agent state), scores each agent
      with its registered rubric
    - If no agent rubric is registered, that agent gets reward=0.0
    - If state doesn't have "agents" dict, falls back to base Rubric scoring

    Per-agent rewards and metrics are stored in state["agents"][agent_id]:
        state["agents"]["player1"]["reward"] = 1.0
        state["agents"]["player1"]["metrics"] = {"win_reward": 1.0}
    """

    def __init__(
        self,
        agent_rubrics: dict[str, Rubric] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the multi-agent rubric.

        Args:
            agent_rubrics: Dict mapping agent_id to Rubric for per-agent scoring.
                Agents not in this dict will receive reward=0.0.
            **kwargs: Additional arguments passed to base Rubric.
        """
        super().__init__(**kwargs)
        self.agent_rubrics: dict[str, Rubric] = agent_rubrics or {}

    def add_agent_rubric(self, agent_id: str, rubric: Rubric) -> None:
        """
        Register a rubric for an agent.

        Args:
            agent_id: The agent identifier.
            rubric: The rubric to use for scoring this agent.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        self.agent_rubrics[agent_id] = rubric

    def remove_agent_rubric(self, agent_id: str) -> Rubric | None:
        """
        Remove and return a rubric for an agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            The removed rubric, or None if not found.
        """
        return self.agent_rubrics.pop(agent_id, None)

    def get_agent_rubric(self, agent_id: str) -> Rubric | None:
        """
        Get the rubric for an agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            The agent's rubric, or None if not registered.
        """
        return self.agent_rubrics.get(agent_id)

    def _create_agent_view(self, state: State, agent_id: str) -> State:
        """
        Create a State view from a specific agent's perspective.

        This creates a temporary State object that the agent's rubric
        can score. It includes:
        - The agent's trajectory as "trajectory"
        - The agent's completion messages as "completion"
        - Original state fields (prompt, answer, info, etc.)
        - Timing info

        Args:
            state: The full multi-agent state.
            agent_id: The agent to create a view for.

        Returns:
            A State object from the agent's perspective.
        """
        agent_state_data = state.get("agents", {}).get(agent_id, {})
        agent_trajectory = agent_state_data.get("trajectory", [])

        # Build agent's completion from their trajectory
        agent_completion: list[Any] = []
        for step in agent_trajectory:
            step_completion = step.get("completion", [])
            if isinstance(step_completion, list):
                agent_completion.extend(step_completion)
            elif step_completion:
                agent_completion.append(step_completion)

        # Create view state
        view = State()
        view["input"] = state.get("input", {})
        view["prompt"] = state.get("prompt")
        view["completion"] = agent_completion
        view["trajectory"] = agent_trajectory
        view["answer"] = state.get("answer", "")
        view["task"] = state.get("task", "")
        view["info"] = state.get("info", {})
        view["example_id"] = state.get("example_id")
        view["agent_id"] = agent_id
        view["is_completed"] = state.get("is_completed", False)
        view["is_truncated"] = state.get("is_truncated", False)
        view["timing"] = state.get("timing", {})
        view["reward"] = None
        view["metrics"] = {}

        return view

    async def score_rollout(self, state: State) -> None:
        """
        Score a rollout, handling both single-agent and multi-agent states.

        For multi-agent states (state["agents"] exists):
        - Scores each agent independently with their registered rubric
        - Stores per-agent rewards in state["agents"][agent_id]["reward"]
        - Stores per-agent metrics in state["agents"][agent_id]["metrics"]
        - Sets state["reward"] to sum of all agent rewards
        - Merges all metrics with agent_id prefix

        For single-agent states:
        - Falls back to base Rubric.score_rollout()

        Args:
            state: The rollout state to score.
        """
        # Check if this is a multi-agent state
        if "agents" not in state:
            # Fall back to base rubric scoring
            await super().score_rollout(state)
            return

        total_reward = 0.0
        all_metrics: dict[str, float] = {}

        for agent_id, agent_state in state["agents"].items():
            rubric = self.agent_rubrics.get(agent_id)

            if rubric is None:
                # No rubric registered for this agent
                agent_state["reward"] = 0.0
                agent_state["metrics"] = {}
                continue

            # Create view and score
            agent_view = self._create_agent_view(state, agent_id)
            await rubric.score_rollout(agent_view)

            # Extract results from scored view
            agent_reward = agent_view["reward"]
            agent_metrics = agent_view["metrics"]

            # Store in agent state
            agent_state["reward"] = agent_reward
            agent_state["metrics"] = agent_metrics

            # Accumulate totals
            total_reward += agent_reward

            # Prefix metrics with agent_id
            for metric_name, metric_value in agent_metrics.items():
                prefixed_name = f"{agent_id}/{metric_name}"
                all_metrics[prefixed_name] = metric_value

        # Set overall state values
        state["reward"] = total_reward
        state["metrics"] = all_metrics

    async def score_group(self, states: list[State]) -> None:
        """
        Score a group of rollouts.

        For multi-agent states, scores each state independently using
        score_rollout. Group-level reward functions in per-agent rubrics
        are not currently supported.

        Args:
            states: List of rollout states to score.
        """
        # Check if any state is multi-agent
        has_multi_agent = any("agents" in state for state in states)

        if not has_multi_agent:
            # All single-agent: use base implementation
            await super().score_group(states)
            return

        # Multi-agent: score each state independently
        # Group scoring across agents is complex and not commonly needed
        for state in states:
            await self.score_rollout(state)

        # Compute advantages relative to group mean
        rewards = [state["reward"] for state in states]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        for state, reward in zip(states, rewards):
            state["advantage"] = reward - avg_reward

            # Also set per-agent advantages if multi-agent
            if "agents" in state:
                for agent_id, agent_state in state["agents"].items():
                    # Per-agent advantage relative to group mean
                    agent_state["advantage"] = agent_state["reward"] - avg_reward
