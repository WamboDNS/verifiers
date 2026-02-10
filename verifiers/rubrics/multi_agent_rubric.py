"""Multi-agent rubric for per-agent scoring in multi-agent environments."""

from __future__ import annotations

import time
from typing import Any

from verifiers.rubrics.rubric import Rubric
from verifiers.types import State


class MultiAgentRubric(Rubric):
    """
    Rubric that scores each agent independently with per-agent rubrics.

    MultiAgentRubric allows assigning different rubrics to different agents,
    enabling agent-specific reward functions and metrics. Each agent is
    scored using its own rubric, and rewards/metrics are stored per-agent
    in state["agents"][agent_id]["reward"] and state["agents"][agent_id]["metrics"].
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
                Non-trainable agents without a rubric get reward=0.0.
                Trainable agents without a rubric raise an error.
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

        # Start from a shallow copy of the full state so that env-specific
        # fields (e.g. secret_leaked, turn_count) are visible to reward
        # functions, then override agent-specific fields.
        view = State(state)
        view["completion"] = agent_completion
        view["trajectory"] = agent_trajectory
        view["agent_id"] = agent_id
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
        if "agents" not in state:
            raise ValueError(
                "MultiAgentRubric.score_rollout() requires a multi-agent state "
                "(state must have 'agents' key). Use a standard Rubric for "
                "single-agent environments."
            )

        start_time = time.time()
        total_reward = 0.0
        all_metrics: dict[str, float] = {}

        for agent_id, agent_state in state["agents"].items():
            rubric = self.agent_rubrics.get(agent_id)

            if rubric is None:
                agent_obj = state.get("_agent_instances", {}).get(agent_id)
                if agent_obj is None or agent_obj.trainable:
                    raise ValueError(f"No rubric registered for trainable agent {agent_id!r}")
                agent_state["reward"] = 0.0
                agent_state["metrics"] = {}
                for rollout in state.get("agent_rollouts", []):
                    if rollout.get("meta", {}).get("agent_id") == agent_id:
                        rollout["total_reward"] = 0.0
                continue

            # Create view and score
            agent_view = self._create_agent_view(state, agent_id)
            await rubric.score_rollout(agent_view)

            # Extract results
            agent_reward = agent_view.get("reward", 0.0) or 0.0
            agent_metrics = agent_view.get("metrics", {}) or {}

            # Store in agent state
            agent_state["reward"] = agent_reward
            agent_state["metrics"] = agent_metrics

            # Update agent_rollouts if present
            agent_rollouts = state.get("agent_rollouts", [])
            for rollout in agent_rollouts:
                meta = rollout.get("meta", {})
                if meta.get("agent_id") == agent_id:
                    rollout["total_reward"] = agent_reward
                    meta["agent_reward"] = agent_reward
                    rollout["meta"] = meta

            # Accumulate totals
            total_reward += agent_reward

            # Prefix metrics with agent_id
            for metric_name, metric_value in agent_metrics.items():
                prefixed_name = f"{agent_id}/{metric_name}"
                all_metrics[prefixed_name] = metric_value

        # Set overall state values
        state["reward"] = total_reward
        state["metrics"] = all_metrics

        # Record scoring timing (matches base Rubric.score_rollout behavior)
        end_time = time.time()
        timing = state.get("timing")
        if timing is not None:
            timing["scoring_ms"] = (end_time - start_time) * 1000
            timing["total_ms"] += timing["scoring_ms"]

    async def score_group(self, states: list[State]) -> None:
        """
        Score a group of rollouts.

        For multi-agent states, scores each state independently using
        score_rollout. Group-level reward functions in per-agent rubrics
        are not currently supported.

        Args:
            states: List of rollout states to score.
        """
        # Score each state independently
        # Group scoring across agents is complex and not commonly needed
        for state in states:
            await self.score_rollout(state)

        # Compute advantages relative to group mean
        rewards = [state.get("reward", 0.0) or 0.0 for state in states]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Compute per-agent means
        agent_ids: set[str] = set()
        for state in states:
            agent_ids.update(state.get("agents", {}).keys())
        agent_means: dict[str, float] = {}
        for agent_id in agent_ids:
            agent_rewards = [
                s.get("agents", {}).get(agent_id, {}).get("reward", 0.0) or 0.0
                for s in states
            ]
            agent_means[agent_id] = (
                sum(agent_rewards) / len(agent_rewards) if agent_rewards else 0.0
            )

        for state, reward in zip(states, rewards):
            state["advantage"] = reward - avg_reward

            # Propagate rewards/advantages to shared trajectory steps
            for t in state.get("trajectory", []):
                if t["advantage"] is None:
                    t["advantage"] = state["advantage"]
                if t["reward"] is None:
                    t["reward"] = state["reward"]

            # Set per-agent advantages and propagate to per-agent trajectory steps
            for agent_id, agent_state in state.get("agents", {}).items():
                agent_reward = agent_state.get("reward", 0.0) or 0.0
                agent_state["advantage"] = agent_reward - agent_means.get(agent_id, 0.0)

                for t in agent_state.get("trajectory", []):
                    if t["advantage"] is None:
                        t["advantage"] = agent_state["advantage"]
                    if t["reward"] is None:
                        t["reward"] = agent_state["reward"]
