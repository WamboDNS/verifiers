"""Trace collector for per-agent trajectory tracking in multi-agent environments."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

from verifiers.types import TrajectoryStep


class TraceCollector:
    """
    Collects per-agent trajectory steps during multi-agent rollouts.

    The TraceCollector tracks which agent produced which trajectory steps,
    enabling per-agent rollout extraction for training. Each agent's steps
    are stored separately, and metadata (trainable, lora_id, etc.) is
    preserved for use during training.

    Example:
        ```python
        collector = TraceCollector(episode_id="ep_001")

        # Register agents with their training metadata
        collector.register_agent(
            agent_id="player1",
            name="Player One",
            trainable=True,
            lora_id=0,
        )
        collector.register_agent(
            agent_id="player2",
            name="Player Two",
            trainable=True,
            lora_id=1,
        )

        # During rollout, add steps for each agent
        collector.add("player1", step1)
        collector.add("player2", step2)

        # Extract per-agent rollouts for training
        rollouts = collector.extract_rollouts()
        # Returns list of dicts, one per agent with non-empty traces
        ```
    """

    def __init__(self, **global_meta: Any):
        """
        Initialize the trace collector.

        Args:
            **global_meta: Global metadata to attach to all extracted rollouts.
                Common examples: episode_id, task, example_id.
        """
        self._traces: dict[str, list[TrajectoryStep]] = defaultdict(list)
        self._agent_meta: dict[str, dict[str, Any]] = {}
        self._global_meta = global_meta

    def register_agent(
        self,
        agent_id: str,
        name: str | None = None,
        trainable: bool = True,
        lora_id: int | None = None,
        **extra_meta: Any,
    ) -> None:
        """
        Register an agent with its training metadata.

        This should be called for each agent before adding any steps.
        The metadata will be included in the extracted rollouts.

        Args:
            agent_id: Unique identifier for the agent.
            name: Human-readable name for logging.
            trainable: Whether this agent should be trained.
            lora_id: LoRA adapter ID for grouped training.
            **extra_meta: Additional metadata to attach to this agent's rollouts.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")

        self._agent_meta[agent_id] = {
            "agent_id": agent_id,
            "agent_name": name or agent_id,
            "trainable": trainable,
            "lora_id": lora_id,
            **extra_meta,
        }

    def add(self, agent_id: str, step: TrajectoryStep) -> None:
        """
        Add a trajectory step for an agent.

        Args:
            agent_id: The agent that produced this step.
            step: The trajectory step to record.

        Raises:
            ValueError: If agent_id is empty.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        self._traces[agent_id].append(step)

    def extract_rollouts(self) -> list[dict[str, Any]]:
        """
        Extract per-agent rollouts for training.

        Returns a list of rollout dicts, one per agent that has recorded steps.
        Each rollout contains:
        - id: Unique rollout ID
        - agent_id: The agent identifier
        - steps: List of TrajectoryStep dicts
        - trajectory: Alias for steps (for compatibility)
        - completion: Concatenated completion messages from all steps
        - is_truncated: Whether the last step was truncated
        - total_reward: Sum of rewards from all steps
        - meta: Combined global and agent metadata

        Returns:
            List of rollout dicts for agents with non-empty traces.
        """
        rollouts = []
        for agent_id, steps in self._traces.items():
            if not steps:
                continue

            agent_meta = self._agent_meta.get(
                agent_id, {"agent_id": agent_id, "agent_name": agent_id}
            )

            # Extract completion messages from all steps
            completion: list[Any] = []
            for step in steps:
                step_completion = step.get("completion", [])
                if isinstance(step_completion, list):
                    completion.extend(step_completion)
                elif step_completion:
                    completion.append(step_completion)

            # Determine if truncated from last step
            last_step = steps[-1]
            is_truncated = last_step.get("is_truncated", False)

            # Sum rewards (treating None as 0)
            total_reward = sum(
                step.get("reward", 0.0) or 0.0 for step in steps
            )

            rollout = {
                "id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "steps": steps,
                "trajectory": steps,  # Alias for compatibility
                "completion": completion,
                "is_truncated": is_truncated,
                "total_reward": total_reward,
                "meta": {**self._global_meta, **agent_meta},
            }
            rollouts.append(rollout)

        return rollouts

    def extract_trainable_rollouts(self) -> list[dict[str, Any]]:
        """
        Extract rollouts only for trainable agents.

        Returns:
            List of rollout dicts for trainable agents with non-empty traces.
        """
        all_rollouts = self.extract_rollouts()
        return [r for r in all_rollouts if r["meta"].get("trainable", True)]

    def get_lora_groups(self) -> dict[int | None, list[str]]:
        """
        Group agents by their LoRA adapter ID.

        Returns:
            Dict mapping lora_id to list of agent_ids sharing that adapter.
            Agents with lora_id=None are grouped under the None key.
        """
        groups: dict[int | None, list[str]] = {}
        for agent_id, meta in self._agent_meta.items():
            if not meta.get("trainable", True):
                continue
            lora_id = meta.get("lora_id")
            if lora_id not in groups:
                groups[lora_id] = []
            groups[lora_id].append(agent_id)
        return groups

    def get_agent_ids(self) -> list[str]:
        """
        Get list of all agent IDs that have recorded steps.

        Returns:
            List of agent IDs with non-empty traces.
        """
        return [aid for aid, steps in self._traces.items() if steps]

    def get_registered_agent_ids(self) -> list[str]:
        """
        Get list of all registered agent IDs.

        Returns:
            List of all registered agent IDs, even if they have no steps.
        """
        return list(self._agent_meta.keys())

    def get_steps(self, agent_id: str) -> list[TrajectoryStep]:
        """
        Get steps for a specific agent.

        Args:
            agent_id: The agent to get steps for.

        Returns:
            Copy of the agent's trajectory steps.
        """
        return list(self._traces.get(agent_id, []))

    def get_agent_meta(self, agent_id: str) -> dict[str, Any]:
        """
        Get metadata for a specific agent.

        Args:
            agent_id: The agent to get metadata for.

        Returns:
            Copy of the agent's metadata, or empty dict if not registered.
        """
        return dict(self._agent_meta.get(agent_id, {}))

    def clear(self) -> None:
        """Clear all recorded traces but keep agent registrations."""
        self._traces.clear()

    def clear_all(self) -> None:
        """Clear all traces and agent registrations."""
        self._traces.clear()
        self._agent_meta.clear()

    def __len__(self) -> int:
        """Return total number of steps across all agents."""
        return sum(len(steps) for steps in self._traces.values())

    def __repr__(self) -> str:
        agent_counts = {aid: len(steps) for aid, steps in self._traces.items() if steps}
        return f"TraceCollector(agents={agent_counts}, global_meta={self._global_meta})"
