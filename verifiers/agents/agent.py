"""Agent abstraction for multi-agent environments."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from verifiers.agents.context import ContextStrategy, FullDialogContext
from verifiers.types import ChatMessage, Messages, SamplingArgs

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@dataclass
class AgentConfig:
    """
    Configuration for an agent in a multi-agent environment.

    This dataclass holds all configuration needed to instantiate an Agent.
    Use Agent.from_config() to create an Agent from this config.

    Attributes:
        agent_id: Unique identifier for this agent within the environment.
        name: Human-readable name for logging. Defaults to agent_id.
        system_prompt: Optional system prompt for the agent's context.
        sampling_args: Optional sampling arguments (temperature, etc.).
        client: Optional OpenAI client. If None, uses environment's client.
        model: Optional model name. If None, uses environment's model.
        trainable: Whether this agent's weights should be trained.
            If False, the agent uses an external/frozen model.
        lora_id: LoRA adapter identifier for grouped training.
            Agents with the same lora_id share weights during training.
            If None and trainable=True, trains the base model directly.
    """

    agent_id: str
    name: str | None = None
    system_prompt: str | None = None
    sampling_args: SamplingArgs | None = None
    client: "AsyncOpenAI | None" = None
    model: str | None = None
    trainable: bool = True
    lora_id: int | None = None


class Agent:
    """
    An agent in a multi-agent environment.

    The Agent class provides:
    - Identity (agent_id, name)
    - Context management via a ContextStrategy
    - Configuration for training (trainable, lora_id)
    - Optional per-agent client/model/sampling overrides

    The Agent does NOT make API calls directly - that is handled by the
    environment. The Agent's responsibility is to manage its conversation
    context and provide prompts for the environment to use.

    Example:
        ```python
        # Create an agent with full dialog context
        agent = Agent(
            agent_id="player1",
            name="Player One",
            system_prompt="You are playing a game.",
            trainable=True,
            lora_id=0,
        )

        # Reset for a new episode
        agent.reset()

        # Get initial observation from environment
        agent.on_env_reset(obs="Game started.", info={})

        # Get prompt for model inference
        prompt = agent.get_prompt()  # Returns messages for API call

        # Record model response
        agent.record_response([{"role": "assistant", "content": "I'll start."}])

        # Record next observation
        agent.on_after_step(obs="Your turn.", info={})
        ```
    """

    def __init__(
        self,
        agent_id: str,
        name: str | None = None,
        system_prompt: str | None = None,
        sampling_args: SamplingArgs | None = None,
        client: "AsyncOpenAI | None" = None,
        model: str | None = None,
        trainable: bool = True,
        lora_id: int | None = None,
        context: ContextStrategy | None = None,
    ):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier for this agent.
            name: Human-readable name. Defaults to agent_id.
            system_prompt: Optional system prompt for the agent.
            sampling_args: Optional per-agent sampling arguments.
            client: Optional per-agent OpenAI client.
            model: Optional per-agent model name.
            trainable: Whether this agent should be trained.
            lora_id: LoRA adapter ID for grouped training.
            context: Context strategy for managing conversation history.
                Defaults to FullDialogContext.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")

        self.agent_id = agent_id
        self.name = name or agent_id
        self.system_prompt = system_prompt
        self.sampling_args = sampling_args or {}
        self.client = client
        self.model = model
        self.trainable = trainable
        self.lora_id = lora_id
        self._ctx = context or FullDialogContext(system_prompt=system_prompt)

    def reset(self, system_prompt: str | None = None) -> None:
        """
        Reset the agent's context for a new episode.

        Args:
            system_prompt: Optional override for the system prompt.
                If None, uses the agent's default system prompt.
        """
        prompt = system_prompt or self.system_prompt
        self._ctx.reset(system_prompt=prompt)

    def on_env_reset(self, obs: str, info: dict[str, Any]) -> None:
        """
        Handle environment reset with initial observation.

        Args:
            obs: Initial observation from the environment.
            info: Additional info from the environment.
        """
        self._ctx.on_env_reset(obs, info)

    def get_prompt(self) -> Messages:
        """
        Get the current prompt for model inference.

        Returns:
            Messages to send to the model.
        """
        return self._ctx.on_before_act()

    def record_response(self, response: Messages) -> None:
        """
        Record the model's response in the context.

        Args:
            response: The model's response messages.
        """
        self._ctx.on_after_act(response)

    def on_after_step(self, obs: str, info: dict[str, Any]) -> None:
        """
        Handle environment step result.

        Args:
            obs: Observation from the environment.
            info: Additional info from the environment.
        """
        self._ctx.on_after_step(obs, info)

    @property
    def messages(self) -> list[ChatMessage]:
        """Get the current conversation history."""
        return self._ctx.messages

    @property
    def context(self) -> ContextStrategy:
        """Get the underlying context strategy."""
        return self._ctx

    @classmethod
    def from_config(
        cls, config: AgentConfig, context: ContextStrategy | None = None
    ) -> "Agent":
        """
        Create an Agent from an AgentConfig.

        Args:
            config: The agent configuration.
            context: Optional context strategy override.

        Returns:
            A new Agent instance.
        """
        return cls(
            agent_id=config.agent_id,
            name=config.name,
            system_prompt=config.system_prompt,
            sampling_args=config.sampling_args,
            client=config.client,
            model=config.model,
            trainable=config.trainable,
            lora_id=config.lora_id,
            context=context,
        )

    def __deepcopy__(self, memo: dict) -> "Agent":
        """Deep-copy the agent but preserve client by reference.

        AsyncOpenAI instances hold connection pools and internal state that
        do not survive deep copy. The client is intentionally shared.
        """
        return Agent(
            agent_id=self.agent_id,
            name=self.name,
            system_prompt=self.system_prompt,
            sampling_args=copy.deepcopy(self.sampling_args, memo),
            client=self.client,  # shared, not deep-copied
            model=self.model,
            trainable=self.trainable,
            lora_id=self.lora_id,
            context=copy.deepcopy(self._ctx, memo),
        )

    def __repr__(self) -> str:
        return (
            f"Agent(agent_id={self.agent_id!r}, name={self.name!r}, "
            f"trainable={self.trainable}, lora_id={self.lora_id})"
        )
