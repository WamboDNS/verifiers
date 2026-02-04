"""Context strategies for managing agent conversation history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from verifiers.types import ChatMessage, Messages


class ContextStrategy(ABC):
    """
    Abstract base class for managing an agent's conversation context.

    A context strategy determines how messages are accumulated and presented
    to the model during a rollout. Different strategies can implement different
    windowing, summarization, or filtering behaviors.

    Subclasses must implement the lifecycle hooks:
    - on_env_reset: Called when the environment resets
    - on_before_act: Called to get messages for model inference
    - on_after_act: Called after the model responds
    - on_after_step: Called after the environment responds
    """

    def __init__(self, system_prompt: str | None = None):
        """
        Initialize the context strategy.

        Args:
            system_prompt: Optional system prompt to prepend to the conversation.
        """
        self._messages: list[ChatMessage] = []
        self._default_system_prompt = system_prompt
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def reset(self, system_prompt: str | None = None) -> None:
        """
        Reset the context to initial state.

        Args:
            system_prompt: Optional override for the system prompt.
                If None, uses the default system prompt from __init__.
        """
        self._messages = []
        prompt = system_prompt or self._default_system_prompt
        if prompt:
            self._messages.append({"role": "system", "content": prompt})

    @property
    def messages(self) -> list[ChatMessage]:
        """Return a copy of the current message history."""
        return list(self._messages)

    @abstractmethod
    def on_env_reset(self, obs: str, info: dict[str, Any]) -> None:
        """
        Called when the environment resets with initial observation.

        Args:
            obs: Initial observation from the environment.
            info: Additional info from the environment reset.
        """
        ...

    @abstractmethod
    def on_before_act(self) -> Messages:
        """
        Called before the agent acts to get the prompt for the model.

        Returns:
            Messages to send to the model for inference.
        """
        ...

    @abstractmethod
    def on_after_act(self, response: Messages) -> None:
        """
        Called after the model responds to record the response.

        Args:
            response: The model's response messages.
        """
        ...

    @abstractmethod
    def on_after_step(self, obs: str, info: dict[str, Any]) -> None:
        """
        Called after the environment step with new observation.

        Args:
            obs: Observation from the environment.
            info: Additional info from the environment step.
        """
        ...


class FullDialogContext(ContextStrategy):
    """
    Context strategy that maintains the full conversation history.

    This is the default strategy: all messages are accumulated and sent
    to the model on each turn. No summarization or windowing is applied.
    """

    def on_env_reset(self, obs: str, info: dict[str, Any]) -> None:
        """Add initial observation as user message if non-empty."""
        if obs:
            self._messages.append({"role": "user", "content": obs})

    def on_before_act(self) -> Messages:
        """Return the full message history for inference."""
        return list(self._messages)

    def on_after_act(self, response: Messages) -> None:
        """Append model response to history."""
        if isinstance(response, list):
            self._messages.extend(response)
        elif isinstance(response, dict):
            self._messages.append(response)
        else:
            raise TypeError(
                f"response must be list or dict, got {type(response).__name__}"
            )

    def on_after_step(self, obs: str, info: dict[str, Any]) -> None:
        """Add environment observation as user message if non-empty."""
        if obs:
            self._messages.append({"role": "user", "content": obs})


class SlidingWindowContext(ContextStrategy):
    """
    Context strategy that maintains a sliding window of recent messages.

    Keeps only the most recent `window_size` messages (excluding the system
    prompt, which is always retained).
    """

    def __init__(self, system_prompt: str | None = None, window_size: int = 10):
        """
        Initialize sliding window context.

        Args:
            system_prompt: Optional system prompt to prepend.
            window_size: Maximum number of non-system messages to retain.
        """
        super().__init__(system_prompt=system_prompt)
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size

    def _trim_to_window(self) -> None:
        """Trim messages to window size, preserving system prompt."""
        has_system = (
            len(self._messages) > 0 and self._messages[0].get("role") == "system"
        )
        if has_system:
            non_system = self._messages[1:]
            if len(non_system) > self._window_size:
                self._messages = [self._messages[0]] + non_system[-self._window_size :]
        else:
            if len(self._messages) > self._window_size:
                self._messages = self._messages[-self._window_size :]

    def on_env_reset(self, obs: str, info: dict[str, Any]) -> None:
        """Add initial observation as user message if non-empty."""
        if obs:
            self._messages.append({"role": "user", "content": obs})
        self._trim_to_window()

    def on_before_act(self) -> Messages:
        """Return the windowed message history for inference."""
        return list(self._messages)

    def on_after_act(self, response: Messages) -> None:
        """Append model response and trim to window."""
        if isinstance(response, list):
            self._messages.extend(response)
        elif isinstance(response, dict):
            self._messages.append(response)
        else:
            raise TypeError(
                f"response must be list or dict, got {type(response).__name__}"
            )
        self._trim_to_window()

    def on_after_step(self, obs: str, info: dict[str, Any]) -> None:
        """Add environment observation and trim to window."""
        if obs:
            self._messages.append({"role": "user", "content": obs})
        self._trim_to_window()
