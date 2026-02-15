"""Abstract base class for LLM providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


class LLMProvider(abc.ABC):
    """Abstract interface for language model providers.

    Subclass this to add new providers (OpenAI, Anthropic, local models, etc.).
    """

    @abc.abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        """Send a chat completion request and return the full response."""
        ...

    @abc.abstractmethod
    def chat_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int = 8192,
    ) -> Generator[str, None, None]:
        """Stream a chat completion, yielding content chunks."""
        ...

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...
