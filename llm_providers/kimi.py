"""Kimi K2.5 provider using Moonshot AI's OpenAI-compatible API."""

from __future__ import annotations

from typing import Generator

from openai import OpenAI

from .base import LLMMessage, LLMProvider, LLMResponse


_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
_DEFAULT_MODEL = "kimi-k2.5"


class KimiProvider(LLMProvider):
    """Moonshot Kimi K2.5 implementation.

    Uses the OpenAI-compatible endpoint at api.moonshot.ai.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        thinking: bool = False,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._thinking = thinking

    # ── public API ───────────────────────────────────────────────

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        payload = self._build_payload(messages, temperature=temperature, max_tokens=max_tokens)
        resp = self._client.chat.completions.create(**payload, stream=False)
        choice = resp.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
            finish_reason=choice.finish_reason or "",
        )

    def chat_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int = 8192,
    ) -> Generator[str, None, None]:
        payload = self._build_payload(messages, temperature=temperature, max_tokens=max_tokens)
        stream = self._client.chat.completions.create(**payload, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    def name(self) -> str:
        return f"Kimi ({self._model})"

    # ── internals ────────────────────────────────────────────────

    def _build_payload(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None,
        max_tokens: int,
    ) -> dict:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        # Kimi-specific params (like thinking) must go through extra_body
        # because the OpenAI SDK doesn't recognise them as top-level args.
        extra: dict = {}
        if self._thinking:
            extra["thinking"] = {"type": "enabled"}
        else:
            extra["thinking"] = {"type": "disabled"}

        payload: dict = {
            "model": self._model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "extra_body": extra,
        }

        return payload
