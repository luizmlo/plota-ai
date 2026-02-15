"""Google AI (Gemini) provider.

Uses the google-genai SDK (https://github.com/googleapis/python-genai).
API key from: https://aistudio.google.com/apikey
"""

from __future__ import annotations

import logging
from typing import Generator

from google import genai
from google.genai import types

from .base import LLMMessage, LLMProvider, LLMResponse

log = logging.getLogger(__name__)


_DEFAULT_MODEL = "gemini-2.0-flash-001"

_SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
]


class GoogleAIProvider(LLMProvider):
    """Google AI (Gemini) implementation.

    Uses the google-genai SDK. API key from:
    https://aistudio.google.com/apikey
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_id = model

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int = 65536,
    ) -> LLMResponse:
        contents = self._messages_to_contents(messages)
        config = self._build_config(temperature, max_tokens)
        prompt_len = 0
        for c in contents:
            if c.parts:
                p = c.parts[0]
                t = getattr(p, "text", None)
                if t is not None:
                    prompt_len += len(t)
        log.debug("chat: model=%s messages=%s contents_count=%s prompt_len=%s", self._model_id, len(messages), len(contents), prompt_len)
        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=config,
            )
        except Exception as e:
            log.exception("chat: generate_content failed: %s", e)
            raise
        text = response.text or ""
        usage = self._usage_from_response(response)
        log.debug("chat: response len=%s usage=%s", len(text), usage)
        return LLMResponse(content=text, model=self._model_id, usage=usage, finish_reason="stop")

    def chat_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int = 65536,
    ) -> Generator[str, None, None]:
        contents = self._messages_to_contents(messages)
        config = self._build_config(temperature, max_tokens)
        log.debug(
            "chat_stream: model=%s messages=%s contents_count=%s",
            self._model_id,
            len(messages),
            len(contents),
        )
        chunk_count = 0
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self._model_id,
                contents=contents,
                config=config,
            ):
                chunk_count += 1
                if chunk_count == 1:
                    log.debug("chat_stream: first chunk type=%s has_text=%s", type(chunk).__name__, bool(getattr(chunk, "text", None)))
                text = getattr(chunk, "text", None)
                if not text and getattr(chunk, "candidates", None):
                    # Fallback: extract text from candidates[0].content.parts
                    try:
                        c0 = chunk.candidates[0]
                        content = getattr(c0, "content", None)
                        parts = getattr(content, "parts", None) or []
                        for p in parts:
                            t = getattr(p, "text", None)
                            if t:
                                text = (text or "") + t
                    except (IndexError, AttributeError, TypeError) as e:
                        log.debug("chat_stream: chunk %s no text, candidates fallback failed: %s", chunk_count, e)
                if text:
                    yield text
                elif chunk_count <= 3 or chunk_count % 50 == 0:
                    log.debug("chat_stream: chunk %s empty text, chunk=%s", chunk_count, chunk)
        except Exception as e:
            log.exception("chat_stream: failed after %s chunks: %s", chunk_count, e)
            raise
        log.debug("chat_stream: done chunks=%s", chunk_count)

    def name(self) -> str:
        return f"Google AI ({self._model_id})"

    def _build_config(
        self,
        temperature: float | None,
        max_tokens: int,
    ) -> types.GenerateContentConfig:
        max_out = max(max_tokens, 32768)
        kwargs: dict = {"max_output_tokens": max_out, "safety_settings": _SAFETY_SETTINGS}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return types.GenerateContentConfig(**kwargs)

    def _messages_to_contents(self, messages: list[LLMMessage]) -> list[types.Content]:
        """Convert LLMMessages to a list of genai Content for generate_content.
        System message is prepended to the first user message.
        If only system message(s) are present, they are sent as a single user content
        so the API receives valid contents (Gemini requires at least one user turn).
        """
        contents: list[types.Content] = []
        system_parts: list[str] = []

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                text = "\n\n".join(system_parts + [m.content]) if system_parts else m.content
                system_parts = []
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=text)],
                    )
                )
            elif m.role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=m.content)],
                    )
                )

        # API requires at least one user content; system-only is used as the prompt
        if not contents and system_parts:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="\n\n".join(system_parts))],
                )
            )
        return contents

    def _usage_from_response(self, response: types.GenerateContentResponse) -> dict:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            usage["prompt_tokens"] = getattr(um, "prompt_token_count", 0) or 0
            usage["completion_tokens"] = getattr(um, "candidates_token_count", 0) or getattr(um, "total_token_count", 0) or 0
        return usage
