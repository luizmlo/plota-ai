"""Google AI (Gemini) provider."""

from __future__ import annotations

from typing import Generator

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from .base import LLMMessage, LLMProvider, LLMResponse


_DEFAULT_MODEL = "gemini-3-flash-preview"

_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


class GoogleAIProvider(LLMProvider):
    """Google AI (Gemini) implementation.

    Uses the google-generativeai SDK. API key from:
    https://aistudio.google.com/apikey
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        genai.configure(api_key=api_key)
        self._model_id = model
        self._model = genai.GenerativeModel(model)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int = 65536,
    ) -> LLMResponse:
        history, prompt = self._prepare_chat(messages)
        config = self._build_config(temperature, max_tokens)

        if history:
            chat = self._model.start_chat(history=history)
            response = chat.send_message(
                prompt,
                generation_config=config,
                safety_settings=_SAFETY_SETTINGS,
                stream=False,
            )
        else:
            response = self._model.generate_content(
                prompt,
                generation_config=config,
                safety_settings=_SAFETY_SETTINGS,
                stream=False,
            )

        text = response.text or ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if response.usage_metadata:
            usage["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
            usage["completion_tokens"] = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(content=text, model=self._model_id, usage=usage, finish_reason="stop")

    def chat_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int = 65536,
    ) -> Generator[str, None, None]:
        history, prompt = self._prepare_chat(messages)
        config = self._build_config(temperature, max_tokens)

        if history:
            chat = self._model.start_chat(history=history)
            response = chat.send_message(
                prompt,
                generation_config=config,
                safety_settings=_SAFETY_SETTINGS,
                stream=True,
            )
        else:
            response = self._model.generate_content(
                prompt,
                generation_config=config,
                safety_settings=_SAFETY_SETTINGS,
                stream=True,
            )

        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def name(self) -> str:
        return f"Google AI ({self._model_id})"

    def _build_config(
        self,
        temperature: float | None,
        max_tokens: int,
    ) -> genai.types.GenerationConfig:
        # Use at least 32k output; Gemini 3 Flash supports up to 64k
        max_out = max(max_tokens, 32768)
        kwargs: dict = {"max_output_tokens": max_out}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return genai.types.GenerationConfig(**kwargs)

    def _prepare_chat(
        self,
        messages: list[LLMMessage],
    ) -> tuple[list, str]:
        """Convert messages to Gemini history + final prompt.
        Returns (history, prompt) where prompt is the last user message.
        """
        if not messages:
            return [], ""

        last = messages[-1]
        if last.role != "user":
            return [], last.content  # fallback

        prompt = last.content
        prev = messages[:-1]
        if not prev:
            return [], prompt

        history = self._messages_to_genai_history(prev)
        return history, prompt

    def _messages_to_genai_history(
        self,
        messages: list[LLMMessage],
    ) -> list:
        """Convert LLMMessages to Gemini Content objects.
        System message is prepended to the first user message.
        """
        history: list = []
        system_parts: list[str] = []
        Part = genai.protos.Part
        Content = genai.protos.Content

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                text = "\n\n".join(system_parts + [m.content]) if system_parts else m.content
                system_parts = []
                history.append(Content(role="user", parts=[Part(text=text)]))
            elif m.role == "assistant":
                history.append(Content(role="model", parts=[Part(text=m.content)]))

        return history
