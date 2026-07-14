from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

logger = logging.getLogger(__name__)

_RATE_LIMIT_BASE_WAIT = 15.0
_RATE_LIMIT_MAX_ATTEMPTS = 8


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str | list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    scratchpad: str = ""


class ModelAdapter(Protocol):
    async def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


def _extract_text(content: Any) -> str | None:
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        joined = "\n".join(p for p in parts if p)
        if joined.strip():
            return joined
    return None


class OpenAIModelAdapter:
    """异步 OpenAI 兼容适配器，内置 429 指数退避重试。"""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        thinking_budget: int = 8000,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=600.0,
            max_retries=0,
        )

    def _extra_body(self) -> dict[str, Any] | None:
        if self.enable_thinking:
            return {"enable_thinking": True, "thinking_budget": self.thinking_budget}
        return None

    async def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        payload = [{"role": m.role, "content": m.content} for m in messages]
        extra = self._extra_body()

        for attempt in range(1, _RATE_LIMIT_MAX_ATTEMPTS + 1):
            try:
                # Some providers (e.g. OpenRouter reasoning models) occasionally
                # spend the whole budget on hidden reasoning and return empty
                # content. Bump temperature slightly on retries to break that.
                temperature = self.temperature
                if attempt > 1:
                    temperature = min(1.0, self.temperature + 0.2 * (attempt - 1))
                kw: dict[str, Any] = dict(
                    model=self.model,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                if extra:
                    kw["extra_body"] = extra

                response = await self._client.chat.completions.create(**kw)
                choices = response.choices or []
                if not choices:
                    raise RuntimeError("Model response missing choices.")
                message = choices[0].message
                text = _extract_text(message.content)
                if text is None:
                    # Fall back to a `reasoning` field if the provider exposes one.
                    text = _extract_text(getattr(message, "reasoning", None))
                if text is None:
                    # Empty content is transient: retry instead of crashing the task.
                    if attempt < _RATE_LIMIT_MAX_ATTEMPTS:
                        logger.warning(
                            "Empty model content (attempt %d/%d). Retrying...",
                            attempt, _RATE_LIMIT_MAX_ATTEMPTS,
                        )
                        await asyncio.sleep(2.0 * attempt)
                        continue
                    raise RuntimeError("Model response missing text content.")
                return text

            except (APIConnectionError, APITimeoutError) as exc:
                # Transient network failures (dropped connection, read timeout).
                # NOTE: APIConnectionError has no `status_code`, so it must be
                # handled before the generic APIError branch below.
                if attempt < _RATE_LIMIT_MAX_ATTEMPTS:
                    wait = _RATE_LIMIT_BASE_WAIT * attempt
                    logger.warning(
                        "Connection error %s (attempt %d/%d). Waiting %.0fs...",
                        type(exc).__name__, attempt, _RATE_LIMIT_MAX_ATTEMPTS, wait,
                    )
                    print(
                        f"Connection error {type(exc).__name__} "
                        f"(attempt {attempt}/{_RATE_LIMIT_MAX_ATTEMPTS}). "
                        f"Waiting {wait:.0f}s...",
                        flush=True,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Model request failed (connection): {exc}") from exc

            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = status_code == 429 or (status_code is not None and status_code >= 500)
                if retryable and attempt < _RATE_LIMIT_MAX_ATTEMPTS:
                    wait = _RATE_LIMIT_BASE_WAIT * attempt
                    logger.warning(
                        "Retryable API error %s (attempt %d/%d). Waiting %.0fs...",
                        status_code, attempt, _RATE_LIMIT_MAX_ATTEMPTS, wait,
                    )
                    print(
                        f"Retryable API error {status_code} "
                        f"(attempt {attempt}/{_RATE_LIMIT_MAX_ATTEMPTS}). "
                        f"Waiting {wait:.0f}s...",
                        flush=True,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Model request failed: {exc}") from exc

        raise RuntimeError("All retry attempts exhausted.")


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
