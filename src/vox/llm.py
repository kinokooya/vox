"""LLM text formatting via OpenAI-compatible API (Ollama / LM Studio)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from openai import OpenAI

from vox.config import LLMConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
あなたは音声認識テキストを整形するアシスタントです。
入力されたテキストを以下のルールに従って整形し、整形後のテキストのみを出力してください。

ルール:
1. フィラー（「えーと」「あのー」「まあ」「えー」「うーん」等）を完全に除去する
2. 言い間違い・繰り返し・言い直しを修正する
3. しどろもどろな発話から意図を汲み取り、明確な文章に再構成する
4. 適切な句読点（。、）を追加する
5. 文法的な誤りを修正する
6. 技術用語・英語用語は正しい表記にする（例: リアクト -> React）
7. 元の意味・意図は変えない。情報を追加しない
8. 整形済みテキストのみを出力する（説明や注釈は一切不要）"""


class LLMError(RuntimeError):
    """Base error for LLM formatting failures."""


class LLMTransientError(LLMError):
    """Retryable LLM error (timeouts, temporary connection failures)."""


class LLMPermanentError(LLMError):
    """Non-retryable LLM error."""


def _is_transient_error(error: Exception) -> bool:
    name = error.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "connection", "rate"))


class LLMFormatter:
    """Formats raw STT text using a local LLM via OpenAI-compatible API."""

    def __init__(
        self,
        config: LLMConfig,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._client = client or OpenAI(
            base_url=config.base_url,
            api_key="not-needed",  # Local LLM doesn't require API key
        )
        self._sleep = sleep_fn

    def format_text(self, raw_text: str) -> str:
        """Send raw STT text to LLM for formatting.

        Args:
            raw_text: Raw transcription from STT engine.

        Returns:
            Formatted text string.

        Raises:
            LLMTransientError: Temporary LLM failure after retries.
            LLMPermanentError: Non-retryable LLM failure.
        """
        if not raw_text.strip():
            return ""

        logger.info("LLM formatting: input=%d chars", len(raw_text))
        max_attempts = self._config.retry_count + 1

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": raw_text},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    timeout=self._config.request_timeout_sec,
                )

                result = (response.choices[0].message.content or "").strip()
                logger.info("LLM formatting: output=%d chars", len(result))
                return result
            except Exception as err:  # noqa: BLE001
                retryable = _is_transient_error(err)
                has_next = attempt < max_attempts
                if retryable and has_next:
                    wait = self._config.retry_backoff_sec * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM transient error (attempt %d/%d): %s; retry in %.2fs",
                        attempt,
                        max_attempts,
                        err,
                        wait,
                    )
                    if wait > 0:
                        self._sleep(wait)
                    continue

                if retryable:
                    logger.error("LLM transient error exhausted retries: %s", err)
                    raise LLMTransientError("LLM request failed after retries") from err

                logger.error("LLM permanent error: %s", err)
                raise LLMPermanentError("LLM request failed") from err

        raise LLMTransientError("LLM request failed after retries")
