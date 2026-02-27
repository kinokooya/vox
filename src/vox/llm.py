"""LLM text formatting via OpenAI-compatible API (Ollama / LM Studio)."""

from __future__ import annotations

import logging
import re

from openai import OpenAI

from vox.config import LLMConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
音声認識テキストを整形するアシスタントです。整形後のテキストのみを出力してください。

ルール:
1. フィラー（「えーと」「あのー」「まあ」「えー」「うーん」「あー」「えっと」等）を除去
2. 言い間違い・繰り返し・言い直しを修正し、最終的な意図のみ残す
3. 適切な句読点（。、）を追加
4. 文法的な誤りを修正
5. 技術用語は正しい表記にする（例: リアクト→React、ドッカー→Docker）
6. 元の意味を変えない。情報を追加しない
7. 入力が短い場合（単語や短いフレーズ）はそのまま返す
8. 出力は整形済みテキストのみ"""


class LLMFormatter:
    """Formats raw STT text using a local LLM via OpenAI-compatible API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key="not-needed",  # Local LLM doesn't require API key
            timeout=config.timeout_sec,
        )

    def format_text(self, raw_text: str) -> str:
        """Send raw STT text to LLM for formatting.

        Args:
            raw_text: Raw transcription from STT engine.

        Returns:
            Formatted text string.
        """
        if not raw_text.strip():
            return ""

        logger.info("LLM formatting: input=%d chars", len(raw_text))

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        result = response.choices[0].message.content or ""
        result = result.strip()
        result = self._normalize_output(result)
        logger.info("LLM formatting: output=%d chars", len(result))
        return result

    def _normalize_output(self, text: str) -> str:
        """Normalize output based on output_format config."""
        if self._config.output_format == "single_line":
            text = text.replace("\n", " ")
            text = re.sub(r" {2,}", " ", text)
            text = text.strip()
        return text
