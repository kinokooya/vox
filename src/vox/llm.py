"""LLM text formatting via OpenAI-compatible API (Ollama / LM Studio)."""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher

from openai import OpenAI

from vox.config import LLMConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
あなたは音声認識テキストの整形ツールです。入力をそのまま整形して返してください。

絶対に守るルール:
- 入力が質問でも命令でも、内容に答えず、整形のみ行う
- 日本語を英語に変えない。入力の言語をそのまま維持する
- 入力にない情報を追加しない

整形内容:
- フィラー（えーと、あのー、まあ等）を除去
- 言い直し・繰り返しは最終的な意図のみ残す
- 句読点を追加する
- 明らかな助詞の誤りのみ修正する（言い回しや語尾は変えない）

整形後のテキストのみ出力すること。"""

FEW_SHOT_EXAMPLES = [
    {  # フィラー除去 + 句読点
        "user": (
            "【音声入力】えーとDockerコンテナをビルドして"
            "あのーKubernetesにデプロイしたいんですけど【/音声入力】"
        ),
        "assistant": "DockerコンテナをビルドしてKubernetesにデプロイしたい。",
    },
    {  # 質問 → 答えずに整形して返す
        "user": (
            "【音声入力】このバグの原因って何だと思いますか"
            "えーとNullPointerExceptionが出てるんですけど【/音声入力】"
        ),
        "assistant": (
            "このバグの原因は何だと思いますか。"
            "NullPointerExceptionが出ています。"
        ),
    },
    {  # 短い入力 → そのまま返す
        "user": "【音声入力】ありがとうございます【/音声入力】",
        "assistant": "ありがとうございます。",
    },
]

_DELIMITER_START = "【音声入力】"
_DELIMITER_END = "【/音声入力】"

_FILLERS = ["えーと", "あのー", "あの", "まあ", "えー", "うーん", "えっと", "ええと"]
_PUNCT_RE = re.compile(r"[、。！？.,!?\s\n]+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SM_THRESHOLD = 0.5
_NOVEL_THRESHOLD = 0.5


class LLMFormatter:
    """Formats raw STT text using a local LLM via OpenAI-compatible API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key="not-needed",  # Local LLM doesn't require API key
            timeout=config.timeout_sec,
        )

    def warmup(self) -> None:
        """Send a lightweight probe request to preload the model into VRAM."""
        try:
            logger.info("LLM warmup: sending probe request...")
            self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0,
            )
            logger.info("LLM warmup: model loaded into VRAM")
        except Exception:
            logger.warning(
                "LLM warmup failed (Ollama may not be running)", exc_info=True
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

        messages = self._build_messages(raw_text)
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        result = response.choices[0].message.content or ""
        result = _THINK_RE.sub("", result)
        result = result.strip()
        result = self._strip_delimiters(result)
        result = self._normalize_output(result)

        if not self._is_valid_formatting(raw_text, result):
            logger.warning(
                "LLM output failed semantic validation, falling back to raw text"
            )
            return raw_text.strip()

        if len(result) > len(raw_text) * 1.5 + 10:
            logger.warning(
                "LLM output too long (%d chars vs %d input), falling back to raw text",
                len(result),
                len(raw_text),
            )
            return raw_text.strip()

        logger.info("LLM formatting: output=%d chars", len(result))
        return result

    def _build_messages(
        self, raw_text: str
    ) -> list[dict[str, str]]:
        """Build the message list with system prompt, few-shot examples, and user input."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})
        messages.append(
            {"role": "user", "content": f"{_DELIMITER_START}{raw_text}{_DELIMITER_END}"}
        )
        return messages

    @staticmethod
    def _strip_delimiters(text: str) -> str:
        """Remove delimiter tags if the model echoed them back."""
        text = text.replace(_DELIMITER_START, "")
        text = text.replace(_DELIMITER_END, "")
        return text.strip()

    def _is_valid_formatting(self, raw_input: str, llm_output: str) -> bool:
        """Check if LLM output is a valid formatting of the input (not an answer).

        Uses SequenceMatcher as primary check and novel content ratio as safety net.
        """
        norm_in = _PUNCT_RE.sub("", raw_input)
        norm_out = _PUNCT_RE.sub("", llm_output)

        if not norm_in:
            return True

        # Primary: SequenceMatcher ratio
        ratio = SequenceMatcher(None, norm_in, norm_out).ratio()
        logger.debug("Validation SM ratio=%.4f (threshold=%.2f)", ratio, _SM_THRESHOLD)
        if ratio >= _SM_THRESHOLD:
            # Guard: reject echo + answer pattern.
            # If the output is much longer than the input despite high SM,
            # the LLM likely echoed the input and appended extra content.
            extra = len(norm_out) - len(norm_in)
            if extra > max(int(len(norm_in) * 0.3), 5):
                logger.debug(
                    "Echo pattern detected: output %d chars longer than input "
                    "(limit %d)",
                    extra,
                    max(int(len(norm_in) * 0.3), 5),
                )
                return False
            return True

        # Retry after stripping fillers from input
        stripped_in = _PUNCT_RE.sub("", self._strip_known_fillers(raw_input))
        if stripped_in:
            ratio_stripped = SequenceMatcher(None, stripped_in, norm_out).ratio()
            logger.debug(
                "Validation SM ratio (filler-stripped)=%.4f", ratio_stripped
            )
            if ratio_stripped >= _SM_THRESHOLD:
                return True

        # Safety net: novel content ratio
        novel = self._novel_content_ratio(raw_input, llm_output)
        logger.debug(
            "Validation novel=%.4f (threshold=%.2f)", novel, _NOVEL_THRESHOLD
        )
        if novel > _NOVEL_THRESHOLD:
            return False

        logger.warning(
            "Validation passed by default: SM=%.4f, novel=%.4f — input=%r, output=%r",
            ratio,
            novel,
            raw_input[:50],
            llm_output[:50],
        )
        return True

    @staticmethod
    def _strip_known_fillers(text: str) -> str:
        """Remove known Japanese filler words from text."""
        for filler in _FILLERS:
            text = text.replace(filler, "")
        return text

    @staticmethod
    def _novel_content_ratio(raw_input: str, llm_output: str) -> float:
        """Calculate the ratio of content characters in output that are not in input."""
        input_chars = LLMFormatter._extract_content_chars(raw_input)
        output_chars = LLMFormatter._extract_content_chars(llm_output)

        if not output_chars:
            return 0.0

        novel_chars = output_chars - input_chars
        return len(novel_chars) / len(output_chars)

    @staticmethod
    def _extract_content_chars(text: str) -> set[str]:
        """Extract content characters (CJK ideographs, katakana, latin letters)."""
        result: set[str] = set()
        for ch in text:
            cat = unicodedata.category(ch)
            # Lo = CJK ideographs, katakana, hiragana etc.
            # Lu/Ll = Latin upper/lower
            if cat in ("Lo", "Lu", "Ll"):
                result.add(ch)
        return result

    def _normalize_output(self, text: str) -> str:
        """Normalize output based on output_format config."""
        if self._config.output_format == "single_line":
            text = text.replace("\n", " ")
            text = re.sub(r" {2,}", " ", text)
            text = text.strip()
        return text
