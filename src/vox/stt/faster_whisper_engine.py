"""FasterWhisper STT engine implementation."""

from __future__ import annotations

import logging
import re

import numpy as np
from faster_whisper import WhisperModel

from vox.config import FasterWhisperConfig
from vox.stt.base import STTEngine

logger = logging.getLogger(__name__)

# Known hallucination patterns (YouTube outros, repeated phrases, etc.)
_HALLUCINATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in [
        r"ご視聴ありがとうございました",
        r"チャンネル登録",
        r"高評価",
        r"ご清聴ありがとうございました",
        r"字幕",
        r"次の動画",
        r"ご覧いただき",
        r"最後までご覧",
    ]
]

# Repetition pattern: same phrase repeated 3+ times
_REPETITION_RE = re.compile(r"(.{2,}?)\1{2,}")

# Simplified Chinese characters NOT used in Japanese.
# Excludes chars shared with Japanese: 着会没当与云参双号叶国据担机条灯点礼随
_SIMPLIFIED_CHINESE_CHARS = set(
    "这们对进过还该让从虽认谢说请问关开连运远选边达总办图书样东两为"
    "时应发业动产专乐义习乡亲买亚仅价传伟伤众优华单卖卫厂变"
    "响员团园围坏块处备复够头夺奖导层币师帮广庆归录忆态怀护报拥择"
    "换标权极构档桥检毕气汇汉济测热灵烦爱环现确离种积稳穷竞笔"
    "类级纪约经绍结给统续综绿网罗职联脑脸节药获虑补观规觉计订记设证评识词"
    "译详语课调谁谈账质贡财责败货贸费资赛赞轮转车轻输辑农适递遗释针钱铁银"
    "错键阅队阳阶险际隐难预领频风飞饭馆验鱼鸡龙"
)


class FasterWhisperEngine(STTEngine):
    """STT engine using faster-whisper (CTranslate2)."""

    def __init__(self, config: FasterWhisperConfig) -> None:
        self._config = config
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        logger.info(
            "Loading faster-whisper model: %s (device=%s, compute=%s)",
            self._config.model,
            self._config.device,
            self._config.compute_type,
        )
        self._model = WhisperModel(
            self._config.model,
            device=self._config.device,
            compute_type=self._config.compute_type,
        )
        logger.info("faster-whisper model loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if sample_rate != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sample_rate}Hz")

        duration = len(audio) / sample_rate

        segments, info = self._model.transcribe(
            audio,
            language=self._config.language,
            beam_size=self._config.beam_size,
            condition_on_previous_text=self._config.condition_on_previous_text,
            no_speech_threshold=self._config.no_speech_threshold,
            log_prob_threshold=self._config.log_prob_threshold,
            compression_ratio_threshold=self._config.compression_ratio_threshold,
            hallucination_silence_threshold=self._config.hallucination_silence_threshold,
            word_timestamps=self._config.hallucination_silence_threshold is not None,
            initial_prompt=self._config.initial_prompt,
            hotwords=self._config.hotwords,
            repetition_penalty=self._config.repetition_penalty,
            patience=self._config.patience,
            vad_filter=True,
            vad_parameters={
                "min_speech_duration_ms": self._config.vad.min_speech_duration_ms,
                "min_silence_duration_ms": self._config.vad.min_silence_duration_ms,
            },
        )

        text = "".join(segment.text for segment in segments).strip()
        prob = info.language_probability
        logger.info("STT result (lang=%s, prob=%.2f): %s", info.language, prob, text)

        text = self._validate_transcription(text, duration)
        return text

    def _validate_transcription(self, text: str, duration: float) -> str:
        """Post-process validation to filter hallucinations and Chinese output."""
        if not text:
            return ""

        # Check for simplified Chinese characters
        if self._contains_simplified_chinese(text):
            logger.warning(
                "Simplified Chinese detected, discarding: %s", text
            )
            return ""

        # Check character-to-duration ratio (short audio producing long text)
        if duration > 0:
            chars_per_sec = len(text) / duration
            # Japanese speech is typically 5-10 chars/sec; >15 is suspicious
            if duration < 3.0 and chars_per_sec > 15.0:
                logger.warning(
                    "Suspicious char/sec ratio (%.1f chars in %.1fs = %.1f c/s), "
                    "discarding: %s",
                    len(text),
                    duration,
                    chars_per_sec,
                    text,
                )
                return ""

        # Check known hallucination patterns
        for pattern in _HALLUCINATION_PATTERNS:
            if pattern.search(text):
                logger.warning(
                    "Hallucination pattern detected, discarding: %s", text
                )
                return ""

        # Check for excessive repetition
        if _REPETITION_RE.search(text):
            logger.warning("Repetition detected, discarding: %s", text)
            return ""

        return text

    def _contains_simplified_chinese(self, text: str) -> bool:
        """Check if text contains simplified Chinese characters."""
        return any(ch in _SIMPLIFIED_CHINESE_CHARS for ch in text)

    def get_vram_usage_mb(self) -> int:
        return 4000  # ~4GB for large-v3-turbo float16
