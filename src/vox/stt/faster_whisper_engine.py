"""FasterWhisper STT engine implementation."""

from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from vox.config import FasterWhisperConfig
from vox.stt.base import STTEngine

logger = logging.getLogger(__name__)


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

        segments, info = self._model.transcribe(
            audio,
            language=self._config.language,
            vad_filter=True,
            vad_parameters={
                "min_speech_duration_ms": self._config.vad.min_speech_duration_ms,
                "min_silence_duration_ms": self._config.vad.min_silence_duration_ms,
            },
        )

        text = "".join(segment.text for segment in segments).strip()
        prob = info.language_probability
        logger.info("STT result (lang=%s, prob=%.2f): %s", info.language, prob, text)
        return text

    def get_vram_usage_mb(self) -> int:
        return 4000  # ~4GB for large-v3-turbo float16
