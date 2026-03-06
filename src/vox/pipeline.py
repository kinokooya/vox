"""Pipeline execution for audio -> STT -> LLM -> text insertion."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class RecorderProtocol(Protocol):
    def stop(self) -> np.ndarray: ...


class STTProtocol(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class LLMProtocol(Protocol):
    def format_text(self, raw_text: str) -> str: ...


class InserterProtocol(Protocol):
    def insert(self, text: str) -> None: ...


class PipelineRunner:
    """Runs one end-to-end processing pipeline for a recorded utterance."""

    def __init__(
        self,
        recorder: RecorderProtocol,
        stt: STTProtocol,
        llm: LLMProtocol,
        inserter: InserterProtocol,
        sample_rate: int,
    ) -> None:
        self._recorder = recorder
        self._stt = stt
        self._llm = llm
        self._inserter = inserter
        self._sample_rate = sample_rate

    def run_once(self) -> bool:
        """Execute one full pipeline run.

        Returns:
            True if text was inserted, False if pipeline ended early.
        """
        audio = self._recorder.stop()
        if len(audio) == 0:
            logger.warning("No audio captured, skipping pipeline")
            return False

        duration = len(audio) / self._sample_rate
        logger.info("[Pipeline] Audio: %.1fs, %d samples", duration, len(audio))

        logger.info("[Pipeline] Running STT...")
        raw_text = self._stt.transcribe(audio, self._sample_rate)
        if not raw_text.strip():
            logger.info("[Pipeline] STT returned empty text, skipping")
            return False
        logger.info("[Pipeline] STT result: %s", raw_text)

        logger.info("[Pipeline] Running LLM formatting...")
        formatted_text = self._llm.format_text(raw_text)
        if not formatted_text.strip():
            logger.info("[Pipeline] LLM returned empty text, skipping")
            return False
        logger.info("[Pipeline] LLM result: %s", formatted_text)

        logger.info("[Pipeline] Inserting text...")
        self._inserter.insert(formatted_text)
        logger.info("[Pipeline] Done")
        return True
