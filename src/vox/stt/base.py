"""STT engine abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STTEngine(ABC):
    """Abstract base class for Speech-to-Text engines."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into GPU memory."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio to text. VAD is handled internally by the engine.

        Args:
            audio: Audio samples as float32 numpy array.
            sample_rate: Sample rate in Hz (expected: 16000).

        Returns:
            Transcribed text string.
        """
        ...

    @abstractmethod
    def get_vram_usage_mb(self) -> int:
        """Return estimated VRAM usage in megabytes."""
        ...
