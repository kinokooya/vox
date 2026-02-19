"""STT engine factory."""

from __future__ import annotations

from vox.config import STTConfig
from vox.stt.base import STTEngine


def create_stt_engine(config: STTConfig) -> STTEngine:
    """Create an STT engine based on configuration."""
    if config.engine == "faster-whisper":
        from vox.stt.faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine(config.faster_whisper)
    elif config.engine == "sensevoice":
        raise NotImplementedError("SenseVoice engine will be implemented in Phase 2")
    else:
        raise ValueError(f"Unknown STT engine: {config.engine}")
