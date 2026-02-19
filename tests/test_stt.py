"""Tests for STT engine abstraction."""

import numpy as np
import pytest

from vox.config import STTConfig
from vox.stt.base import STTEngine
from vox.stt.factory import create_stt_engine


class MockSTTEngine(STTEngine):
    """Mock STT engine for testing the interface."""

    def __init__(self) -> None:
        self._loaded = False

    def load_model(self) -> None:
        self._loaded = True

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        return "test transcription"

    def get_vram_usage_mb(self) -> int:
        return 100


def test_mock_engine_interface():
    engine = MockSTTEngine()
    engine.load_model()
    result = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert result == "test transcription"
    assert engine.get_vram_usage_mb() == 100


def test_mock_engine_not_loaded():
    engine = MockSTTEngine()
    with pytest.raises(RuntimeError):
        engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)


def test_factory_unknown_engine():
    config = STTConfig(engine="unknown")
    with pytest.raises(ValueError, match="Unknown STT engine"):
        create_stt_engine(config)


def test_factory_sensevoice_not_implemented():
    config = STTConfig(engine="sensevoice")
    with pytest.raises(NotImplementedError):
        create_stt_engine(config)
