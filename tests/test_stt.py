"""Tests for STT engine abstraction."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.config import FasterWhisperConfig, STTConfig
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


@patch("vox.stt.faster_whisper_engine.WhisperModel")
def test_transcribe_passes_new_params(mock_whisper_model_cls):
    """New params (hotwords, repetition_penalty, patience) are passed to transcribe()."""
    from vox.stt.faster_whisper_engine import FasterWhisperEngine

    config = FasterWhisperConfig(
        hotwords="Claude Code, GitHub",
        repetition_penalty=1.1,
        patience=2.0,
        initial_prompt="Claude Code。プログラミングに関する音声入力。",
        beam_size=5,
    )

    mock_model = MagicMock()
    mock_info = MagicMock()
    mock_info.language = "ja"
    mock_info.language_probability = 0.99
    mock_segment = MagicMock()
    mock_segment.text = "テスト"
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    mock_whisper_model_cls.return_value = mock_model

    engine = FasterWhisperEngine(config)
    engine.load_model()

    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, 16000)

    assert result == "テスト"
    call_kwargs = mock_model.transcribe.call_args[1]
    assert call_kwargs["hotwords"] == "Claude Code, GitHub"
    assert call_kwargs["repetition_penalty"] == 1.1
    assert call_kwargs["patience"] == 2.0
    assert call_kwargs["initial_prompt"] == "Claude Code。プログラミングに関する音声入力。"
    assert call_kwargs["beam_size"] == 5
