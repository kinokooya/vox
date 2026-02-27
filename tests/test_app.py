"""Tests for VoxApp pipeline orchestration."""

from unittest.mock import MagicMock, patch

import numpy as np

from vox.config import AppConfig


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_llm_timeout_falls_back_to_raw_text(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """When LLM raises an exception, the pipeline should fall back to raw STT text."""
    from vox.app import VoxApp

    config = AppConfig()

    # Set up mock STT engine (text must be >20 chars to avoid LLM skip)
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "えーとこれはテスト用の長い音声認識テキストです"
    mock_stt_factory.return_value = mock_stt

    # Set up mock recorder that returns valid audio
    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = np.ones(16000, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    # Set up mock LLM that raises an exception (simulating timeout)
    mock_llm = MagicMock()
    mock_llm.format_text.side_effect = Exception("Request timed out")
    mock_llm_cls.return_value = mock_llm

    # Set up mock inserter
    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)

    # Run the pipeline directly
    app._process_pipeline()

    # LLM was called with the raw text
    mock_llm.format_text.assert_called_once_with("えーとこれはテスト用の長い音声認識テキストです")

    # Inserter was called with the raw text as fallback
    mock_inserter.insert.assert_called_once_with("えーとこれはテスト用の長い音声認識テキストです")


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_llm_success_inserts_formatted_text(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """When LLM succeeds, the pipeline should insert the formatted text."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "えーと raw text"
    mock_stt_factory.return_value = mock_stt

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = np.ones(16000, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_llm = MagicMock()
    mock_llm.format_text.return_value = "formatted text"
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)
    app._process_pipeline()

    mock_inserter.insert.assert_called_once_with("formatted text")


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_empty_audio_skips_processing(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """When recorder returns empty audio, pipeline should skip entirely."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_stt = MagicMock()
    mock_stt_factory.return_value = mock_stt

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = np.array([], dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)
    app._process_pipeline()

    # STT and LLM should not be called
    mock_stt.transcribe.assert_not_called()
    mock_llm.format_text.assert_not_called()
    mock_inserter.insert.assert_not_called()
