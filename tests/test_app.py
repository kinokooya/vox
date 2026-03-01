"""Tests for VoxApp pipeline orchestration."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.config import AppConfig, LLMConfig, STTConfig


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


# --- start() parallel loading tests ---


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_start_loads_stt_and_warms_llm(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """start() should call load_model() and warmup() in parallel."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_stt = MagicMock()
    mock_stt_factory.return_value = mock_stt

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    mock_recorder = MagicMock()
    mock_recorder_cls.return_value = mock_recorder

    mock_hotkey = MagicMock()
    mock_hotkey_cls.return_value = mock_hotkey

    app = VoxApp(config)
    app.start()

    mock_stt.load_model.assert_called_once()
    mock_stt.get_vram_usage_mb.assert_called_once()
    mock_llm.warmup.assert_called_once()
    mock_recorder.open.assert_called_once()
    mock_hotkey.start.assert_called_once()


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_start_stt_failure_propagates(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """When STT load_model() fails, start() should raise and not open recorder/hotkey."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_stt = MagicMock()
    mock_stt.load_model.side_effect = RuntimeError("CUDA out of memory")
    mock_stt_factory.return_value = mock_stt

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    mock_recorder = MagicMock()
    mock_recorder_cls.return_value = mock_recorder

    mock_hotkey = MagicMock()
    mock_hotkey_cls.return_value = mock_hotkey

    app = VoxApp(config)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        app.start()

    mock_recorder.open.assert_not_called()
    mock_hotkey.start.assert_not_called()


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_start_llm_warmup_failure_does_not_crash(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """When LLM warmup() fails, start() should still succeed."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_stt = MagicMock()
    mock_stt_factory.return_value = mock_stt

    mock_llm = MagicMock()
    mock_llm.warmup.side_effect = ConnectionError("Ollama not running")
    mock_llm_cls.return_value = mock_llm

    mock_recorder = MagicMock()
    mock_recorder_cls.return_value = mock_recorder

    mock_hotkey = MagicMock()
    mock_hotkey_cls.return_value = mock_hotkey

    app = VoxApp(config)
    app.start()

    mock_stt.load_model.assert_called_once()
    mock_recorder.open.assert_called_once()
    mock_hotkey.start.assert_called_once()


# --- Word replacement tests ---


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_apply_word_replacements_unit(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """_apply_word_replacements should replace all configured words."""
    from vox.app import VoxApp

    config = AppConfig(
        stt=STTConfig(word_replacements={"クロードコード": "Claude Code", "ギットハブ": "GitHub"})
    )
    mock_stt_factory.return_value = MagicMock()
    mock_llm_cls.return_value = MagicMock()
    mock_inserter_cls.return_value = MagicMock()

    app = VoxApp(config)
    result = app._apply_word_replacements("クロードコードを使ってギットハブにプッシュ")
    assert result == "Claude Codeを使ってGitHubにプッシュ"


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_apply_word_replacements_no_match(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """When no replacements match, text is returned unchanged."""
    from vox.app import VoxApp

    config = AppConfig(
        stt=STTConfig(word_replacements={"クロードコード": "Claude Code"})
    )
    mock_stt_factory.return_value = MagicMock()
    mock_llm_cls.return_value = MagicMock()
    mock_inserter_cls.return_value = MagicMock()

    app = VoxApp(config)
    assert app._apply_word_replacements("テスト文章です") == "テスト文章です"


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_applies_word_replacements(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """Pipeline should apply word replacements to STT output before LLM."""
    from vox.app import VoxApp

    config = AppConfig(
        stt=STTConfig(word_replacements={"クロードコード": "Claude Code"}),
    )

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "クロードコードを使って開発しています"
    mock_stt_factory.return_value = mock_stt

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = np.ones(16000, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_llm = MagicMock()
    mock_llm.format_text.return_value = "Claude Codeを使って開発しています。"
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)
    app._process_pipeline()

    # LLM should receive the replaced text
    mock_llm.format_text.assert_called_once_with("Claude Codeを使って開発しています")
    mock_inserter.insert.assert_called_once_with("Claude Codeを使って開発しています。")


# --- LLM disabled tests ---


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_llm_disabled_inserts_raw_stt_text(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """When llm.enabled=False, pipeline should skip LLM and insert raw STT text."""
    from vox.app import VoxApp

    config = AppConfig(llm=LLMConfig(enabled=False))

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "えーとこれはテスト用の長い音声認識テキストです"
    mock_stt_factory.return_value = mock_stt

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = np.ones(16000, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)
    app._process_pipeline()

    # LLM should NOT be called
    mock_llm.format_text.assert_not_called()

    # Inserter should receive the raw STT text
    mock_inserter.insert.assert_called_once_with("えーとこれはテスト用の長い音声認識テキストです")


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_start_llm_disabled_skips_warmup(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """When llm.enabled=False, start() should skip LLM warmup."""
    from vox.app import VoxApp

    config = AppConfig(llm=LLMConfig(enabled=False))

    mock_stt = MagicMock()
    mock_stt_factory.return_value = mock_stt

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    mock_recorder = MagicMock()
    mock_recorder_cls.return_value = mock_recorder

    mock_hotkey = MagicMock()
    mock_hotkey_cls.return_value = mock_hotkey

    app = VoxApp(config)
    app.start()

    mock_stt.load_model.assert_called_once()
    mock_llm.warmup.assert_not_called()
    mock_recorder.open.assert_called_once()
    mock_hotkey.start.assert_called_once()


# --- Duplicate insertion prevention tests ---


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_release_without_press_is_ignored(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """_on_key_release without prior _on_key_press should be a no-op."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_recorder = MagicMock()
    mock_recorder_cls.return_value = mock_recorder

    mock_stt_factory.return_value = MagicMock()
    mock_llm_cls.return_value = MagicMock()
    mock_inserter_cls.return_value = MagicMock()

    mock_hotkey = MagicMock()
    mock_hotkey_cls.return_value = mock_hotkey

    app = VoxApp(config)

    # Call release without press — should be ignored
    app._on_key_release()

    # No pipeline thread should have been started
    assert app._worker is None
    mock_recorder.stop.assert_not_called()
    mock_hotkey.set_enabled.assert_not_called()


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_key_press_during_cooldown_is_ignored(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """Key press within cooldown window after pipeline end should be ignored."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = np.ones(16000, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "テスト"
    mock_stt_factory.return_value = mock_stt

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    mock_hotkey = MagicMock()
    mock_hotkey_cls.return_value = mock_hotkey

    app = VoxApp(config)

    # Simulate pipeline just finished
    app._last_pipeline_end = time.monotonic()

    # Key press during cooldown should be ignored
    app._on_key_press()

    # recorder.start() should NOT have been called
    mock_recorder.start.assert_not_called()
    assert app._recording_active is False
