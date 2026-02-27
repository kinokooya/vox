"""Tests for pipeline improvements: min duration, LLM skip, output normalization."""

from unittest.mock import MagicMock, patch

import numpy as np

from vox.config import AppConfig, LLMConfig
from vox.llm import LLMFormatter

# --- Min audio duration tests ---


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_skips_short_audio(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """Audio shorter than min_duration_sec should be skipped."""
    from vox.app import VoxApp

    config = AppConfig()
    config.audio.min_duration_sec = 0.5

    mock_stt = MagicMock()
    mock_stt_factory.return_value = mock_stt

    mock_recorder = MagicMock()
    # 0.3s of audio at 16kHz (below 0.5s threshold)
    mock_recorder.stop.return_value = np.ones(4800, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)
    app._process_pipeline()

    mock_stt.transcribe.assert_not_called()
    mock_llm.format_text.assert_not_called()
    mock_inserter.insert.assert_not_called()


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_processes_long_enough_audio(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """Audio at or above min_duration_sec should be processed."""
    from vox.app import VoxApp

    config = AppConfig()
    config.audio.min_duration_sec = 0.5

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "テスト"
    mock_stt_factory.return_value = mock_stt

    mock_recorder = MagicMock()
    # 1.0s of audio at 16kHz (above 0.5s threshold)
    mock_recorder.stop.return_value = np.ones(16000, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_llm = MagicMock()
    mock_llm.format_text.return_value = "テスト"
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)
    app._process_pipeline()

    mock_stt.transcribe.assert_called_once()
    mock_inserter.insert.assert_called_once()


# --- LLM skip tests ---


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_skips_llm_for_short_text(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """Short STT text without fillers should skip LLM."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "OK"
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
    # Inserter should be called with the raw text
    mock_inserter.insert.assert_called_once_with("OK")


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_does_not_skip_llm_with_fillers(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """Short text containing fillers should still go through LLM."""
    from vox.app import VoxApp

    config = AppConfig()

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "えーとOK"
    mock_stt_factory.return_value = mock_stt

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = np.ones(16000, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_llm = MagicMock()
    mock_llm.format_text.return_value = "OK"
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)
    app._process_pipeline()

    # LLM SHOULD be called because of filler
    mock_llm.format_text.assert_called_once_with("えーとOK")
    mock_inserter.insert.assert_called_once_with("OK")


@patch("vox.app.TextInserter")
@patch("vox.app.LLMFormatter")
@patch("vox.app.create_stt_engine")
@patch("vox.app.AudioRecorder")
@patch("vox.app.HotkeyListener")
def test_pipeline_does_not_skip_llm_for_long_text(
    mock_hotkey_cls,
    mock_recorder_cls,
    mock_stt_factory,
    mock_llm_cls,
    mock_inserter_cls,
):
    """Text longer than skip_short_max_chars should go through LLM."""
    from vox.app import VoxApp

    config = AppConfig()

    long_text = "これは20文字を超えるテキストです。整形が必要です。"
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = long_text
    mock_stt_factory.return_value = mock_stt

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = np.ones(48000, dtype=np.float32)
    mock_recorder_cls.return_value = mock_recorder

    mock_llm = MagicMock()
    mock_llm.format_text.return_value = "整形済みテキスト。"
    mock_llm_cls.return_value = mock_llm

    mock_inserter = MagicMock()
    mock_inserter_cls.return_value = mock_inserter

    app = VoxApp(config)
    app._process_pipeline()

    mock_llm.format_text.assert_called_once()


def test_should_skip_llm_disabled():
    """When skip_short is False, LLM is never skipped."""
    from vox.app import VoxApp

    config = AppConfig()
    config.llm.skip_short = False

    with patch("vox.app.HotkeyListener"), \
         patch("vox.app.AudioRecorder"), \
         patch("vox.app.create_stt_engine"), \
         patch("vox.app.LLMFormatter"), \
         patch("vox.app.TextInserter"):
        app = VoxApp(config)
        assert app._should_skip_llm("OK") is False


# --- Output normalization tests ---


@patch("vox.llm.OpenAI")
def test_single_line_removes_newlines(mock_openai_cls):
    """single_line format should replace newlines with spaces."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "line one\nline two\nline three"
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig(output_format="single_line")
    formatter = LLMFormatter(config)
    result = formatter.format_text("line one line two line three")

    assert "\n" not in result
    assert result == "line one line two line three"


@patch("vox.llm.OpenAI")
def test_multi_line_preserves_newlines(mock_openai_cls):
    """multi_line format should preserve newlines."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "line one\nline two"
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig(output_format="multi_line")
    formatter = LLMFormatter(config)
    result = formatter.format_text("test input")

    assert result == "line one\nline two"


@patch("vox.llm.OpenAI")
def test_single_line_collapses_multiple_spaces(mock_openai_cls):
    """single_line should collapse multiple spaces into one."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "word1\n\n\nword2"
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig(output_format="single_line")
    formatter = LLMFormatter(config)
    result = formatter.format_text("word1 word2")

    assert result == "word1 word2"


# --- Config backward compatibility tests ---


def test_config_backward_compat_no_new_fields():
    """Existing config without new fields should still work with defaults."""
    config = AppConfig()
    assert config.audio.min_duration_sec == 0.5
    assert config.llm.output_format == "single_line"
    assert config.llm.skip_short is True
    assert config.llm.skip_short_max_chars == 20
    assert config.llm.max_tokens == 512
    assert config.stt.faster_whisper.beam_size == 1
    assert config.stt.faster_whisper.condition_on_previous_text is False
    assert config.stt.faster_whisper.no_speech_threshold == 0.6
    assert config.stt.faster_whisper.log_prob_threshold == -0.5
    assert config.stt.faster_whisper.compression_ratio_threshold == 2.0
    assert config.stt.faster_whisper.hallucination_silence_threshold == 2.0
    assert config.stt.faster_whisper.initial_prompt is None


def test_config_partial_override():
    """Config with only some fields overridden should use defaults for the rest."""
    config = AppConfig(
        llm={"model": "custom-model"},  # type: ignore[arg-type]
        audio={"sample_rate": 44100},  # type: ignore[arg-type]
    )
    assert config.llm.model == "custom-model"
    assert config.llm.skip_short is True  # default
    assert config.audio.sample_rate == 44100
    assert config.audio.min_duration_sec == 0.5  # default
