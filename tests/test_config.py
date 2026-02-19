"""Tests for configuration loading."""

from pathlib import Path

import pytest
import yaml

from vox.config import AppConfig, AudioConfig, LLMConfig, load_config


def test_default_config():
    config = AppConfig()
    assert config.stt.engine == "faster-whisper"
    assert config.llm.model == "qwen2.5:7b-instruct-q4_K_M"
    assert config.hotkey.trigger_key == "alt_r"
    assert config.audio.sample_rate == 16000
    assert config.audio.max_duration_sec == 60
    assert config.insertion.restore_clipboard is True


def test_load_config_from_yaml(tmp_path: Path):
    data = {
        "stt": {"engine": "sensevoice"},
        "llm": {"model": "qwen2.5:3b-instruct-q4_K_M"},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    config = load_config(path)
    assert config.stt.engine == "sensevoice"
    assert config.llm.model == "qwen2.5:3b-instruct-q4_K_M"
    # Defaults still work
    assert config.hotkey.trigger_key == "alt_r"


def test_load_config_missing_file():
    config = load_config(Path("nonexistent.yaml"))
    assert config.stt.engine == "faster-whisper"


def test_vad_config_defaults():
    config = AppConfig()
    assert config.stt.faster_whisper.vad.min_speech_duration_ms == 250
    assert config.stt.faster_whisper.vad.min_silence_duration_ms == 500


def test_invalid_audio_config():
    with pytest.raises(Exception):
        AudioConfig(sample_rate=-1)


def test_invalid_llm_temperature():
    with pytest.raises(Exception):
        LLMConfig(temperature=5.0)
