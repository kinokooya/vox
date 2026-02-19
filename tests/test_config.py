"""Tests for configuration loading."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from vox.config import AppConfig, load_config


def test_default_config():
    config = AppConfig()
    assert config.stt.engine == "faster-whisper"
    assert config.llm.model == "qwen2.5:7b-instruct-q4_K_M"
    assert config.hotkey.trigger_key == "alt_r"
    assert config.audio.sample_rate == 16000
    assert config.audio.max_duration_sec == 60
    assert config.insertion.restore_clipboard is True


def test_load_config_from_yaml():
    data = {
        "stt": {"engine": "sensevoice"},
        "llm": {"model": "qwen2.5:3b-instruct-q4_K_M"},
    }
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = Path(f.name)

    config = load_config(path)
    assert config.stt.engine == "sensevoice"
    assert config.llm.model == "qwen2.5:3b-instruct-q4_K_M"
    # Defaults still work
    assert config.hotkey.trigger_key == "alt_r"

    path.unlink()


def test_load_config_missing_file():
    config = load_config(Path("nonexistent.yaml"))
    assert config.stt.engine == "faster-whisper"


def test_vad_config_defaults():
    config = AppConfig()
    assert config.stt.faster_whisper.vad.min_speech_duration_ms == 250
    assert config.stt.faster_whisper.vad.min_silence_duration_ms == 500
