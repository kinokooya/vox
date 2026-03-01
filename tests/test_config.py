"""Tests for configuration loading."""

from pathlib import Path

import pytest
import yaml

from vox.config import AppConfig, AudioConfig, LLMConfig, load_config


def test_default_config():
    config = AppConfig()
    assert config.stt.engine == "faster-whisper"
    assert config.llm.model == "qwen3:14b"
    assert config.hotkey.trigger_key == "alt_r"
    assert config.audio.sample_rate == 16000
    assert config.audio.max_duration_sec == 60
    assert config.insertion.restore_clipboard is True


def test_faster_whisper_new_defaults():
    config = AppConfig()
    fw = config.stt.faster_whisper
    assert fw.beam_size == 5
    assert fw.hotwords is None
    assert fw.repetition_penalty == 1.0
    assert fw.patience == 1.0
    assert fw.initial_prompt is None


def test_word_replacements_default_empty():
    config = AppConfig()
    assert config.stt.word_replacements == {}


def test_word_replacements_from_yaml(tmp_path: Path):
    data = {
        "stt": {
            "word_replacements": {"クロードコード": "Claude Code", "ギットハブ": "GitHub"},
            "faster_whisper": {
                "hotwords": "Claude Code, GitHub",
                "repetition_penalty": 1.1,
                "patience": 2.0,
            },
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    config = load_config(path)
    assert config.stt.word_replacements == {"クロードコード": "Claude Code", "ギットハブ": "GitHub"}
    assert config.stt.faster_whisper.hotwords == "Claude Code, GitHub"
    assert config.stt.faster_whisper.repetition_penalty == 1.1
    assert config.stt.faster_whisper.patience == 2.0


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
