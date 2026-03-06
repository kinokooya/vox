"""Configuration management using Pydantic + YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class VADConfig(BaseModel):
    min_speech_duration_ms: int = Field(default=250, gt=0)
    min_silence_duration_ms: int = Field(default=500, gt=0)


class FasterWhisperConfig(BaseModel):
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ja"
    vad: VADConfig = Field(default_factory=VADConfig)


class SenseVoiceConfig(BaseModel):
    model: str = "FunAudioLLM/SenseVoiceSmall"
    device: str = "cuda"
    language: str = "auto"


class STTConfig(BaseModel):
    engine: str = "faster-whisper"
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)
    sensevoice: SenseVoiceConfig = Field(default_factory=SenseVoiceConfig)


class LLMConfig(BaseModel):
    backend: str = "ollama"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class HotkeyConfig(BaseModel):
    trigger_key: str = "alt_r"


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    max_duration_sec: int = Field(default=60, gt=0)


class InsertionConfig(BaseModel):
    pre_paste_delay_ms: int = Field(default=50, ge=0)
    restore_clipboard: bool = True


class AppConfig(BaseModel):
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    hotkey: HotkeyConfig = Field(default_factory=HotkeyConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    insertion: InsertionConfig = Field(default_factory=InsertionConfig)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file. Falls back to defaults if file not found."""
    if path is None:
        path = Path("config.yaml")

    if not path.exists():
        return AppConfig()

    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        data: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        data = loaded
    else:
        raise ValueError("Config file must contain a YAML mapping at the root")

    return AppConfig(**data)
