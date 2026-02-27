"""Configuration management using Pydantic + YAML."""

from __future__ import annotations

from pathlib import Path

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
    beam_size: int = Field(default=5, ge=1, le=10)
    condition_on_previous_text: bool = False
    no_speech_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    log_prob_threshold: float = Field(default=-0.5, ge=-5.0, le=0.0)
    compression_ratio_threshold: float = Field(default=2.0, gt=0.0)
    hallucination_silence_threshold: float | None = Field(default=2.0, ge=0.0)
    initial_prompt: str | None = None
    hotwords: str | None = None
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    patience: float = Field(default=1.0, ge=0.0)
    vad: VADConfig = VADConfig()


class SenseVoiceConfig(BaseModel):
    model: str = "FunAudioLLM/SenseVoiceSmall"
    device: str = "cuda"
    language: str = "auto"


class STTConfig(BaseModel):
    engine: str = "faster-whisper"
    word_replacements: dict[str, str] = Field(default_factory=dict)
    faster_whisper: FasterWhisperConfig = FasterWhisperConfig()
    sensevoice: SenseVoiceConfig = SenseVoiceConfig()


class LLMConfig(BaseModel):
    backend: str = "ollama"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    timeout_sec: float = Field(default=30.0, gt=0)
    output_format: str = Field(default="single_line")
    skip_short: bool = True
    skip_short_max_chars: int = Field(default=20, gt=0)


class HotkeyConfig(BaseModel):
    trigger_key: str = "alt_r"


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    max_duration_sec: int = Field(default=60, gt=0)
    min_duration_sec: float = Field(default=0.5, ge=0.0)


class InsertionConfig(BaseModel):
    pre_paste_delay_ms: int = Field(default=50, ge=0)
    restore_clipboard: bool = True


class AppConfig(BaseModel):
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    hotkey: HotkeyConfig = HotkeyConfig()
    audio: AudioConfig = AudioConfig()
    insertion: InsertionConfig = InsertionConfig()


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file. Falls back to defaults if file not found."""
    if path is None:
        path = Path("config.yaml")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()
