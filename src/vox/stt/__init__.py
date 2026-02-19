"""STT engine abstraction layer."""

from vox.stt.base import STTEngine
from vox.stt.factory import create_stt_engine

__all__ = ["STTEngine", "create_stt_engine"]
