"""Tests for AudioRecorder frame clearing on stop()."""

import numpy as np

from vox.config import AudioConfig
from vox.recorder import AudioRecorder


def test_double_stop_returns_empty():
    """Calling stop() twice should return empty on the second call."""
    config = AudioConfig()
    recorder = AudioRecorder(config)

    # Simulate recording by injecting frames directly
    recorder._is_recording = True
    recorder._frames = [np.ones(100, dtype=np.float32)]
    recorder._frame_count = 100

    first = recorder.stop()
    assert len(first) == 100

    second = recorder.stop()
    assert len(second) == 0


def test_stop_clears_frame_count():
    """stop() should reset _frame_count to 0."""
    config = AudioConfig()
    recorder = AudioRecorder(config)

    recorder._is_recording = True
    recorder._frames = [np.ones(100, dtype=np.float32)]
    recorder._frame_count = 100

    recorder.stop()
    assert recorder._frame_count == 0
    assert recorder._frames == []
