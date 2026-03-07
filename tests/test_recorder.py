"""Tests for AudioRecorder lifecycle and buffering."""

from __future__ import annotations

import numpy as np

from vox.config import AudioConfig
from vox.recorder import AudioRecorder


class FakeStream:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def test_start_stop_and_collect_audio() -> None:
    stream = FakeStream()
    holder = {}

    def factory(_rate, _channels, callback):
        holder["callback"] = callback
        return stream

    recorder = AudioRecorder(
        AudioConfig(sample_rate=16000, max_duration_sec=60),
        stream_factory=factory,
    )

    recorder.start()
    callback = holder["callback"]
    callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, object(), 0)
    callback(np.array([[0.3]], dtype=np.float32), 1, object(), 0)
    audio = recorder.stop()

    assert stream.started is True
    assert stream.stopped is True
    assert stream.closed is True
    assert np.allclose(audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_start_failure_returns_empty_audio() -> None:
    def factory(_rate, _channels, _callback):
        raise RuntimeError("no audio device")

    recorder = AudioRecorder(AudioConfig(), stream_factory=factory)

    recorder.start()
    audio = recorder.stop()

    assert audio.size == 0


def test_max_duration_stops_buffer_growth() -> None:
    stream = FakeStream()
    holder = {}

    def factory(_rate, _channels, callback):
        holder["callback"] = callback
        return stream

    recorder = AudioRecorder(AudioConfig(sample_rate=2, max_duration_sec=1), stream_factory=factory)

    recorder.start()
    callback = holder["callback"]
    callback(np.array([[1.0], [2.0]], dtype=np.float32), 2, object(), 0)
    callback(np.array([[3.0]], dtype=np.float32), 1, object(), 0)
    audio = recorder.stop()

    assert np.allclose(audio, np.array([1.0, 2.0], dtype=np.float32))
