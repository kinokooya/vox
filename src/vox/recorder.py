"""Audio recording using sounddevice."""

from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

from vox.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records audio from the microphone while triggered.

    The audio stream is opened once and kept alive for the lifetime of the
    recorder.  Recording on/off is controlled by a flag so that the OS
    microphone resource is never repeatedly acquired and released.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._frames: list[np.ndarray] = []
        self._frame_count = 0
        self._is_recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._max_frames = config.sample_rate * config.max_duration_sec

    def open(self) -> None:
        """Open the audio stream (call once at startup)."""
        if self._stream is not None:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype="float32",
                callback=self._audio_callback,
                latency="low",
            )
            self._stream.start()
            logger.info("Audio stream opened (rate=%dHz)", self._config.sample_rate)
        except Exception:
            logger.exception("Failed to open audio stream")

    def close(self) -> None:
        """Close the audio stream (call once at shutdown)."""
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                logger.exception("Error closing audio stream")
            logger.info("Audio stream closed")

    def start(self) -> None:
        """Start capturing audio frames."""
        with self._lock:
            if self._is_recording:
                return
            self._frames = []
            self._frame_count = 0
            self._is_recording = True
        logger.info("Recording started (max=%ds)", self._config.max_duration_sec)

    def stop(self) -> np.ndarray:
        """Stop capturing and return audio as numpy array."""
        with self._lock:
            self._is_recording = False
            frames = self._frames
            self._frames = []
            self._frame_count = 0

        if not frames:
            logger.warning("No audio frames recorded")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        duration = len(audio) / self._config.sample_rate
        logger.info("Recording stopped: %.1fs, %d samples", duration, len(audio))
        return audio

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)
        with self._lock:
            if not self._is_recording:
                return
            if self._frame_count >= self._max_frames:
                self._is_recording = False
                max_s = self._config.max_duration_sec
                logger.warning("Max recording duration reached (%ds)", max_s)
                return
            self._frames.append(indata.copy())
            self._frame_count += len(indata)
