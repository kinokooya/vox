"""Audio recording using sounddevice."""

from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

from vox.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records audio from the microphone while triggered."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._frames: list[np.ndarray] = []
        self._frame_count = 0
        self._is_recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._max_frames = config.sample_rate * config.max_duration_sec

    def start(self) -> None:
        """Start recording audio."""
        with self._lock:
            if self._is_recording:
                return
            self._frames = []
            self._frame_count = 0

        rate = self._config.sample_rate
        max_dur = self._config.max_duration_sec
        try:
            stream = sd.InputStream(
                samplerate=rate,
                channels=self._config.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            stream.start()
        except Exception:
            logger.exception("Failed to start audio stream")
            return

        with self._lock:
            self._is_recording = True
            self._stream = stream
        logger.info("Recording started (rate=%dHz, max=%ds)", rate, max_dur)

    def stop(self) -> np.ndarray:
        """Stop recording and return audio as numpy array.

        Returns:
            Audio samples as float32 mono numpy array.
        """
        with self._lock:
            self._is_recording = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Error closing audio stream")

        if not self._frames:
            logger.warning("No audio frames recorded")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self._frames, axis=0).flatten()
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
