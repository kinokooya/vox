"""Main application orchestrator - ties all components together."""

from __future__ import annotations

import logging
import threading
from enum import Enum

from vox.config import AppConfig
from vox.hotkey import HotkeyListener
from vox.inserter import TextInserter
from vox.llm import LLMFormatter
from vox.pipeline import PipelineRunner
from vox.stt import create_stt_engine

logger = logging.getLogger(__name__)


class AppState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    STOPPING = "stopping"
    STOPPED = "stopped"


def _create_recorder(config: AppConfig):
    # Delayed import avoids pulling audio dependencies during pure unit tests.
    from vox.recorder import AudioRecorder

    return AudioRecorder(config.audio)


class VoxApp:
    """Main application: hotkey -> record -> STT -> LLM -> insert."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._recorder = _create_recorder(config)
        self._stt = create_stt_engine(config.stt)
        self._llm = LLMFormatter(config.llm)
        self._inserter = TextInserter(config.insertion)
        self._pipeline = PipelineRunner(
            recorder=self._recorder,
            stt=self._stt,
            llm=self._llm,
            inserter=self._inserter,
            sample_rate=config.audio.sample_rate,
        )
        self._hotkey = HotkeyListener(
            config.hotkey,
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._state = AppState.IDLE
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        """Initialize models and start listening."""
        logger.info("=== Vox starting ===")
        logger.info("STT engine: %s", self._config.stt.engine)
        logger.info("LLM model: %s", self._config.llm.model)

        logger.info("Loading STT model...")
        self._stt.load_model()
        vram = self._stt.get_vram_usage_mb()
        logger.info("STT model loaded (VRAM: ~%dMB)", vram)

        self._hotkey.start()
        with self._lock:
            if self._state == AppState.STOPPED:
                self._state = AppState.IDLE
        trigger = self._config.hotkey.trigger_key
        logger.info("=== Vox ready - press and hold %s to speak ===", trigger)

    def stop(self) -> None:
        """Stop the application. Safe to call multiple times."""
        with self._lock:
            if self._state == AppState.STOPPED:
                return
            prev_state = self._state
            self._state = AppState.STOPPING
            worker = self._worker

        self._hotkey.set_enabled(False)
        self._hotkey.stop()

        if prev_state == AppState.RECORDING:
            # Best-effort cleanup for an active audio stream.
            self._recorder.stop()

        if worker is not None and worker.is_alive():
            logger.info("Waiting for pipeline to finish...")
            worker.join(timeout=10)

        with self._lock:
            self._worker = None
            self._state = AppState.STOPPED
        logger.info("=== Vox stopped ===")

    def _on_key_press(self) -> None:
        with self._lock:
            if self._state != AppState.IDLE:
                logger.info("Ignoring key press in state=%s", self._state)
                return
            self._state = AppState.RECORDING
        self._recorder.start()

    def _on_key_release(self) -> None:
        with self._lock:
            if self._state != AppState.RECORDING:
                return
            self._state = AppState.PROCESSING
        self._hotkey.set_enabled(False)

        thread = threading.Thread(target=self._process_pipeline, name="vox-pipeline")
        with self._lock:
            self._worker = thread
        thread.start()

    def _process_pipeline(self) -> None:
        """Execute one pipeline run in a worker thread."""
        try:
            self._pipeline.run_once()
        except Exception:
            logger.exception("[Pipeline] Error during processing")
        finally:
            should_enable_hotkey = False
            with self._lock:
                self._worker = None
                if self._state not in (AppState.STOPPING, AppState.STOPPED):
                    self._state = AppState.IDLE
                    should_enable_hotkey = True
            if should_enable_hotkey:
                self._hotkey.set_enabled(True)
