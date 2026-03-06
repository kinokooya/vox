"""Main application orchestrator - ties all components together."""

from __future__ import annotations

import logging
import threading

from vox.config import AppConfig
from vox.hotkey import HotkeyListener
from vox.inserter import TextInserter
from vox.llm import LLMFormatter
from vox.pipeline import PipelineRunner
from vox.recorder import AudioRecorder
from vox.stt import create_stt_engine

logger = logging.getLogger(__name__)


class VoxApp:
    """Main application: hotkey -> record -> STT -> LLM -> insert."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._recorder = AudioRecorder(config.audio)
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
        self._processing = False
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
        trigger = self._config.hotkey.trigger_key
        logger.info("=== Vox ready - press and hold %s to speak ===", trigger)

    def stop(self) -> None:
        """Stop the application, waiting for any in-flight work."""
        self._hotkey.stop()
        worker = self._worker
        if worker is not None and worker.is_alive():
            logger.info("Waiting for pipeline to finish...")
            worker.join(timeout=10)
        logger.info("=== Vox stopped ===")

    def _on_key_press(self) -> None:
        with self._lock:
            if self._processing:
                logger.info("Processing in progress, ignoring key press")
                return
        self._recorder.start()

    def _on_key_release(self) -> None:
        with self._lock:
            if self._processing:
                return
            self._processing = True
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
            with self._lock:
                self._processing = False
                self._worker = None
            self._hotkey.set_enabled(True)
