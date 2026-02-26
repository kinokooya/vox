"""Main application orchestrator — ties all components together."""

from __future__ import annotations

import logging
import threading

from vox.config import AppConfig
from vox.hotkey import HotkeyListener
from vox.inserter import TextInserter
from vox.llm import LLMFormatter
from vox.recorder import AudioRecorder
from vox.stt import create_stt_engine

logger = logging.getLogger(__name__)


class VoxApp:
    """Main application: hotkey → record → STT → LLM → insert."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._recorder = AudioRecorder(config.audio)
        self._stt = create_stt_engine(config.stt)
        self._llm = LLMFormatter(config.llm)
        self._inserter = TextInserter(config.insertion)
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

        self._recorder.open()
        self._hotkey.start()
        trigger = self._config.hotkey.trigger_key
        logger.info("=== Vox ready — press and hold %s to speak ===", trigger)

    def stop(self) -> None:
        """Stop the application, waiting for any in-flight work."""
        self._hotkey.stop()
        worker = self._worker
        if worker is not None and worker.is_alive():
            logger.info("Waiting for pipeline to finish...")
            worker.join(timeout=10)
        self._recorder.close()
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

        thread = threading.Thread(target=self._process_pipeline)
        with self._lock:
            self._worker = thread
        thread.start()

    def _process_pipeline(self) -> None:
        """Execute the full pipeline: record stop → STT → LLM → insert."""
        try:
            # 1. Stop recording and get audio
            audio = self._recorder.stop()
            if len(audio) == 0:
                logger.warning("No audio captured, skipping pipeline")
                return

            duration = len(audio) / self._config.audio.sample_rate
            logger.info("[Pipeline] Audio: %.1fs, %d samples", duration, len(audio))

            # 2. STT
            logger.info("[Pipeline] Running STT...")
            raw_text = self._stt.transcribe(audio, self._config.audio.sample_rate)
            if not raw_text.strip():
                logger.info("[Pipeline] STT returned empty text, skipping")
                return
            logger.info("[Pipeline] STT result: %s", raw_text)

            # 3. LLM formatting (fallback to raw text on failure)
            logger.info("[Pipeline] Running LLM formatting...")
            try:
                formatted_text = self._llm.format_text(raw_text)
                logger.info("[Pipeline] LLM result: %s", formatted_text)
            except Exception:
                logger.warning("[Pipeline] LLM failed, falling back to raw STT text")
                formatted_text = raw_text

            # 4. Text insertion
            logger.info("[Pipeline] Inserting text...")
            self._inserter.insert(formatted_text)
            logger.info("[Pipeline] Done!")

        except Exception:
            logger.exception("[Pipeline] Error during processing")
        finally:
            with self._lock:
                self._processing = False
                self._worker = None
            self._hotkey.set_enabled(True)
