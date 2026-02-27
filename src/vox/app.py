"""Main application orchestrator — ties all components together."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
        self._recording_active = False
        self._last_pipeline_end = 0.0
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        """Initialize models and start listening."""
        logger.info("=== Vox starting ===")
        logger.info("STT engine: %s", self._config.stt.engine)
        logger.info("LLM model: %s", self._config.llm.model)

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            stt_future = pool.submit(self._load_stt)
            llm_future = pool.submit(self._warmup_llm)

            # STT is required — propagate failure
            stt_future.result()
            # LLM warmup is best-effort — already logged inside _warmup_llm
            llm_future.result()

        elapsed = time.monotonic() - t0
        logger.info("Model loading completed in %.1fs", elapsed)

        self._recorder.open()
        self._hotkey.start()
        trigger = self._config.hotkey.trigger_key
        logger.info("=== Vox ready — press and hold %s to speak ===", trigger)

    def _load_stt(self) -> None:
        """Load the STT model (runs in thread pool)."""
        t0 = time.monotonic()
        logger.info("Loading STT model...")
        self._stt.load_model()
        vram = self._stt.get_vram_usage_mb()
        elapsed = time.monotonic() - t0
        logger.info("STT model loaded in %.1fs (VRAM: ~%dMB)", elapsed, vram)

    def _warmup_llm(self) -> None:
        """Warm up the LLM (runs in thread pool). Failures are non-fatal."""
        try:
            t0 = time.monotonic()
            self._llm.warmup()
            elapsed = time.monotonic() - t0
            logger.info("LLM warmup completed in %.1fs", elapsed)
        except Exception:
            logger.warning("LLM warmup failed, continuing without warmup", exc_info=True)

    def stop(self) -> None:
        """Stop the application, waiting for any in-flight work."""
        self._hotkey.stop()
        worker = self._worker
        if worker is not None and worker.is_alive():
            logger.info("Waiting for pipeline to finish...")
            worker.join(timeout=10)
        self._recorder.close()
        logger.info("=== Vox stopped ===")

    _PIPELINE_COOLDOWN_SEC = 0.3

    def _on_key_press(self) -> None:
        with self._lock:
            if self._processing:
                logger.info("Processing in progress, ignoring key press")
                return
            if time.monotonic() - self._last_pipeline_end < self._PIPELINE_COOLDOWN_SEC:
                logger.info("Pipeline cooldown active, ignoring key press")
                return
            self._recording_active = True
        self._recorder.start()

    def _on_key_release(self) -> None:
        with self._lock:
            if not self._recording_active:
                logger.debug("Release without press, ignoring")
                return
            self._recording_active = False
            if self._processing:
                return
            self._processing = True
        self._hotkey.set_enabled(False)

        thread = threading.Thread(target=self._process_pipeline)
        with self._lock:
            self._worker = thread
        thread.start()

    _FILLERS = ["えーと", "あのー", "あの", "まあ", "えー", "うーん", "えっと"]

    def _should_skip_llm(self, text: str) -> bool:
        """Skip LLM for short, clean text without fillers."""
        if not self._config.llm.skip_short:
            return False
        if len(text) > self._config.llm.skip_short_max_chars:
            return False
        return not any(f in text for f in self._FILLERS)

    def _apply_word_replacements(self, text: str) -> str:
        """Apply word replacements from config (e.g. katakana → Latin)."""
        for old, new in self._config.stt.word_replacements.items():
            text = text.replace(old, new)
        return text

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

            # 1b. Minimum duration check
            if duration < self._config.audio.min_duration_sec:
                logger.info(
                    "[Pipeline] Audio too short (%.1fs), skipping", duration
                )
                return

            # 2. STT
            logger.info("[Pipeline] Running STT...")
            raw_text = self._stt.transcribe(audio, self._config.audio.sample_rate)
            if not raw_text.strip():
                logger.info("[Pipeline] STT returned empty text, skipping")
                return
            raw_text = self._apply_word_replacements(raw_text)
            logger.info("[Pipeline] STT result: %s", raw_text)

            # 3. LLM formatting (skip for short text, fallback to raw on failure)
            if self._should_skip_llm(raw_text):
                logger.info("[Pipeline] Short text, skipping LLM: %s", raw_text)
                formatted_text = raw_text
            else:
                logger.info("[Pipeline] Running LLM formatting...")
                try:
                    formatted_text = self._llm.format_text(raw_text)
                    logger.info("[Pipeline] LLM result: %s", formatted_text)
                except Exception:
                    logger.warning(
                        "[Pipeline] LLM failed, falling back to raw STT text"
                    )
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
                self._last_pipeline_end = time.monotonic()
            self._hotkey.set_enabled(True)
