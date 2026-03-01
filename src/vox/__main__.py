"""Entry point for Vox: python -m vox"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

from vox.app import VoxApp
from vox.config import load_config


def _register_cuda_dll_dirs() -> None:
    """Add NVIDIA CUDA DLL directories installed via pip to the DLL search path."""
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site_packages.exists():
        return
    for bin_dir in site_packages.glob("*/bin"):
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _setup_logging() -> None:
    """Configure logging. Use file output when stderr is unavailable (pythonw.exe)."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%H:%M:%S"

    handlers: list[logging.Handler] = []

    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())

    # Always write to log file for debugging
    log_path = Path(__file__).resolve().parent.parent.parent / "vox.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


def main() -> None:
    _register_cuda_dll_dirs()
    _setup_logging()

    config_path = Path("config.yaml")
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])

    config = load_config(config_path)
    app = VoxApp(config)

    def shutdown(signum: int, frame: object) -> None:
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        app.start()
        # Keep the main thread alive (hotkey listener runs in daemon thread)
        signal.pause()
    except AttributeError:
        # signal.pause() not available on Windows — use Event instead
        import threading

        stop_event = threading.Event()
        try:
            stop_event.wait()
        except KeyboardInterrupt:
            pass
    finally:
        app.stop()


if __name__ == "__main__":
    main()
