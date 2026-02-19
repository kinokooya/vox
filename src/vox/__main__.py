"""Entry point for Vox: python -m vox"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

from vox.app import VoxApp
from vox.config import load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

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
