"""System tray icon for Vox."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _create_icon_image() -> Image.Image:
    """Generate a 64x64 microphone icon programmatically."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Microphone body (rounded rectangle approximated by ellipse + rectangle)
    draw.ellipse([22, 8, 42, 24], fill="#4FC3F7")
    draw.rectangle([22, 16, 42, 36], fill="#4FC3F7")
    draw.ellipse([22, 28, 42, 44], fill="#4FC3F7")

    # Stand arc
    draw.arc([16, 28, 48, 52], start=0, end=180, fill="#B0BEC5", width=3)

    # Stand pole
    draw.line([32, 52, 32, 58], fill="#B0BEC5", width=3)

    # Base
    draw.line([24, 58, 40, 58], fill="#B0BEC5", width=3)

    return img


def create_tray_icon(on_quit: Callable[[], None]) -> Icon:
    """Create and return a pystray Icon (call icon.run() to start the event loop).

    Parameters
    ----------
    on_quit:
        Callback invoked when the user selects "Quit" from the tray menu.
    """
    config_path = _PROJECT_ROOT / "config.yaml"
    log_path = _PROJECT_ROOT / "vox.log"

    def open_config() -> None:
        logger.info("Opening config file: %s", config_path)
        os.startfile(str(config_path))  # type: ignore[attr-defined]

    def open_log() -> None:
        logger.info("Opening log file: %s", log_path)
        os.startfile(str(log_path))  # type: ignore[attr-defined]

    def quit_app() -> None:
        logger.info("Quit requested from tray menu")
        on_quit()

    menu = Menu(
        MenuItem("設定ファイルを開く", lambda: open_config()),
        MenuItem("ログを開く", lambda: open_log()),
        Menu.SEPARATOR,
        MenuItem("終了", lambda: quit_app()),
    )

    icon = Icon(
        name="Vox",
        icon=_create_icon_image(),
        title="Vox - AI Voice Input",
        menu=menu,
    )

    return icon
