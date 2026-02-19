"""Text insertion via clipboard + Ctrl+V (Windows)."""

from __future__ import annotations

import ctypes
import logging
import time

import pyperclip

from vox.config import InsertionConfig

logger = logging.getLogger(__name__)

# Win32 virtual key codes
VK_CONTROL = 0x11
VK_V = 0x56
KEYEVENTF_KEYUP = 0x0002


def _send_ctrl_v() -> None:
    """Simulate Ctrl+V keypress using Win32 keybd_event."""
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(VK_V, 0, 0, 0)
    user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


class TextInserter:
    """Inserts text into the active text field via clipboard paste."""

    def __init__(self, config: InsertionConfig) -> None:
        self._config = config

    def insert(self, text: str) -> None:
        """Insert text into the active window.

        Saves the current clipboard, sets the text, sends Ctrl+V,
        then restores the original clipboard content.
        """
        if not text:
            logger.warning("Empty text, skipping insertion")
            return

        captured = False
        original_clipboard = ""
        if self._config.restore_clipboard:
            try:
                original_clipboard = pyperclip.paste()
                captured = True
            except Exception:
                logger.warning("Could not read clipboard, will not restore")

        pyperclip.copy(text)
        time.sleep(self._config.pre_paste_delay_ms / 1000.0)

        try:
            _send_ctrl_v()
            logger.info("Text inserted: %d chars", len(text))
        finally:
            if self._config.restore_clipboard and captured:
                time.sleep(0.1)  # Wait for paste to complete
                try:
                    pyperclip.copy(original_clipboard)
                except Exception:
                    logger.warning("Could not restore clipboard")
