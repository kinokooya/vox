"""Text insertion via clipboard + WM_PASTE (Windows)."""

from __future__ import annotations

import ctypes
import logging
import time

import pyperclip

from vox.config import InsertionConfig

logger = logging.getLogger(__name__)

# Win32 constants
WM_PASTE = 0x0302


def _send_paste() -> None:
    """Send WM_PASTE to the focused control (bypasses keyboard hooks)."""
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        logger.warning("No foreground window found for paste")
        return

    # Attach to the foreground thread to discover which control has focus
    target_tid = user32.GetWindowThreadProcessId(hwnd, None)
    our_tid = kernel32.GetCurrentThreadId()

    attached = False
    if target_tid != our_tid:
        attached = bool(user32.AttachThreadInput(our_tid, target_tid, True))

    try:
        hwnd_focus = user32.GetFocus()
    finally:
        if attached:
            user32.AttachThreadInput(our_tid, target_tid, False)

    target = hwnd_focus or hwnd
    user32.SendMessageW(target, WM_PASTE, 0, 0)
    logger.debug("WM_PASTE sent to hwnd=0x%X", target)


class TextInserter:
    """Inserts text into the active text field via clipboard paste."""

    def __init__(self, config: InsertionConfig) -> None:
        self._config = config

    def insert(self, text: str) -> None:
        """Insert text into the active window.

        Saves the current clipboard, sets the text, sends WM_PASTE,
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
            _send_paste()
            logger.info("Text inserted: %d chars", len(text))
        finally:
            if self._config.restore_clipboard and captured:
                time.sleep(0.1)  # Wait for paste to complete
                try:
                    pyperclip.copy(original_clipboard)
                except Exception:
                    logger.warning("Could not restore clipboard")
