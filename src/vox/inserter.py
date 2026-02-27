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
        self._pending_clipboard: str | None = None

    def insert(self, text: str) -> None:
        """Insert text into the active window.

        Saves the current clipboard, sets the text, sends WM_PASTE.
        Clipboard is restored lazily on next insert() or via
        restore_clipboard_if_pending().
        """
        if not text:
            logger.warning("Empty text, skipping insertion")
            return

        # Restore any previous pending clipboard first
        self._do_restore()

        if self._config.restore_clipboard:
            try:
                self._pending_clipboard = pyperclip.paste()
            except Exception:
                logger.warning("Could not read clipboard, will not restore")
                self._pending_clipboard = None

        pyperclip.copy(text)
        time.sleep(self._config.pre_paste_delay_ms / 1000.0)

        _send_paste()
        logger.info("Text inserted: %d chars", len(text))

    def restore_clipboard_if_pending(self) -> None:
        """Restore clipboard if a previous insert left content pending."""
        self._do_restore()

    def _do_restore(self) -> None:
        if self._pending_clipboard is None:
            return
        try:
            pyperclip.copy(self._pending_clipboard)
            logger.debug("Clipboard restored")
        except Exception:
            logger.warning("Could not restore clipboard")
        finally:
            self._pending_clipboard = None
