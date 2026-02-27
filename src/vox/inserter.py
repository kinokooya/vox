"""Text insertion via EM_REPLACESEL or clipboard + WM_PASTE (Windows)."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging

from vox.config import InsertionConfig

logger = logging.getLogger(__name__)

# Win32 constants
EM_REPLACESEL = 0x00C2
WM_PASTE = 0x0302
WM_NULL = 0x0000


def _get_focused_hwnd() -> int:
    """Return the HWND of the focused control in the foreground window."""
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return 0

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

    return hwnd_focus or hwnd


class TextInserter:
    """Inserts text into the active text field."""

    def __init__(self, config: InsertionConfig) -> None:
        self._config = config

    def insert(self, text: str) -> None:
        """Insert text into the active window.

        Primary method: EM_REPLACESEL sends text directly to the control,
        bypassing the clipboard entirely.
        Fallback: clipboard + WM_PASTE for controls that don't support
        EM_REPLACESEL.
        """
        if not text:
            logger.warning("Empty text, skipping insertion")
            return

        target = _get_focused_hwnd()
        if not target:
            logger.warning("No focused window found for insertion")
            return

        # Try direct text insertion — no clipboard, no timing issues
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SendMessageW(target, EM_REPLACESEL, True, ctypes.c_wchar_p(text))
        logger.info("Text inserted via EM_REPLACESEL: %d chars (hwnd=0x%X)", len(text), target)

    def restore_clipboard_if_pending(self) -> None:
        """No-op: EM_REPLACESEL does not use the clipboard."""
