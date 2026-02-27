"""Text insertion via clipboard + WM_PASTE window message (Windows).

Sets the clipboard to the desired text, then sends WM_PASTE directly
to the focused control via SendMessage.  Unlike SendInput-based
approaches (Ctrl+V keystrokes, KEYEVENTF_UNICODE), WM_PASTE is a
window message that bypasses the keyboard input pipeline entirely,
so the IME has no opportunity to intercept or duplicate the text.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import time

from vox.config import InsertionConfig

logger = logging.getLogger(__name__)

# Win32 constants
WM_PASTE = 0x0302

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


class _GUITHREADINFO(ctypes.Structure):
    """Win32 GUITHREADINFO — used to find the focused window handle."""

    _fields_ = [
        ("cbSize", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("hwndActive", ctypes.wintypes.HWND),
        ("hwndFocus", ctypes.wintypes.HWND),
        ("hwndCapture", ctypes.wintypes.HWND),
        ("hwndMenuOwner", ctypes.wintypes.HWND),
        ("hwndMoveSize", ctypes.wintypes.HWND),
        ("hwndCaret", ctypes.wintypes.HWND),
        ("rcCaret", ctypes.wintypes.RECT),
    ]


# ---------------------------------------------------------------------------
# Win32 API setup
# ---------------------------------------------------------------------------

user32 = ctypes.windll.user32  # type: ignore[attr-defined]
kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

# Clipboard
user32.OpenClipboard.argtypes = [ctypes.wintypes.HWND]
user32.OpenClipboard.restype = ctypes.wintypes.BOOL
user32.CloseClipboard.argtypes = []
user32.CloseClipboard.restype = ctypes.wintypes.BOOL
user32.EmptyClipboard.argtypes = []
user32.EmptyClipboard.restype = ctypes.wintypes.BOOL
user32.SetClipboardData.argtypes = [ctypes.wintypes.UINT, ctypes.wintypes.HANDLE]
user32.SetClipboardData.restype = ctypes.wintypes.HANDLE
user32.GetClipboardData.argtypes = [ctypes.wintypes.UINT]
user32.GetClipboardData.restype = ctypes.wintypes.HANDLE

# Global memory
kernel32.GlobalAlloc.argtypes = [ctypes.wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = ctypes.wintypes.HANDLE
kernel32.GlobalLock.argtypes = [ctypes.wintypes.HANDLE]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [ctypes.wintypes.HANDLE]
kernel32.GlobalUnlock.restype = ctypes.wintypes.BOOL
kernel32.GlobalFree.argtypes = [ctypes.wintypes.HANDLE]
kernel32.GlobalFree.restype = ctypes.wintypes.HANDLE

# Window / thread
user32.GetGUIThreadInfo.argtypes = [ctypes.wintypes.DWORD, ctypes.POINTER(_GUITHREADINFO)]
user32.GetGUIThreadInfo.restype = ctypes.wintypes.BOOL
user32.SendMessageW.argtypes = [
    ctypes.wintypes.HWND,
    ctypes.wintypes.UINT,
    ctypes.wintypes.WPARAM,
    ctypes.wintypes.LPARAM,
]
user32.SendMessageW.restype = ctypes.wintypes.LPARAM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_focused_hwnd() -> int:
    """Return the HWND of the currently focused control (or 0)."""
    gui = _GUITHREADINFO()
    gui.cbSize = ctypes.sizeof(_GUITHREADINFO)
    if user32.GetGUIThreadInfo(0, ctypes.byref(gui)):
        return gui.hwndFocus or gui.hwndActive or 0
    return 0


def _set_clipboard_text(text: str) -> None:
    """Write *text* to the Windows clipboard as CF_UNICODETEXT."""
    encoded = text.encode("utf-16-le") + b"\x00\x00"  # null-terminated
    h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
    if not h_mem:
        raise OSError("GlobalAlloc failed")

    ptr = kernel32.GlobalLock(h_mem)
    if not ptr:
        kernel32.GlobalFree(h_mem)
        raise OSError("GlobalLock failed")

    try:
        ctypes.memmove(ptr, encoded, len(encoded))
    finally:
        kernel32.GlobalUnlock(h_mem)

    if not user32.OpenClipboard(None):
        kernel32.GlobalFree(h_mem)
        raise OSError("OpenClipboard failed")
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_UNICODETEXT, h_mem):
            kernel32.GlobalFree(h_mem)
            raise OSError("SetClipboardData failed")
        # After SetClipboardData succeeds, the system owns h_mem.
    finally:
        user32.CloseClipboard()


def _get_clipboard_text() -> str | None:
    """Read CF_UNICODETEXT from the clipboard, or *None* if unavailable."""
    if not user32.OpenClipboard(None):
        return None
    try:
        h_data = user32.GetClipboardData(CF_UNICODETEXT)
        if not h_data:
            return None
        ptr = kernel32.GlobalLock(h_data)
        if not ptr:
            return None
        try:
            return ctypes.wstring_at(ptr)  # type: ignore[arg-type]
        finally:
            kernel32.GlobalUnlock(h_data)
    finally:
        user32.CloseClipboard()


# ---------------------------------------------------------------------------
# TextInserter
# ---------------------------------------------------------------------------


class TextInserter:
    """Inserts text into the active text field via clipboard + WM_PASTE."""

    def __init__(self, config: InsertionConfig) -> None:
        self._config = config
        self._saved_clipboard: str | None = None

    def insert(self, text: str) -> None:
        """Insert *text* by setting the clipboard and sending WM_PASTE.

        WM_PASTE is a window message sent directly to the focused control.
        It bypasses the keyboard input pipeline entirely, so the IME cannot
        intercept or duplicate the text — unlike SendInput Ctrl+V or
        KEYEVENTF_UNICODE which go through the low-level keyboard hook
        chain where IME sits.
        """
        if not text:
            logger.warning("Empty text, skipping insertion")
            return

        # Save current clipboard content for later restoration
        if self._config.restore_clipboard:
            try:
                self._saved_clipboard = _get_clipboard_text()
            except Exception:
                logger.debug("Could not read clipboard for save", exc_info=True)
                self._saved_clipboard = None

        # Set the clipboard to the desired text
        _set_clipboard_text(text)

        # Optional delay before pasting
        if self._config.pre_paste_delay_ms > 0:
            time.sleep(self._config.pre_paste_delay_ms / 1000)

        # Find the focused control and send WM_PASTE
        hwnd = _get_focused_hwnd()
        if not hwnd:
            logger.warning("No focused window found, cannot paste")
            return

        user32.SendMessageW(hwnd, WM_PASTE, 0, 0)

        logger.info(
            "Text inserted via WM_PASTE to hwnd=%#x: %d chars",
            hwnd,
            len(text),
        )

    def restore_clipboard_if_pending(self) -> None:
        """Restore the clipboard content that was saved before insertion."""
        if self._saved_clipboard is not None:
            try:
                _set_clipboard_text(self._saved_clipboard)
                logger.debug("Clipboard restored")
            except Exception:
                logger.debug("Could not restore clipboard", exc_info=True)
            finally:
                self._saved_clipboard = None
