"""Text insertion via clipboard + Ctrl+V paste (Windows).

Uses the clipboard to set text and then sends Ctrl+V via SendInput.
Unlike KEYEVENTF_UNICODE, paste operations are not intercepted by IME,
which prevents the duplicate-text bug on Windows 10/11 with TSF-based
Japanese input methods.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import time

from vox.config import InsertionConfig

logger = logging.getLogger(__name__)

# Win32 constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

VK_LCONTROL = 0xA2
VK_V = 0x56


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _INPUT(ctypes.Structure):
    """Win32 INPUT structure for SendInput."""

    class _U(ctypes.Union):
        _fields_ = [
            ("ki", _KEYBDINPUT),
            # Pad to at least sizeof(MOUSEINPUT) so SendInput array stride
            # matches the real INPUT size on 64-bit Windows (40 bytes).
            ("_pad", ctypes.c_byte * 32),
        ]

    _anonymous_ = ("_u",)
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("_u", _U),
    ]


# ---------------------------------------------------------------------------
# Win32 clipboard helpers
# ---------------------------------------------------------------------------

user32 = ctypes.windll.user32  # type: ignore[attr-defined]
kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

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

kernel32.GlobalAlloc.argtypes = [ctypes.wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = ctypes.wintypes.HANDLE
kernel32.GlobalLock.argtypes = [ctypes.wintypes.HANDLE]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [ctypes.wintypes.HANDLE]
kernel32.GlobalUnlock.restype = ctypes.wintypes.BOOL
kernel32.GlobalFree.argtypes = [ctypes.wintypes.HANDLE]
kernel32.GlobalFree.restype = ctypes.wintypes.HANDLE


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
    """Inserts text into the active text field via clipboard + Ctrl+V."""

    def __init__(self, config: InsertionConfig) -> None:
        self._config = config
        self._saved_clipboard: str | None = None

    def insert(self, text: str) -> None:
        """Insert *text* by setting the clipboard and sending Ctrl+V.

        Uses ``VK_LCONTROL`` (left Ctrl) so that pynput's hotkey listener
        — which monitors ``ctrl_r`` (right Ctrl) — does not re-trigger.
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

        # Build Ctrl+V key events: LCtrl down → V down → V up → LCtrl up
        events: list[_INPUT] = []

        ctrl_down = _INPUT(type=INPUT_KEYBOARD)
        ctrl_down.ki.wVk = VK_LCONTROL
        ctrl_down.ki.dwFlags = 0
        events.append(ctrl_down)

        v_down = _INPUT(type=INPUT_KEYBOARD)
        v_down.ki.wVk = VK_V
        v_down.ki.dwFlags = 0
        events.append(v_down)

        v_up = _INPUT(type=INPUT_KEYBOARD)
        v_up.ki.wVk = VK_V
        v_up.ki.dwFlags = KEYEVENTF_KEYUP
        events.append(v_up)

        ctrl_up = _INPUT(type=INPUT_KEYBOARD)
        ctrl_up.ki.wVk = VK_LCONTROL
        ctrl_up.ki.dwFlags = KEYEVENTF_KEYUP
        events.append(ctrl_up)

        arr = (_INPUT * len(events))(*events)
        sent = user32.SendInput(len(events), arr, ctypes.sizeof(_INPUT))

        logger.info(
            "Text inserted via clipboard + Ctrl+V: %d chars (%d/%d events)",
            len(text),
            sent,
            len(events),
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
