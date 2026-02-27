"""Text insertion via SendInput with KEYEVENTF_UNICODE (Windows).

Sends Unicode characters directly as keyboard events, bypassing the
clipboard and IME.  Works with all applications including Chrome,
modern Notepad, and standard Win32 controls.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging

from vox.config import InsertionConfig

logger = logging.getLogger(__name__)

# Win32 constants
INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002


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


class TextInserter:
    """Inserts text into the active text field using KEYEVENTF_UNICODE."""

    def __init__(self, config: InsertionConfig) -> None:
        self._config = config

    def insert(self, text: str) -> None:
        """Insert text by sending Unicode keyboard events via SendInput.

        Each character is sent as a key-down + key-up pair with the
        KEYEVENTF_UNICODE flag.  All events are batched into a single
        SendInput call for atomicity and speed.
        """
        if not text:
            logger.warning("Empty text, skipping insertion")
            return

        events: list[_INPUT] = []
        for char in text:
            code = ord(char)
            # Key down
            inp_down = _INPUT(type=INPUT_KEYBOARD)
            inp_down.ki.wVk = 0
            inp_down.ki.wScan = code
            inp_down.ki.dwFlags = KEYEVENTF_UNICODE
            inp_down.ki.time = 0
            inp_down.ki.dwExtraInfo = None
            events.append(inp_down)
            # Key up
            inp_up = _INPUT(type=INPUT_KEYBOARD)
            inp_up.ki.wVk = 0
            inp_up.ki.wScan = code
            inp_up.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
            inp_up.ki.time = 0
            inp_up.ki.dwExtraInfo = None
            events.append(inp_up)

        arr = (_INPUT * len(events))(*events)
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        sent = user32.SendInput(len(events), arr, ctypes.sizeof(_INPUT))

        logger.info(
            "Text inserted via SendInput KEYEVENTF_UNICODE: "
            "%d chars (%d/%d events)",
            len(text),
            sent,
            len(events),
        )

    def restore_clipboard_if_pending(self) -> None:
        """No-op: KEYEVENTF_UNICODE does not use the clipboard."""
