"""Text insertion via SendInput with KEYEVENTF_UNICODE (Windows).

Sends Unicode characters directly as keyboard events, bypassing the
clipboard.  IME is temporarily dissociated from the target window to
prevent the input method from intercepting the injected key events and
producing duplicate / garbled text.
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


def _get_focused_hwnd() -> int:
    """Return the HWND of the currently focused control (or 0)."""
    gui = _GUITHREADINFO()
    gui.cbSize = ctypes.sizeof(_GUITHREADINFO)
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    if user32.GetGUIThreadInfo(0, ctypes.byref(gui)):
        return gui.hwndFocus or gui.hwndActive or 0
    return 0


class TextInserter:
    """Inserts text into the active text field using KEYEVENTF_UNICODE."""

    def __init__(self, config: InsertionConfig) -> None:
        self._config = config

    @staticmethod
    def _build_events(text: str) -> list[_INPUT]:
        """Build a list of KEYEVENTF_UNICODE key-down/up pairs for *text*."""
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
        return events

    def insert(self, text: str) -> None:
        """Insert text by sending Unicode keyboard events via SendInput.

        Each character is sent as a key-down + key-up pair with the
        KEYEVENTF_UNICODE flag.  All events are batched into a single
        SendInput call for atomicity and speed.

        The target window's IME context is temporarily dissociated so
        the input method does not intercept the injected events.
        """
        if not text:
            logger.warning("Empty text, skipping insertion")
            return

        events = self._build_events(text)
        arr = (_INPUT * len(events))(*events)

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        imm32 = ctypes.windll.imm32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        # --- Temporarily dissociate IME from the focused window ----------
        hwnd = _get_focused_hwnd()
        prev_ctx = None
        attached = False
        target_tid = 0

        if hwnd:
            my_tid = kernel32.GetCurrentThreadId()
            target_tid = user32.GetWindowThreadProcessId(hwnd, None)
            if target_tid and target_tid != my_tid:
                attached = bool(
                    user32.AttachThreadInput(my_tid, target_tid, True)
                )
            # ImmAssociateContext returns the previous IME context (or 0)
            prev_ctx = imm32.ImmAssociateContext(hwnd, None)
            if prev_ctx:
                logger.debug(
                    "IME context dissociated from hwnd=%#x (prev=%#x)",
                    hwnd,
                    prev_ctx,
                )

        try:
            sent = user32.SendInput(
                len(events), arr, ctypes.sizeof(_INPUT)
            )
            logger.info(
                "Text inserted via SendInput KEYEVENTF_UNICODE: "
                "%d chars (%d/%d events)",
                len(text),
                sent,
                len(events),
            )
        finally:
            # --- Restore IME context ------------------------------------
            if prev_ctx and hwnd:
                # Wait for the message queue to process injected events
                time.sleep(0.05)
                imm32.ImmAssociateContext(hwnd, prev_ctx)
                logger.debug(
                    "IME context restored for hwnd=%#x", hwnd
                )
            if attached and target_tid:
                my_tid = kernel32.GetCurrentThreadId()
                user32.AttachThreadInput(my_tid, target_tid, False)

    def restore_clipboard_if_pending(self) -> None:
        """No-op: KEYEVENTF_UNICODE does not use the clipboard."""
