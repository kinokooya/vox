"""Text insertion via clipboard + Ctrl+V (Windows)."""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Protocol, cast

import pyperclip

from vox.config import InsertionConfig

logger = logging.getLogger(__name__)

# Win32 virtual key codes
VK_CONTROL = 0x11
VK_V = 0x56
KEYEVENTF_KEYUP = 0x0002


class InsertionError(RuntimeError):
    """Base error for insertion failures."""


class ClipboardAccessError(InsertionError):
    """Clipboard read/write operation failed."""


class PasteActionError(InsertionError):
    """Paste key action failed."""


class ClipboardProtocol(Protocol):
    def paste(self) -> str: ...

    def copy(self, text: str) -> None: ...


class PasteControllerProtocol(Protocol):
    def paste(self) -> None: ...


class PyperclipClipboard:
    def paste(self) -> str:
        return cast(str, pyperclip.paste())

    def copy(self, text: str) -> None:
        pyperclip.copy(text)


class Win32PasteController:
    def paste(self) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(VK_V, 0, 0, 0)
        user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


class TextInserter:
    """Inserts text into the active text field via clipboard paste."""

    def __init__(
        self,
        config: InsertionConfig,
        clipboard: ClipboardProtocol | None = None,
        paste_controller: PasteControllerProtocol | None = None,
        sleep_fn=time.sleep,
    ) -> None:
        self._config = config
        self._clipboard = clipboard or PyperclipClipboard()
        self._paste_controller = paste_controller or Win32PasteController()
        self._sleep = sleep_fn

    def insert(self, text: str) -> None:
        """Insert text into the active window.

        Saves the current clipboard, sets the text, sends Ctrl+V,
        then restores the original clipboard content.

        Raises:
            ClipboardAccessError: clipboard write failed.
            PasteActionError: paste key action failed.
        """
        if not text:
            logger.warning("Empty text, skipping insertion")
            return

        captured = False
        original_clipboard = ""
        if self._config.restore_clipboard:
            try:
                original_clipboard = self._clipboard.paste()
                captured = True
            except Exception as err:  # noqa: BLE001
                logger.warning("Could not read clipboard, will not restore: %s", err)

        try:
            self._clipboard.copy(text)
        except Exception as err:  # noqa: BLE001
            logger.error("Clipboard write failed: %s", err)
            raise ClipboardAccessError("Could not write clipboard") from err

        self._sleep(self._config.pre_paste_delay_ms / 1000.0)

        try:
            self._paste_controller.paste()
            logger.info("Text inserted: %d chars", len(text))
        except Exception as err:  # noqa: BLE001
            logger.error("Paste action failed: %s", err)
            raise PasteActionError("Could not trigger paste action") from err
        finally:
            if self._config.restore_clipboard and captured:
                self._sleep(0.1)
                try:
                    self._clipboard.copy(original_clipboard)
                except Exception as err:  # noqa: BLE001
                    logger.warning("Could not restore clipboard: %s", err)
