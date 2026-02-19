"""Push-to-Talk hotkey listener using pynput."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from pynput import keyboard

from vox.config import HotkeyConfig

logger = logging.getLogger(__name__)

# Mapping of config key names to pynput Key objects
_KEY_MAP: dict[str, keyboard.Key] = {
    "alt_r": keyboard.Key.alt_r,
    "alt_l": keyboard.Key.alt_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl_l": keyboard.Key.ctrl_l,
}


class HotkeyListener:
    """Listens for Push-to-Talk key events (hold mode)."""

    def __init__(
        self,
        config: HotkeyConfig,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> None:
        self._trigger_key = _KEY_MAP.get(config.trigger_key)
        if self._trigger_key is None:
            supported = list(_KEY_MAP)
            raise ValueError(f"Unknown trigger key: {config.trigger_key}. Supported: {supported}")
        self._on_press = on_press
        self._on_release = on_release
        self._listener: keyboard.Listener | None = None
        self._is_pressed = False
        self._enabled = True
        self._lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable hotkey processing."""
        with self._lock:
            self._enabled = enabled

    def start(self) -> None:
        """Start listening for hotkey events in a background thread."""
        self._listener = keyboard.Listener(
            on_press=self._handle_press,
            on_release=self._handle_release,
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("Hotkey listener started (trigger: %s)", self._trigger_key)

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener:
            self._listener.stop()
            logger.info("Hotkey listener stopped")

    def _handle_press(self, key: Any) -> None:
        if key != self._trigger_key:
            return
        with self._lock:
            if not self._enabled or self._is_pressed:
                return
            self._is_pressed = True
        logger.debug("Trigger key pressed")
        self._on_press()

    def _handle_release(self, key: Any) -> None:
        if key != self._trigger_key:
            return
        with self._lock:
            if not self._is_pressed:
                return
            self._is_pressed = False
        logger.debug("Trigger key released")
        self._on_release()
