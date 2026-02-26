"""Tests for hotkey listener and alt_gr alias handling."""

from unittest.mock import MagicMock

from pynput import keyboard

from vox.config import HotkeyConfig
from vox.hotkey import HotkeyListener, _KEY_ALIASES, _KEY_MAP


def test_key_aliases_alt_r_contains_alt_gr():
    """alt_r alias set should include both alt_r and alt_gr."""
    assert keyboard.Key.alt_r in _KEY_ALIASES["alt_r"]
    assert keyboard.Key.alt_gr in _KEY_ALIASES["alt_r"]


def test_key_map_contains_alt_gr():
    """_KEY_MAP should have an explicit alt_gr entry."""
    assert "alt_gr" in _KEY_MAP
    assert _KEY_MAP["alt_gr"] == keyboard.Key.alt_gr


def test_hotkey_listener_trigger_keys_include_alt_gr():
    """HotkeyListener configured with alt_r should accept both alt_r and alt_gr."""
    config = HotkeyConfig(trigger_key="alt_r")
    listener = HotkeyListener(config, on_press=MagicMock(), on_release=MagicMock())
    assert keyboard.Key.alt_r in listener._trigger_keys
    assert keyboard.Key.alt_gr in listener._trigger_keys


def test_hotkey_listener_alt_gr_press_fires_callback():
    """Pressing alt_gr should fire the on_press callback when trigger is alt_r."""
    on_press = MagicMock()
    on_release = MagicMock()
    config = HotkeyConfig(trigger_key="alt_r")
    listener = HotkeyListener(config, on_press=on_press, on_release=on_release)

    # Simulate an alt_gr key press via the internal handler
    listener._handle_press(keyboard.Key.alt_gr)
    on_press.assert_called_once()


def test_hotkey_listener_alt_gr_release_fires_callback():
    """Releasing alt_gr should fire the on_release callback when trigger is alt_r."""
    on_press = MagicMock()
    on_release = MagicMock()
    config = HotkeyConfig(trigger_key="alt_r")
    listener = HotkeyListener(config, on_press=on_press, on_release=on_release)

    # Must press first to set _is_pressed
    listener._handle_press(keyboard.Key.alt_gr)
    listener._handle_release(keyboard.Key.alt_gr)
    on_release.assert_called_once()
