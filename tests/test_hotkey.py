"""Tests for hotkey listener and alt_gr alias handling."""

from unittest.mock import MagicMock

from pynput import keyboard

from vox.config import HotkeyConfig
from vox.hotkey import _KEY_ALIASES, _KEY_MAP, HotkeyListener


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


def test_release_while_disabled_does_not_fire_callback():
    """Release event while disabled should not fire on_release callback."""
    on_press = MagicMock()
    on_release = MagicMock()
    config = HotkeyConfig(trigger_key="alt_r")
    listener = HotkeyListener(config, on_press=on_press, on_release=on_release)

    # Press while enabled, then disable, then release
    listener._handle_press(keyboard.Key.alt_r)
    listener.set_enabled(False)
    listener._handle_release(keyboard.Key.alt_r)

    on_release.assert_not_called()


def test_set_enabled_false_resets_is_pressed():
    """set_enabled(False) should clear _is_pressed so stale state doesn't linger."""
    on_press = MagicMock()
    on_release = MagicMock()
    config = HotkeyConfig(trigger_key="alt_r")
    listener = HotkeyListener(config, on_press=on_press, on_release=on_release)

    listener._handle_press(keyboard.Key.alt_r)
    assert listener._is_pressed is True

    listener.set_enabled(False)
    assert listener._is_pressed is False


def test_release_without_press_does_not_fire_callback():
    """Release event without prior press should not fire on_release callback."""
    on_press = MagicMock()
    on_release = MagicMock()
    config = HotkeyConfig(trigger_key="alt_r")
    listener = HotkeyListener(config, on_press=on_press, on_release=on_release)

    listener._handle_release(keyboard.Key.alt_r)
    on_release.assert_not_called()
