"""Tests for HotkeyListener behavior."""

from __future__ import annotations

from vox.config import HotkeyConfig
from vox.hotkey import HotkeyListener


class FakeListenerImpl:
    def __init__(self, on_press, on_release) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.daemon = False
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_press_release_callbacks(monkeypatch) -> None:
    events: list[str] = []
    listener = HotkeyListener(
        HotkeyConfig(trigger_key="alt_r"),
        on_press=lambda: events.append("press"),
        on_release=lambda: events.append("release"),
    )

    key = listener._trigger_key
    assert key is not None

    listener._handle_press(key)
    listener._handle_press(key)
    listener._handle_release(key)
    listener._handle_release(key)

    assert events == ["press", "release"]


def test_disabled_listener_ignores_press() -> None:
    events: list[str] = []
    listener = HotkeyListener(
        HotkeyConfig(trigger_key="alt_r"),
        on_press=lambda: events.append("press"),
        on_release=lambda: events.append("release"),
    )
    key = listener._trigger_key
    assert key is not None

    listener.set_enabled(False)
    listener._handle_press(key)

    assert events == []


def test_start_and_stop(monkeypatch) -> None:
    monkeypatch.setattr("vox.hotkey.keyboard.Listener", FakeListenerImpl)

    listener = HotkeyListener(
        HotkeyConfig(trigger_key="alt_r"),
        on_press=lambda: None,
        on_release=lambda: None,
    )

    listener.start()
    assert isinstance(listener._listener, FakeListenerImpl)
    assert listener._listener.daemon is True
    assert listener._listener.started is True

    listener.stop()
    assert listener._listener.stopped is True


def test_unknown_trigger_key_raises() -> None:
    try:
        HotkeyListener(
            HotkeyConfig(trigger_key="bad_key"),
            on_press=lambda: None,
            on_release=lambda: None,
        )
    except ValueError as err:
        assert "Unknown trigger key" in str(err)
    else:
        raise AssertionError("Expected ValueError")
