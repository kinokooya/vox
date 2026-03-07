"""Tests for entrypoint behavior in vox.__main__."""

from __future__ import annotations

import sys

import pytest

import vox.__main__ as entry


def test_main_loads_config_and_stops_on_error(monkeypatch) -> None:
    loaded_paths = []
    app_events = []

    class FakeApp:
        def __init__(self, _config) -> None:
            app_events.append("init")

        def start(self) -> None:
            app_events.append("start")

        def stop(self) -> None:
            app_events.append("stop")

    def fake_load_config(path):
        loaded_paths.append(path)
        return object()

    monkeypatch.setattr(entry, "load_config", fake_load_config)
    monkeypatch.setattr(entry, "VoxApp", FakeApp)
    monkeypatch.setattr(entry.signal, "signal", lambda *_args, **_kwargs: None)

    def raise_runtime() -> None:
        raise RuntimeError("done")

    monkeypatch.setattr(entry.signal, "pause", raise_runtime)
    monkeypatch.setattr(sys, "argv", ["vox", "custom.yaml"])

    with pytest.raises(RuntimeError, match="done"):
        entry.main()

    assert str(loaded_paths[0]).endswith("custom.yaml")
    assert app_events == ["init", "start", "stop"]


def test_main_windows_fallback_path(monkeypatch) -> None:
    app_events = []

    class FakeApp:
        def __init__(self, _config) -> None:
            app_events.append("init")

        def start(self) -> None:
            app_events.append("start")

        def stop(self) -> None:
            app_events.append("stop")

    class FakeEvent:
        def wait(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(entry, "load_config", lambda _path: object())
    monkeypatch.setattr(entry, "VoxApp", FakeApp)
    monkeypatch.setattr(entry.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(entry.signal, "pause", lambda: (_ for _ in ()).throw(AttributeError()))
    monkeypatch.setattr("threading.Event", FakeEvent)
    monkeypatch.setattr(sys, "argv", ["vox"])

    entry.main()

    assert app_events == ["init", "start", "stop"]
