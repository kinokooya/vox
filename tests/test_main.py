"""Tests for entrypoint behavior in vox.__main__."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import vox.__main__ as entry


class FakeApp:
    def __init__(self, _config) -> None:
        self.events: list[str] = ["init"]

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")


def test_main_uses_cli_config_and_stops_on_icon_error(monkeypatch, tmp_path: Path) -> None:
    loaded_paths = []
    app_holder = {}

    def fake_load_config(path):
        loaded_paths.append(path)
        return object()

    def fake_app_ctor(config):
        app = FakeApp(config)
        app_holder["app"] = app
        return app

    class FakeIcon:
        def run(self) -> None:
            raise RuntimeError("tray failed")

        def stop(self) -> None:
            return None

    monkeypatch.setattr(entry, "_register_cuda_dll_dirs", lambda: None)
    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_kill_existing_instances", lambda: None)
    monkeypatch.setattr(entry, "_pid_path", lambda: tmp_path / "vox.pid")
    monkeypatch.setattr(entry, "load_config", fake_load_config)
    monkeypatch.setattr(entry, "VoxApp", fake_app_ctor)
    monkeypatch.setattr(entry, "create_tray_icon", lambda _on_quit: FakeIcon())
    monkeypatch.setattr(entry.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["vox", "custom.yaml"])

    with pytest.raises(RuntimeError, match="tray failed"):
        entry.main()

    assert str(loaded_paths[0]).endswith("custom.yaml")
    assert app_holder["app"].events == ["init", "start", "stop"]


def test_main_cleanup_from_tray_quit(monkeypatch, tmp_path: Path) -> None:
    app_holder = {}

    def fake_app_ctor(config):
        app = FakeApp(config)
        app_holder["app"] = app
        return app

    class FakeIcon:
        def __init__(self, on_quit) -> None:
            self._on_quit = on_quit
            self.stop_called = 0

        def run(self) -> None:
            self._on_quit()

        def stop(self) -> None:
            self.stop_called += 1

    icon_holder = {}

    def fake_create_icon(on_quit):
        icon = FakeIcon(on_quit)
        icon_holder["icon"] = icon
        return icon

    monkeypatch.setattr(entry, "_register_cuda_dll_dirs", lambda: None)
    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_kill_existing_instances", lambda: None)
    monkeypatch.setattr(entry, "_pid_path", lambda: tmp_path / "vox.pid")
    monkeypatch.setattr(entry, "load_config", lambda _path: object())
    monkeypatch.setattr(entry, "VoxApp", fake_app_ctor)
    monkeypatch.setattr(entry, "create_tray_icon", fake_create_icon)
    monkeypatch.setattr(entry.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["vox"])

    entry.main()

    # stop() is called once via tray cleanup, and once in finally.
    assert app_holder["app"].events == ["init", "start", "stop", "stop"]
    assert icon_holder["icon"].stop_called == 1
