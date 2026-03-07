"""Tests for VoxApp state transitions and shutdown behavior."""

from __future__ import annotations

import threading
import time

from vox.app import AppState, VoxApp
from vox.config import AppConfig


class FakeRecorder:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self):
        self.stopped += 1
        return []


class FakeSTT:
    def load_model(self) -> None:
        return None

    def get_vram_usage_mb(self) -> int:
        return 1


class FakeLLM:
    pass


class FakeInserter:
    pass


class FakeHotkey:
    def __init__(self, _config, on_press, on_release) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.start_calls = 0
        self.stop_calls = 0
        self.enabled_values: list[bool] = []

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def set_enabled(self, enabled: bool) -> None:
        self.enabled_values.append(enabled)


class FastPipeline:
    def run_once(self) -> bool:
        return True


class BlockingPipeline:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.allow_finish = threading.Event()

    def run_once(self) -> bool:
        self.started.set()
        self.allow_finish.wait(timeout=1.0)
        return True


def build_app(monkeypatch, pipeline):
    recorder = FakeRecorder()
    stt = FakeSTT()
    hotkey_holder: dict[str, FakeHotkey] = {}

    def fake_create_recorder(_config):
        return recorder

    def fake_create_stt_engine(_stt_config):
        return stt

    def fake_hotkey_ctor(config, on_press, on_release):
        hotkey = FakeHotkey(config, on_press, on_release)
        hotkey_holder["instance"] = hotkey
        return hotkey

    monkeypatch.setattr("vox.app._create_recorder", fake_create_recorder)
    monkeypatch.setattr("vox.app.create_stt_engine", fake_create_stt_engine)
    monkeypatch.setattr("vox.app.LLMFormatter", lambda _cfg: FakeLLM())
    monkeypatch.setattr("vox.app.TextInserter", lambda _cfg: FakeInserter())
    monkeypatch.setattr("vox.app.PipelineRunner", lambda **_kwargs: pipeline)
    monkeypatch.setattr("vox.app.HotkeyListener", fake_hotkey_ctor)

    app = VoxApp(AppConfig())
    return app, recorder, stt, hotkey_holder["instance"]


def test_stop_is_idempotent(monkeypatch) -> None:
    app, _recorder, _stt, hotkey = build_app(monkeypatch, FastPipeline())

    app.start()
    app.stop()
    app.stop()

    assert app._state == AppState.STOPPED
    assert hotkey.stop_calls == 1
    assert hotkey.enabled_values[-1] is False


def test_stop_during_processing_keeps_hotkey_disabled(monkeypatch) -> None:
    pipeline = BlockingPipeline()
    app, _recorder, _stt, hotkey = build_app(monkeypatch, pipeline)

    app.start()
    app._on_key_press()
    app._on_key_release()

    assert pipeline.started.wait(timeout=1.0)
    assert app._state == AppState.PROCESSING

    stopper = threading.Thread(target=app.stop)
    stopper.start()
    time.sleep(0.05)
    pipeline.allow_finish.set()
    stopper.join(timeout=1.0)

    assert not stopper.is_alive()
    assert app._state == AppState.STOPPED
    assert hotkey.enabled_values[-1] is False
