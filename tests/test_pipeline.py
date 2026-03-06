"""Tests for PipelineRunner."""

import numpy as np

from vox.pipeline import PipelineRunner


class StubRecorder:
    def __init__(self, audio: np.ndarray) -> None:
        self._audio = audio

    def stop(self) -> np.ndarray:
        return self._audio


class StubSTT:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.calls += 1
        return self._text


class StubLLM:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def format_text(self, raw_text: str) -> str:
        self.calls += 1
        return self._text


class StubInserter:
    def __init__(self) -> None:
        self.inserted: list[str] = []

    def insert(self, text: str) -> None:
        self.inserted.append(text)


RunnerParts = tuple[PipelineRunner, StubSTT, StubLLM, StubInserter]


def build_runner(
    audio: np.ndarray,
    stt_text: str,
    llm_text: str,
) -> RunnerParts:
    recorder = StubRecorder(audio)
    stt = StubSTT(stt_text)
    llm = StubLLM(llm_text)
    inserter = StubInserter()
    runner = PipelineRunner(
        recorder=recorder,
        stt=stt,
        llm=llm,
        inserter=inserter,
        sample_rate=16000,
    )
    return runner, stt, llm, inserter


def test_pipeline_skips_on_empty_audio() -> None:
    runner, stt, llm, inserter = build_runner(
        np.array([], dtype=np.float32),
        "raw",
        "formatted",
    )

    assert runner.run_once() is False
    assert stt.calls == 0
    assert llm.calls == 0
    assert inserter.inserted == []


def test_pipeline_skips_on_empty_stt_result() -> None:
    runner, stt, llm, inserter = build_runner(
        np.zeros(16000, dtype=np.float32),
        "   ",
        "formatted",
    )

    assert runner.run_once() is False
    assert stt.calls == 1
    assert llm.calls == 0
    assert inserter.inserted == []


def test_pipeline_skips_on_empty_llm_result() -> None:
    runner, stt, llm, inserter = build_runner(
        np.zeros(16000, dtype=np.float32),
        "raw",
        "   ",
    )

    assert runner.run_once() is False
    assert stt.calls == 1
    assert llm.calls == 1
    assert inserter.inserted == []


def test_pipeline_inserts_formatted_text() -> None:
    runner, stt, llm, inserter = build_runner(
        np.zeros(16000, dtype=np.float32),
        "raw",
        "formatted",
    )

    assert runner.run_once() is True
    assert stt.calls == 1
    assert llm.calls == 1
    assert inserter.inserted == ["formatted"]
