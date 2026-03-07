"""Tests for TextInserter abstractions and error handling."""

import pytest

from vox.config import InsertionConfig
from vox.inserter import ClipboardAccessError, PasteActionError, TextInserter


class FakeClipboard:
    def __init__(self, initial: str = "old") -> None:
        self.buffer = initial
        self.history: list[str] = []
        self.raise_on_paste = False
        self.raise_on_copy = False

    def paste(self) -> str:
        if self.raise_on_paste:
            raise RuntimeError("cannot read")
        return self.buffer

    def copy(self, text: str) -> None:
        if self.raise_on_copy:
            raise RuntimeError("cannot write")
        self.buffer = text
        self.history.append(text)


class FakePasteController:
    def __init__(self) -> None:
        self.calls = 0
        self.raise_on_paste = False

    def paste(self) -> None:
        self.calls += 1
        if self.raise_on_paste:
            raise RuntimeError("paste failed")


def build_inserter(clipboard: FakeClipboard, paste: FakePasteController) -> TextInserter:
    return TextInserter(
        InsertionConfig(),
        clipboard=clipboard,
        paste_controller=paste,
        sleep_fn=lambda _v: None,
    )


def test_insert_empty_text_skips() -> None:
    clipboard = FakeClipboard()
    paste = FakePasteController()
    inserter = build_inserter(clipboard, paste)

    inserter.insert("")

    assert paste.calls == 0
    assert clipboard.history == []


def test_insert_success_with_restore() -> None:
    clipboard = FakeClipboard(initial="original")
    paste = FakePasteController()
    inserter = build_inserter(clipboard, paste)

    inserter.insert("new text")

    assert paste.calls == 1
    assert clipboard.history == ["new text", "original"]
    assert clipboard.buffer == "original"


def test_insert_without_clipboard_capture_still_pastes() -> None:
    clipboard = FakeClipboard(initial="original")
    clipboard.raise_on_paste = True
    paste = FakePasteController()
    inserter = build_inserter(clipboard, paste)

    inserter.insert("new text")

    assert paste.calls == 1
    assert clipboard.history == ["new text"]
    assert clipboard.buffer == "new text"


def test_insert_raises_on_clipboard_write_failure() -> None:
    clipboard = FakeClipboard()
    clipboard.raise_on_copy = True
    paste = FakePasteController()
    inserter = build_inserter(clipboard, paste)

    with pytest.raises(ClipboardAccessError):
        inserter.insert("new text")

    assert paste.calls == 0


def test_insert_raises_on_paste_failure_and_restores() -> None:
    clipboard = FakeClipboard(initial="original")
    paste = FakePasteController()
    paste.raise_on_paste = True
    inserter = build_inserter(clipboard, paste)

    with pytest.raises(PasteActionError):
        inserter.insert("new text")

    assert paste.calls == 1
    assert clipboard.history == ["new text", "original"]
    assert clipboard.buffer == "original"
