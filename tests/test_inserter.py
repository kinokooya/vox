"""Tests for text insertion (paste method selection logic)."""

from unittest.mock import MagicMock, patch

from vox.config import InsertionConfig
from vox.inserter import TextInserter, _is_chromium_window

# ---------------------------------------------------------------------------
# _is_chromium_window detection
# ---------------------------------------------------------------------------


@patch("vox.inserter._get_foreground_class_name")
def test_is_chromium_window_chrome(mock_class_name: MagicMock):
    mock_class_name.return_value = "Chrome_WidgetWin_1"
    assert _is_chromium_window() is True


@patch("vox.inserter._get_foreground_class_name")
def test_is_chromium_window_notepad(mock_class_name: MagicMock):
    mock_class_name.return_value = "Notepad"
    assert _is_chromium_window() is False


@patch("vox.inserter._get_foreground_class_name")
def test_is_chromium_window_empty(mock_class_name: MagicMock):
    mock_class_name.return_value = ""
    assert _is_chromium_window() is False


# ---------------------------------------------------------------------------
# TextInserter.insert — paste method selection
# ---------------------------------------------------------------------------


@patch("vox.inserter._send_ctrl_v")
@patch("vox.inserter._get_focused_hwnd", return_value=0x1234)
@patch("vox.inserter._set_clipboard_text")
@patch("vox.inserter._get_clipboard_text", return_value=None)
@patch("vox.inserter._is_chromium_window", return_value=True)
def test_auto_mode_chromium_uses_ctrl_v(
    _mock_is_chromium: MagicMock,
    _mock_get_cb: MagicMock,
    _mock_set_cb: MagicMock,
    _mock_focused: MagicMock,
    mock_ctrl_v: MagicMock,
):
    config = InsertionConfig(method="auto", pre_paste_delay_ms=0)
    inserter = TextInserter(config)
    inserter.insert("hello")
    mock_ctrl_v.assert_called_once()


@patch("vox.inserter.user32")
@patch("vox.inserter._get_focused_hwnd", return_value=0x1234)
@patch("vox.inserter._set_clipboard_text")
@patch("vox.inserter._get_clipboard_text", return_value=None)
@patch("vox.inserter._is_chromium_window", return_value=False)
def test_auto_mode_native_uses_wm_paste(
    _mock_is_chromium: MagicMock,
    _mock_get_cb: MagicMock,
    _mock_set_cb: MagicMock,
    mock_focused: MagicMock,
    mock_user32: MagicMock,
):
    config = InsertionConfig(method="auto", pre_paste_delay_ms=0)
    inserter = TextInserter(config)
    inserter.insert("hello")
    mock_user32.SendMessageW.assert_called_once_with(0x1234, 0x0302, 0, 0)


@patch("vox.inserter.user32")
@patch("vox.inserter._get_focused_hwnd", return_value=0x1234)
@patch("vox.inserter._set_clipboard_text")
@patch("vox.inserter._get_clipboard_text", return_value=None)
def test_wm_paste_forced(
    _mock_get_cb: MagicMock,
    _mock_set_cb: MagicMock,
    mock_focused: MagicMock,
    mock_user32: MagicMock,
):
    config = InsertionConfig(method="wm_paste", pre_paste_delay_ms=0)
    inserter = TextInserter(config)
    inserter.insert("hello")
    mock_user32.SendMessageW.assert_called_once_with(0x1234, 0x0302, 0, 0)


@patch("vox.inserter._send_ctrl_v")
@patch("vox.inserter._set_clipboard_text")
@patch("vox.inserter._get_clipboard_text", return_value=None)
def test_ctrl_v_forced(
    _mock_get_cb: MagicMock,
    _mock_set_cb: MagicMock,
    mock_ctrl_v: MagicMock,
):
    config = InsertionConfig(method="ctrl_v", pre_paste_delay_ms=0)
    inserter = TextInserter(config)
    inserter.insert("hello")
    mock_ctrl_v.assert_called_once()


@patch("vox.inserter._set_clipboard_text")
@patch("vox.inserter._get_clipboard_text", return_value=None)
def test_insert_empty_text_skips(
    _mock_get_cb: MagicMock,
    mock_set_cb: MagicMock,
):
    config = InsertionConfig(pre_paste_delay_ms=0)
    inserter = TextInserter(config)
    inserter.insert("")
    mock_set_cb.assert_not_called()
