"""Tests for the system tray icon module."""

from unittest.mock import MagicMock, patch

from vox.tray import create_tray_icon

_SEPARATOR_LABEL = "- - - -"


def _non_separator_items(icon):
    """Return menu items excluding separators."""
    return [item for item in icon.menu.items if str(item) != _SEPARATOR_LABEL]


def test_create_tray_icon_returns_icon():
    """create_tray_icon should return a pystray Icon instance."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    assert icon.name == "Vox"
    assert icon.title == "Vox - AI Voice Input"
    assert icon.icon is not None
    # Icon image should be 64x64 RGBA
    assert icon.icon.size == (64, 64)


def test_menu_has_expected_items():
    """The tray menu should have the expected labels."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    items = _non_separator_items(icon)
    labels = [str(item) for item in items]

    assert "設定ファイルを開く" in labels
    assert "ログを開く" in labels
    assert "終了" in labels


def test_menu_has_separator():
    """The menu should contain a separator before the quit item."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    items = list(icon.menu.items)
    separators = [i for i, item in enumerate(items) if str(item) == _SEPARATOR_LABEL]
    assert len(separators) >= 1


@patch("vox.tray.os.startfile")
def test_open_config_calls_startfile(mock_startfile):
    """Selecting 'Open config' should call os.startfile with config.yaml path."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    items = _non_separator_items(icon)
    config_item = items[0]  # First item is "設定ファイルを開く"

    # pystray MenuItem.__call__(icon) triggers the action
    config_item(icon)

    mock_startfile.assert_called_once()
    call_arg = mock_startfile.call_args[0][0]
    assert "config.yaml" in call_arg


@patch("vox.tray.os.startfile")
def test_open_log_calls_startfile(mock_startfile):
    """Selecting 'Open log' should call os.startfile with vox.log path."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    items = _non_separator_items(icon)
    log_item = items[1]  # Second item is "ログを開く"

    log_item(icon)

    mock_startfile.assert_called_once()
    call_arg = mock_startfile.call_args[0][0]
    assert "vox.log" in call_arg


def test_quit_invokes_on_quit_callback():
    """Selecting 'Quit' should invoke the on_quit callback."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    items = _non_separator_items(icon)
    quit_item = items[-1]  # Last non-separator item is "終了"

    quit_item(icon)

    on_quit.assert_called_once()
