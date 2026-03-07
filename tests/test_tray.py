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
    assert icon.icon.size == (64, 64)


def test_menu_structure():
    """The tray menu should contain 3 actionable items and at least one separator."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    items = _non_separator_items(icon)
    assert len(items) == 3

    all_items = list(icon.menu.items)
    separators = [item for item in all_items if str(item) == _SEPARATOR_LABEL]
    assert len(separators) >= 1


@patch("vox.tray.os.startfile", create=True)
def test_open_config_calls_startfile(mock_startfile):
    """Selecting first item should open config.yaml."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    config_item = _non_separator_items(icon)[0]
    config_item(icon)

    mock_startfile.assert_called_once()
    call_arg = mock_startfile.call_args[0][0]
    assert "config.yaml" in call_arg


@patch("vox.tray.os.startfile", create=True)
def test_open_log_calls_startfile(mock_startfile):
    """Selecting second item should open vox.log."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    log_item = _non_separator_items(icon)[1]
    log_item(icon)

    mock_startfile.assert_called_once()
    call_arg = mock_startfile.call_args[0][0]
    assert "vox.log" in call_arg


def test_quit_invokes_on_quit_callback():
    """Selecting quit item should call the callback."""
    on_quit = MagicMock()
    icon = create_tray_icon(on_quit)

    quit_item = _non_separator_items(icon)[-1]
    quit_item(icon)

    on_quit.assert_called_once()
