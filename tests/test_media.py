"""Tests for MediaController — media auto-pause during recording."""

from unittest.mock import patch

from vox.config import MediaConfig
from vox.media import MediaController


class TestMediaControllerDisabled:
    def test_disabled_pause_is_noop(self):
        """When disabled, pause_if_playing does nothing."""
        ctrl = MediaController(MediaConfig(enabled=False))
        # Should not raise, should not set _did_pause
        ctrl.pause_if_playing()
        assert ctrl._did_pause is False  # noqa: SLF001

    def test_disabled_resume_is_noop(self):
        """When disabled, resume_if_we_paused does nothing."""
        ctrl = MediaController(MediaConfig(enabled=False))
        ctrl._did_pause = True  # noqa: SLF001
        ctrl.resume_if_we_paused()
        # _did_pause should remain True (not touched because enabled=False)
        assert ctrl._did_pause is True  # noqa: SLF001


class TestMediaControllerPause:
    @patch("vox.media._send_media_play_pause")
    @patch("vox.media._get_peak_value", return_value=0.5)
    def test_high_peak_triggers_pause(self, mock_peak, mock_send):
        """When peak is above threshold, media key is sent and _did_pause is set."""
        ctrl = MediaController(MediaConfig(enabled=True, peak_threshold=0.01))
        ctrl.pause_if_playing()
        mock_send.assert_called_once()
        assert ctrl._did_pause is True  # noqa: SLF001

    @patch("vox.media._send_media_play_pause")
    @patch("vox.media._get_peak_value", return_value=0.001)
    def test_low_peak_skips_pause(self, mock_peak, mock_send):
        """When peak is below threshold, no media key is sent."""
        ctrl = MediaController(MediaConfig(enabled=True, peak_threshold=0.01))
        ctrl.pause_if_playing()
        mock_send.assert_not_called()
        assert ctrl._did_pause is False  # noqa: SLF001


class TestMediaControllerResume:
    @patch("vox.media._send_media_play_pause")
    def test_resume_when_we_paused(self, mock_send):
        """When _did_pause is True, resume sends media key and resets flag."""
        ctrl = MediaController(MediaConfig(enabled=True))
        ctrl._did_pause = True  # noqa: SLF001
        ctrl.resume_if_we_paused()
        mock_send.assert_called_once()
        assert ctrl._did_pause is False  # noqa: SLF001

    @patch("vox.media._send_media_play_pause")
    def test_resume_skipped_when_we_did_not_pause(self, mock_send):
        """When _did_pause is False, resume does nothing."""
        ctrl = MediaController(MediaConfig(enabled=True))
        ctrl._did_pause = False  # noqa: SLF001
        ctrl.resume_if_we_paused()
        mock_send.assert_not_called()


class TestMediaControllerErrorSafety:
    @patch("vox.media._get_peak_value", side_effect=OSError("COM error"))
    def test_pause_failure_does_not_propagate(self, mock_peak):
        """Exceptions in pause_if_playing are caught and logged."""
        ctrl = MediaController(MediaConfig(enabled=True))
        ctrl.pause_if_playing()  # should not raise
        assert ctrl._did_pause is False  # noqa: SLF001

    @patch("vox.media._send_media_play_pause", side_effect=OSError("key error"))
    def test_resume_failure_does_not_propagate(self, mock_send):
        """Exceptions in resume_if_we_paused are caught, _did_pause is reset."""
        ctrl = MediaController(MediaConfig(enabled=True))
        ctrl._did_pause = True  # noqa: SLF001
        ctrl.resume_if_we_paused()  # should not raise
        assert ctrl._did_pause is False  # noqa: SLF001
