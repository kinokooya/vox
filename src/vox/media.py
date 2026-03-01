"""Media playback control — auto-pause during recording (Windows).

Uses WASAPI COM API to detect audio output (peak level), and
VK_MEDIA_PLAY_PAUSE to toggle media playback.  No external
dependencies — only ctypes/windll (same pattern as inserter.py).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import sys

from vox.config import MediaConfig

logger = logging.getLogger(__name__)

# VK_MEDIA_PLAY_PAUSE key code
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_KEYUP = 0x0002

# COM constants
CLSCTX_ALL = 23  # CLSCTX_INPROC_SERVER | CLSCTX_INPROC_HANDLER | CLSCTX_LOCAL_SERVER
COINIT_MULTITHREADED = 0x0
E_DATA_FLOW_RENDER = 0  # eRender
E_ROLE_MULTIMEDIA = 1  # eMultimedia

# GUID bytes for WASAPI COM interfaces
# {A95664D2-9614-4F35-A746-DE8DB63617E6}  MMDeviceEnumerator
CLSID_MMDeviceEnumerator = bytes([
    0xD2, 0x64, 0x56, 0xA9, 0x14, 0x96, 0x35, 0x4F,
    0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6,
])
# {BCDE0395-E52F-467C-8E3D-C4579291692E}  IMMDeviceEnumerator
IID_IMMDeviceEnumerator = bytes([
    0x95, 0x03, 0xDE, 0xBC, 0x2F, 0xE5, 0x7C, 0x46,
    0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E,
])
# {C02216F6-8C67-4B5B-9D00-D008E73E0064}  IAudioMeterInformation
IID_IAudioMeterInformation = bytes([
    0xF6, 0x16, 0x22, 0xC0, 0x67, 0x8C, 0x5B, 0x4B,
    0x9D, 0x00, 0xD0, 0x08, 0xE7, 0x3E, 0x00, 0x64,
])

# Only define COM vtable structures on Windows
if sys.platform == "win32":

    class IMMDevice(ctypes.Structure):
        pass

    class IMMDeviceEnumerator(ctypes.Structure):
        pass

    class IAudioMeterInformation(ctypes.Structure):
        pass

    # IAudioMeterInformation vtable
    class IAudioMeterInformationVtbl(ctypes.Structure):
        _fields_ = [
            # IUnknown
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.CFUNCTYPE(ctypes.wintypes.ULONG, ctypes.c_void_p)),
            # IAudioMeterInformation
            ("GetPeakValue", ctypes.CFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
            )),
        ]

    IAudioMeterInformation._fields_ = [  # noqa: SLF001
        ("lpVtbl", ctypes.POINTER(IAudioMeterInformationVtbl)),
    ]

    # IMMDevice vtable
    class IMMDeviceVtbl(ctypes.Structure):
        _fields_ = [
            # IUnknown
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.CFUNCTYPE(ctypes.wintypes.ULONG, ctypes.c_void_p)),
            # IMMDevice
            ("Activate", ctypes.CFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.c_char_p,  # REFIID
                ctypes.wintypes.DWORD,  # dwClsCtx
                ctypes.c_void_p,  # pActivationParams
                ctypes.POINTER(ctypes.c_void_p),  # ppInterface
            )),
        ]

    IMMDevice._fields_ = [  # noqa: SLF001
        ("lpVtbl", ctypes.POINTER(IMMDeviceVtbl)),
    ]

    # IMMDeviceEnumerator vtable
    class IMMDeviceEnumeratorVtbl(ctypes.Structure):
        _fields_ = [
            # IUnknown
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.CFUNCTYPE(ctypes.wintypes.ULONG, ctypes.c_void_p)),
            # IMMDeviceEnumerator
            ("EnumAudioEndpoints", ctypes.c_void_p),
            ("GetDefaultAudioEndpoint", ctypes.CFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.wintypes.UINT,  # dataFlow
                ctypes.wintypes.UINT,  # role
                ctypes.POINTER(ctypes.c_void_p),  # ppEndpoint
            )),
        ]

    IMMDeviceEnumerator._fields_ = [  # noqa: SLF001
        ("lpVtbl", ctypes.POINTER(IMMDeviceEnumeratorVtbl)),
    ]


def _get_peak_value() -> float:
    """Get the current audio output peak level (0.0–1.0) via WASAPI COM.

    Returns 0.0 on any failure (non-Windows, COM error, etc.).
    """
    if sys.platform != "win32":
        return 0.0

    ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]

    hr = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    # S_OK=0, S_FALSE=1 (already initialized), RPC_E_CHANGED_MODE=0x80010106
    needs_uninit = hr == 0

    enumerator_ptr = ctypes.c_void_p()
    device_ptr = ctypes.c_void_p()
    meter_ptr = ctypes.c_void_p()

    try:
        # Create MMDeviceEnumerator
        hr = ole32.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            None,
            CLSCTX_ALL,
            IID_IMMDeviceEnumerator,
            ctypes.byref(enumerator_ptr),
        )
        if hr != 0:
            logger.debug("CoCreateInstance failed: 0x%08x", hr & 0xFFFFFFFF)
            return 0.0

        enumerator = ctypes.cast(
            enumerator_ptr, ctypes.POINTER(IMMDeviceEnumerator)
        ).contents

        # Get default audio render endpoint
        hr = enumerator.lpVtbl.contents.GetDefaultAudioEndpoint(
            enumerator_ptr,
            E_DATA_FLOW_RENDER,
            E_ROLE_MULTIMEDIA,
            ctypes.byref(device_ptr),
        )
        if hr != 0:
            logger.debug("GetDefaultAudioEndpoint failed: 0x%08x", hr & 0xFFFFFFFF)
            return 0.0

        device = ctypes.cast(
            device_ptr, ctypes.POINTER(IMMDevice)
        ).contents

        # Activate IAudioMeterInformation
        hr = device.lpVtbl.contents.Activate(
            device_ptr,
            IID_IAudioMeterInformation,
            CLSCTX_ALL,
            None,
            ctypes.byref(meter_ptr),
        )
        if hr != 0:
            logger.debug("Activate IAudioMeterInformation failed: 0x%08x", hr & 0xFFFFFFFF)
            return 0.0

        meter = ctypes.cast(
            meter_ptr, ctypes.POINTER(IAudioMeterInformation)
        ).contents

        # Get peak value
        peak = ctypes.c_float(0.0)
        hr = meter.lpVtbl.contents.GetPeakValue(
            meter_ptr,
            ctypes.byref(peak),
        )
        if hr != 0:
            logger.debug("GetPeakValue failed: 0x%08x", hr & 0xFFFFFFFF)
            return 0.0

        return float(peak.value)

    finally:
        # Release COM objects in reverse order
        if meter_ptr.value is not None:
            meter = ctypes.cast(
                meter_ptr, ctypes.POINTER(IAudioMeterInformation)
            ).contents
            meter.lpVtbl.contents.Release(meter_ptr)
        if device_ptr.value is not None:
            device = ctypes.cast(
                device_ptr, ctypes.POINTER(IMMDevice)
            ).contents
            device.lpVtbl.contents.Release(device_ptr)
        if enumerator_ptr.value is not None:
            enumerator = ctypes.cast(
                enumerator_ptr, ctypes.POINTER(IMMDeviceEnumerator)
            ).contents
            enumerator.lpVtbl.contents.Release(enumerator_ptr)
        if needs_uninit:
            ole32.CoUninitialize()


def _send_media_play_pause() -> None:
    """Send VK_MEDIA_PLAY_PAUSE key event to toggle media playback."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
    user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_KEYUP, 0)


class MediaController:
    """Auto-pause media during voice recording.

    Checks WASAPI audio peak level to detect active playback,
    then sends VK_MEDIA_PLAY_PAUSE to pause/resume.
    """

    def __init__(self, config: MediaConfig) -> None:
        self._config = config
        self._did_pause = False

    def pause_if_playing(self) -> None:
        """Pause media if audio is currently playing. Call on key press."""
        if not self._config.enabled:
            return
        try:
            peak = _get_peak_value()
            logger.debug("Audio peak level: %.4f", peak)
            if peak > self._config.peak_threshold:
                logger.info("Media playing (peak=%.4f), sending pause", peak)
                _send_media_play_pause()
                self._did_pause = True
            else:
                logger.debug("No media playing (peak=%.4f), skipping pause", peak)
                self._did_pause = False
        except Exception:
            logger.warning("Media pause failed, continuing", exc_info=True)
            self._did_pause = False

    def resume_if_we_paused(self) -> None:
        """Resume media if we previously paused it. Call on pipeline completion."""
        if not self._config.enabled:
            return
        if not self._did_pause:
            return
        try:
            logger.info("Resuming media playback")
            _send_media_play_pause()
        except Exception:
            logger.warning("Media resume failed, continuing", exc_info=True)
        finally:
            self._did_pause = False
