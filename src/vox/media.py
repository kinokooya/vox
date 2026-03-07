"""Media playback control — auto-pause during recording (Windows).

Uses WASAPI COM API to detect audio output (peak level), and
VK_MEDIA_PLAY_PAUSE to toggle media playback.  No external
dependencies — only ctypes/windll (same pattern as inserter.py).

COM objects are cached for the lifetime of the MediaController to
avoid repeated CoInitialize/CoUninitialize cycles that conflict
with pynput's COM usage on the same listener thread.
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
E_DATA_FLOW_RENDER = 0  # eRender
E_ROLE_MULTIMEDIA = 1  # eMultimedia

# GUID bytes for WASAPI COM interfaces
# {BCDE0395-E52F-467C-8E3D-C4579291692E}  CLSID_MMDeviceEnumerator
CLSID_MMDeviceEnumerator = bytes([
    0x95, 0x03, 0xDE, 0xBC, 0x2F, 0xE5, 0x7C, 0x46,
    0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E,
])
# {A95664D2-9614-4F35-A746-DE8DB63617E6}  IID_IMMDeviceEnumerator
IID_IMMDeviceEnumerator = bytes([
    0xD2, 0x64, 0x56, 0xA9, 0x14, 0x96, 0x35, 0x4F,
    0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6,
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


def _release_com(ptr: ctypes.c_void_p | None, iface: type) -> None:
    """Release a COM interface pointer (safe, ignores errors)."""
    if ptr is not None and ptr.value is not None:
        try:
            obj = ctypes.cast(ptr, ctypes.POINTER(iface)).contents
            obj.lpVtbl.contents.Release(ptr)
        except Exception:
            pass


def _send_media_play_pause() -> None:
    """Send VK_MEDIA_PLAY_PAUSE key event to toggle media playback."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
    user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_KEYUP, 0)


class MediaController:
    """Auto-pause media during voice recording.

    Caches WASAPI COM objects for the default audio render endpoint
    to avoid repeated COM initialization/teardown that conflicts with
    pynput's listener thread.
    """

    def __init__(self, config: MediaConfig) -> None:
        self._config = config
        self._did_pause = False
        # Cached COM pointers (lazily initialized on first use)
        self._enumerator_ptr: ctypes.c_void_p | None = None
        self._device_ptr: ctypes.c_void_p | None = None
        self._meter_ptr: ctypes.c_void_p | None = None
        self._com_initialized = False

    def _init_meter(self) -> bool:
        """Create and cache WASAPI COM objects. Returns True on success."""
        if self._meter_ptr is not None:
            return True
        if sys.platform != "win32":
            return False

        ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]

        # Initialize COM (once per thread, kept alive until close())
        if not self._com_initialized:
            hr = ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
            # S_OK=0, S_FALSE=1: success. 0x80010106: different apartment, still usable.
            if hr < 0 and (hr & 0xFFFFFFFF) != 0x80010106:
                logger.warning("CoInitializeEx failed: 0x%08x", hr & 0xFFFFFFFF)
                return False
            self._com_initialized = True

        # Create MMDeviceEnumerator
        enumerator_ptr = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            CLSID_MMDeviceEnumerator, None, CLSCTX_ALL,
            IID_IMMDeviceEnumerator, ctypes.byref(enumerator_ptr),
        )
        if hr != 0:
            logger.warning("CoCreateInstance(MMDeviceEnumerator) failed: 0x%08x", hr & 0xFFFFFFFF)
            return False

        # Get default audio render endpoint
        device_ptr = ctypes.c_void_p()
        enumerator = ctypes.cast(
            enumerator_ptr, ctypes.POINTER(IMMDeviceEnumerator)
        ).contents
        hr = enumerator.lpVtbl.contents.GetDefaultAudioEndpoint(
            enumerator_ptr, E_DATA_FLOW_RENDER, E_ROLE_MULTIMEDIA,
            ctypes.byref(device_ptr),
        )
        if hr != 0:
            logger.warning("GetDefaultAudioEndpoint failed: 0x%08x", hr & 0xFFFFFFFF)
            _release_com(enumerator_ptr, IMMDeviceEnumerator)
            return False

        # Activate IAudioMeterInformation
        meter_ptr = ctypes.c_void_p()
        device = ctypes.cast(device_ptr, ctypes.POINTER(IMMDevice)).contents
        hr = device.lpVtbl.contents.Activate(
            device_ptr, IID_IAudioMeterInformation, CLSCTX_ALL,
            None, ctypes.byref(meter_ptr),
        )
        if hr != 0:
            logger.warning("Activate(IAudioMeterInformation) failed: 0x%08x", hr & 0xFFFFFFFF)
            _release_com(device_ptr, IMMDevice)
            _release_com(enumerator_ptr, IMMDeviceEnumerator)
            return False

        # Success — cache all pointers
        self._enumerator_ptr = enumerator_ptr
        self._device_ptr = device_ptr
        self._meter_ptr = meter_ptr
        logger.info("WASAPI audio meter initialized")
        return True

    def _release_meter(self) -> None:
        """Release cached COM objects (reverse order)."""
        if sys.platform != "win32":
            return
        _release_com(self._meter_ptr, IAudioMeterInformation)
        self._meter_ptr = None
        _release_com(self._device_ptr, IMMDevice)
        self._device_ptr = None
        _release_com(self._enumerator_ptr, IMMDeviceEnumerator)
        self._enumerator_ptr = None

    def _get_peak(self) -> float:
        """Get current audio output peak level (0.0–1.0).

        Uses cached COM meter; re-initializes on failure.
        """
        if not self._init_meter():
            return 0.0

        meter = ctypes.cast(
            self._meter_ptr, ctypes.POINTER(IAudioMeterInformation)
        ).contents
        peak = ctypes.c_float(0.0)
        hr = meter.lpVtbl.contents.GetPeakValue(
            self._meter_ptr, ctypes.byref(peak),
        )
        if hr != 0:
            # Meter may be stale (e.g. audio device changed), retry once
            logger.info("GetPeakValue failed (0x%08x), re-initializing", hr & 0xFFFFFFFF)
            self._release_meter()
            if not self._init_meter():
                return 0.0
            meter = ctypes.cast(
                self._meter_ptr, ctypes.POINTER(IAudioMeterInformation)
            ).contents
            hr = meter.lpVtbl.contents.GetPeakValue(
                self._meter_ptr, ctypes.byref(peak),
            )
            if hr != 0:
                logger.warning("GetPeakValue retry failed: 0x%08x", hr & 0xFFFFFFFF)
                self._release_meter()
                return 0.0

        return float(peak.value)

    def pause_if_playing(self) -> None:
        """Pause media if audio is currently playing. Call on key press."""
        if not self._config.enabled:
            return
        try:
            peak = self._get_peak()
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

    def close(self) -> None:
        """Release COM resources."""
        self._release_meter()
