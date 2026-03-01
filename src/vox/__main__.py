"""Entry point for Vox: python -m vox"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from vox.app import VoxApp
from vox.config import load_config
from vox.tray import create_tray_icon


def _register_cuda_dll_dirs() -> None:
    """Add NVIDIA CUDA DLL directories installed via pip to the DLL search path."""
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site_packages.exists():
        return
    for bin_dir in site_packages.glob("*/bin"):
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _setup_logging() -> None:
    """Configure logging. Use file output when stderr is unavailable (pythonw.exe)."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%H:%M:%S"

    handlers: list[logging.Handler] = []

    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())

    # Always write to log file for debugging
    log_path = Path(__file__).resolve().parent.parent.parent / "vox.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


def _pid_path() -> Path:
    """Return the path to the PID file (project root / vox.pid)."""
    return Path(__file__).resolve().parent.parent.parent / "vox.pid"


def _kill_existing_instances() -> None:
    """Kill any existing Vox processes before starting."""
    logger = logging.getLogger(__name__)
    my_pid = os.getpid()
    my_ppid = os.getppid()
    # On Windows venv, python.exe is a trampoline launcher that spawns the
    # real interpreter as a child.  We must exclude both our own PID and the
    # launcher (parent) PID to avoid killing ourselves.
    my_pids = {my_pid, my_ppid}
    pid_file = _pid_path()

    # 1. PID ファイルがあればそのプロセスを停止
    if pid_file.exists():
        for line in pid_file.read_text().strip().splitlines():
            try:
                pid = int(line.strip())
                if pid not in my_pids:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True, timeout=5,
                    )
            except (ValueError, subprocess.TimeoutExpired, OSError):
                pass
        pid_file.unlink(missing_ok=True)

    # 2. コマンドラインで vox プロセスを検出して停止（PID ファイルが欠損しても動作）
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process"
                " -Filter \"Name='pythonw.exe' or Name='python.exe'\" |"
                " Where-Object { $_.CommandLine -match '-m\\s+vox'"
                f" -and $_.ProcessId -ne {my_pid}"
                f" -and $_.ProcessId -ne {my_ppid} }} |"
                " Select-Object -ExpandProperty ProcessId",
            ],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            try:
                pid = int(line.strip())
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5,
                )
                logger.info("Killed existing Vox process (PID %d)", pid)
            except (ValueError, subprocess.TimeoutExpired):
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass


def main() -> None:
    _register_cuda_dll_dirs()
    _setup_logging()
    _kill_existing_instances()

    config_path = Path("config.yaml")
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])

    config = load_config(config_path)
    app = VoxApp(config)

    # Write PID file so stop.bat can find us.
    # On Windows, venv pythonw.exe is a launcher trampoline that spawns the
    # real Python interpreter as a child process.  Record both the launcher
    # (parent) PID and the real (self) PID so stop.bat can kill the full tree.
    pid_file = _pid_path()
    pid_file.write_text(f"{os.getppid()}\n{os.getpid()}\n", encoding="utf-8")

    # Use a list so _cleanup can reference icon before it's assigned
    icon_ref: list[object] = []

    def _cleanup() -> None:
        app.stop()
        pid_file.unlink(missing_ok=True)
        if icon_ref:
            icon_ref[0].stop()  # type: ignore[union-attr]

    icon = create_tray_icon(on_quit=_cleanup)
    icon_ref.append(icon)

    def shutdown(signum: int, frame: object) -> None:
        _cleanup()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        app.start()
        # pystray.Icon.run() blocks the main thread (required by pystray on Windows)
        icon.run()
    finally:
        app.stop()
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
