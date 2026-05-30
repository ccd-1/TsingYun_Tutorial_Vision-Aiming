"""ctypes bindings for the C++ Kalman tracker (task3)."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _library_path() -> Path:
    env_dir = os.environ.get("TSINGYUN_HW_BUILD_DIR")

    if sys.platform == "darwin":
        build_dir = Path(env_dir) if env_dir else REPO_ROOT / "build" / "hw"
        return build_dir / "tasks" / "task3-tracker" / "libhw_task3_tracker_shared.dylib"

    if sys.platform == "win32":
        build_dir = Path(env_dir) if env_dir else REPO_ROOT / "build" / "hw-ninja"
        # Try Release directory first (MSVC), then root (Ninja)
        candidate = build_dir / "tasks" / "task3-tracker" / "Release" / "hw_task3_tracker_shared.dll"
        if candidate.exists():
            return candidate
        return build_dir / "tasks" / "task3-tracker" / "hw_task3_tracker_shared.dll"

    build_dir = Path(env_dir) if env_dir else REPO_ROOT / "build" / "hw"
    return build_dir / "tasks" / "task3-tracker" / "libhw_task3_tracker_shared.so"


def _resolve_library_path() -> Path:
    candidate = _library_path()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find the task3 shared library. Build the C++ targets first.\n"
        f"Expected:\n{candidate}"
    )


_LIB_PATH = _resolve_library_path()

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    dll_dirs = {_LIB_PATH.parent}
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for entry in path_entries:
        if entry:
            dll_dirs.add(Path(entry))
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))

_lib = ctypes.CDLL(str(_LIB_PATH))

# --- function signatures ---
_lib.tracker_create.restype = ctypes.c_void_p
_lib.tracker_create_with_params.argtypes = [ctypes.c_double, ctypes.c_double]
_lib.tracker_create_with_params.restype = ctypes.c_void_p
_lib.tracker_destroy.argtypes = [ctypes.c_void_p]
_lib.tracker_is_tracking.argtypes = [ctypes.c_void_p]
_lib.tracker_is_tracking.restype = ctypes.c_int
_lib.tracker_reset.argtypes = [ctypes.c_void_p]
_lib.tracker_get_position.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
_lib.tracker_update.argtypes = [
    ctypes.c_void_p,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
_lib.tracker_predict.argtypes = [
    ctypes.c_void_p, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
_lib.tracker_last_error.restype = ctypes.c_char_p


def _check_tracker_error() -> None:
    msg = _lib.tracker_last_error()
    if msg is not None:
        error_str = msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else str(msg)
        if "NotImplementedError" in error_str:
            raise NotImplementedError(error_str)
        raise RuntimeError(error_str)


class KalmanTracker:
    """Python wrapper around the C++ KalmanTracker from task3."""

    def __init__(self, process_noise: float = 0.05, measurement_noise: float = 10.0) -> None:
        self._ptr = _lib.tracker_create_with_params(process_noise, measurement_noise)

    def __del__(self) -> None:
        if hasattr(self, '_ptr') and self._ptr is not None:
            _lib.tracker_destroy(self._ptr)

    @property
    def is_tracking(self) -> bool:
        result = bool(_lib.tracker_is_tracking(self._ptr))
        return result

    def reset(self) -> None:
        _lib.tracker_reset(self._ptr)

    def get_position(self) -> tuple[float, float, float]:
        ox, oy, oz = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        _lib.tracker_get_position(self._ptr, ctypes.byref(ox), ctypes.byref(oy), ctypes.byref(oz))
        _check_tracker_error()
        return (ox.value, oy.value, oz.value)

    def update(self, x: float, y: float, z: float, dt: float) -> tuple[float, float, float]:
        ox, oy, oz = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        _lib.tracker_update(self._ptr, x, y, z, dt,
                            ctypes.byref(ox), ctypes.byref(oy), ctypes.byref(oz))
        _check_tracker_error()
        return (ox.value, oy.value, oz.value)

    def predict(self, dt: float) -> tuple[float, float, float]:
        ox, oy, oz = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        _lib.tracker_predict(self._ptr, dt, ctypes.byref(ox), ctypes.byref(oy), ctypes.byref(oz))
        _check_tracker_error()
        return (ox.value, oy.value, oz.value)
