"""CPU-heavy face inference isolated from the small FastAPI process.

A single spawned worker handles requests; it goes away after an idle window.
No biometric vectors or photos are stored outside the normal request lifetime.
Face matching, spoofing policies and return payloads remain in face.py.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import os
import threading

_IDLE_SECONDS = max(30, min(600, int(os.getenv("FACE_WORKER_IDLE_SECONDS", "90"))))
_TIMEOUT_SECONDS = max(30, min(600, int(os.getenv("FACE_WORKER_TIMEOUT_SECONDS", "240"))))
_LOCK = threading.Lock()
_POOL = None
_IDLE_TIMER = None
_ACTIVE = 0


def _infer(image_bytes: bytes, require_liveness: bool, soft_liveness: bool):
    # Import only inside the process. No DeepFace/TensorFlow memory in the API.
    from app.services.face import _analyze_selfie_local
    return _analyze_selfie_local(image_bytes, require_liveness, soft_liveness)


def _retire(expected_pool):
    global _POOL, _IDLE_TIMER
    with _LOCK:
        if _POOL is not expected_pool or _ACTIVE:
            return
        _POOL = None
        _IDLE_TIMER = None
    expected_pool.shutdown(wait=False, cancel_futures=False)


def _close_on_exit():
    global _POOL, _IDLE_TIMER
    with _LOCK:
        pool, _POOL = _POOL, None
        timer, _IDLE_TIMER = _IDLE_TIMER, None
    if timer:
        timer.cancel()
    if pool:
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(_close_on_exit)


def analyze_selfie_isolated(image_bytes: bytes, require_liveness: bool, soft_liveness: bool):
    global _POOL, _IDLE_TIMER, _ACTIVE
    from app.services.face import FaceError
    with _LOCK:
        if _IDLE_TIMER is not None:
            _IDLE_TIMER.cancel()
            _IDLE_TIMER = None
        if _POOL is None:
            # "spawn" is intentional; "fork" from an ASGI server is not safe
            # when threads, TensorFlow and DB connections already exist.
            _POOL = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        pool = _POOL
        _ACTIVE += 1
    try:
        return pool.submit(_infer, image_bytes, require_liveness, soft_liveness).result(
            timeout=_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        raise FaceError("Yuzni tekshirish vaqti tugadi, iltimos qayta urinib ko‘ring") from exc
    except BrokenProcessPool as exc:
        # Give the next request a clean worker instead of reusing a broken pool.
        with _LOCK:
            if _POOL is pool:
                _POOL = None
        raise FaceError("Yuzni tekshirish xizmati vaqtincha mavjud emas") from exc
    finally:
        with _LOCK:
            _ACTIVE -= 1
            if _ACTIVE == 0 and _POOL is pool:
                timer = threading.Timer(_IDLE_SECONDS, _retire, args=(pool,))
                timer.daemon = True
                _IDLE_TIMER = timer
                timer.start()


def worker_running() -> bool:
    """Diagnostic helper; not a public endpoint."""
    with _LOCK:
        return _POOL is not None
