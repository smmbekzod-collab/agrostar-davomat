"""Offline, no-credential tests for optional model page-cache advice.

No Railway mutation, AI inference, network, biometric data, or filesystem deletes.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
from unittest import TestCase, main, mock


def load_worker():
    path = Path(__file__).with_name("face_worker_module.py")
    spec = importlib.util.spec_from_file_location("test_only_face_worker_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IdleCacheTests(TestCase):
    def setUp(self):
        self.worker = load_worker()

    def test_disabled_by_default(self):
        with mock.patch.object(self.worker, "_MODEL_CACHE_RECLAIM", False), \
             mock.patch("os.posix_fadvise") as advice:
            self.worker._advise_idle_model_cache()
            advice.assert_not_called()

    def test_only_large_regular_model_weights_are_advised(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / ".deepface" / "weights"
            base.mkdir(parents=True)
            model = base / "facenet512_weights.h5"
            with model.open("wb") as f:
                f.write(b"SAFE")
                f.truncate(5 * 1024 * 1024)
            (base / "README.txt").write_text("never touch other files")
            (base / "small.h5").write_bytes(b"tiny")
            (base / "weight_link.h5").symlink_to(model)

            events = []
            def record(fd, offset, length, advice):
                events.append((os.fstat(fd).st_size, offset, length, advice))

            with mock.patch.object(self.worker, "_MODEL_CACHE_RECLAIM", True), \
                 mock.patch.object(Path, "home", return_value=Path(tmp)), \
                 mock.patch("os.posix_fadvise", side_effect=record):
                self.worker._advise_idle_model_cache()

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0], (5*1024*1024, 0, 0, os.POSIX_FADV_DONTNEED))
            self.assertEqual(model.stat().st_size, 5*1024*1024)
            with model.open("rb") as f:
                self.assertEqual(f.read(4), b"SAFE")
            self.assertEqual((base / "README.txt").read_text(), "never touch other files")

    def test_kernel_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / ".deepface" / "weights"
            d.mkdir(parents=True)
            with (d / "test.h5").open("wb") as f:
                f.truncate(5 * 1024 * 1024)
            with mock.patch.object(self.worker, "_MODEL_CACHE_RECLAIM", True), \
                 mock.patch.object(Path, "home", return_value=Path(tmp)), \
                 mock.patch("os.posix_fadvise", side_effect=OSError("not supported")):
                self.worker._advise_idle_model_cache()

    def test_idle_retirement_waits_for_process_exit(self):
        class Pool:
            shutdown_calls = []
            def shutdown(self, *, wait, cancel_futures):
                self.shutdown_calls.append((wait, cancel_futures))
        pool = Pool()
        with mock.patch.object(self.worker, "_POOL", pool), \
             mock.patch.object(self.worker, "_ACTIVE", 0), \
             mock.patch.object(self.worker, "_MODEL_CACHE_RECLAIM", True), \
             mock.patch.object(self.worker, "_advise_idle_model_cache") as advice:
            self.worker._retire(pool)
            self.assertIsNone(self.worker._POOL)
            advice.assert_called_once()
        self.assertEqual(pool.shutdown_calls, [(True, False)])

    def test_stale_timer_does_not_affect_new_worker(self):
        class Pool:
            def shutdown(self, **kwargs):
                raise AssertionError("must not shut down a different worker")
        current, stale = Pool(), Pool()
        with mock.patch.object(self.worker, "_POOL", current), \
             mock.patch.object(self.worker, "_ACTIVE", 0), \
             mock.patch.object(self.worker, "_advise_idle_model_cache") as advice:
            self.worker._retire(stale)
            advice.assert_not_called()

    def test_no_eviction_if_new_request_begins(self):
        class Pool:
            def shutdown(self, *, wait, cancel_futures):
                worker._POOL = object()
                worker._ACTIVE = 1
        worker = self.worker
        p = Pool()
        with mock.patch.object(worker, "_POOL", p), \
             mock.patch.object(worker, "_ACTIVE", 0), \
             mock.patch.object(worker, "_MODEL_CACHE_RECLAIM", True), \
             mock.patch.object(worker, "_advise_idle_model_cache") as advice:
            worker._retire(p)
            advice.assert_not_called()


if __name__ == "__main__":
    main(verbosity=2)
