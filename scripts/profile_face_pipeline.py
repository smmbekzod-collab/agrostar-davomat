"""Read-only face-stage profiler using a public image. No production connections."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import types
from types import SimpleNamespace
from zipfile import ZipFile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

CANDIDATE = Path("davomat_v4_2_lazy_deepface_candidate.zip")
OUTPUT = Path("davomat_face_stage_profile.json")


def rss_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as status:
        for row in status:
            if row.startswith("VmRSS:"):
                return round(int(row.split()[1]) / 1024, 2)
    raise RuntimeError("Linux proc status unavailable")


def load_face():
    with ZipFile(CANDIDATE) as archive:
        matches = [entry.filename for entry in archive.infolist()
                   if entry.filename.endswith("/backend/app/services/face.py")]
        if len(matches) != 1:
            raise RuntimeError("Exactly one audited backend face.py is required")
        source = archive.read(matches[0])

    app = types.ModuleType("app")
    app.__path__ = []
    core = types.ModuleType("app.core")
    core.__path__ = []
    config = types.ModuleType("app.core.config")
    config.settings = SimpleNamespace(
        face_detector="opencv", face_model="Facenet512",
        face_threshold=0.3, max_upload_bytes=10_000_000)
    sys.modules.update({
        "app": app, "app.core": core, "app.core.config": config
    })
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "face.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("app.services.face", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def main():
    started = time.perf_counter()
    face = load_face()
    api_import = round(time.perf_counter() - started, 4)
    assert "deepface" not in sys.modules, "Expected lazy DeepFace import"

    started = time.perf_counter()
    from deepface import DeepFace
    ml_import = round(time.perf_counter() - started, 4)

    from PIL import Image
    from skimage.data import astronaut

    image = Image.fromarray(astronaut()).convert("RGB").crop((70, 0, 360, 360))
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=92)
    image_bytes = data.getvalue()

    events = []
    original_extract = DeepFace.extract_faces
    original_represent = DeepFace.represent
    original_decode = face._decode

    def record(stage, fn, *args, **kwargs):
        start = time.perf_counter()
        succeeded = False
        try:
            result = fn(*args, **kwargs)
            succeeded = True
            return result
        finally:
            events.append({
                "stage": stage,
                "detector": kwargs.get("detector_backend"),
                "seconds": round(time.perf_counter() - start, 4),
                "completed": succeeded,
                "rss_mb": rss_mb(),
            })

    def extract(*args, **kwargs):
        stage = "liveness_detection" if kwargs.get("anti_spoofing") else "face_detection"
        return record(stage, original_extract, *args, **kwargs)

    def represent(*args, **kwargs):
        return record("embedding", original_represent, *args, **kwargs)

    def decode(*args, **kwargs):
        return record("decode", original_decode, *args, **kwargs)

    DeepFace.extract_faces = staticmethod(extract)
    DeepFace.represent = staticmethod(represent)
    face._decode = decode

    results = []
    for scenario, strict in (
        ("first_face", False), ("second_face", False), ("strict_liveness", True)
    ):
        events.clear()
        start = time.perf_counter()
        try:
            embedding, _, _ = face.analyze_selfie(image_bytes, require_liveness=strict)
            outcome = "accepted"
            embedding_dimension = len(embedding)
        except Exception as exc:
            outcome = type(exc).__name__
            embedding_dimension = None

        stages = list(events)
        results.append({
            "scenario": scenario,
            "outcome": outcome,
            "embedding_dimension": embedding_dimension,
            "total_seconds": round(time.perf_counter() - start, 4),
            "stages": stages,
            "rss_after_mb": rss_mb(),
        })

    assert all(r["outcome"] == "accepted" for r in results[:2])
    assert results[0]["embedding_dimension"] == results[1]["embedding_dimension"]
    assert any(s["stage"] == "embedding" for s in results[0]["stages"])

    payload = {
        "source": "public skimage astronaut only",
        "production_affected": False,
        "api_import_seconds": api_import,
        "ml_import_seconds": ml_import,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print("FACE_STAGE_PROFILE " + json.dumps(payload, ensure_ascii=False))
    print("PASS: Isolated public-face profile generated without production access")


if __name__ == "__main__":
    main()
