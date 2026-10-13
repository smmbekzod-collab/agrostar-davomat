"""End-to-end RetinaFace public-image equivalence test (NO DEPLOYMENT).

Compares worker's real baseline local function against only-face.py patched version.
Strict anti-spoofing and existing 512-D vectors must match. No private photos,
database, API endpoints, or production environment are accessed.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
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

import numpy as np
from PIL import Image
from skimage.data import astronaut

ORIGINAL = Path("davomat_v4_2_face_worker_candidate.zip")
CANDIDATE = Path("davomat_v4_2_retinaface_reuse_NOT_DEPLOYED.zip")
OUTPUT = Path("davomat_retinaface_worker_equivalence.json")


def load_face(zip_path: Path, module_name: str):
    with ZipFile(zip_path) as zipped:
        members = [x.filename for x in zipped.infolist()
                   if x.filename.endswith("/backend/app/services/face.py")]
        if len(members) != 1:
            raise RuntimeError("Face module must exist exactly once")
        source = zipped.read(members[0])

    with tempfile.TemporaryDirectory() as folder:
        file = Path(folder) / (module_name + ".py")
        file.write_bytes(source)
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert hasattr(module, "_analyze_selfie_local")
    return module


def cosine_disagreement(a, b) -> float:
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    if a.shape != (512,) or b.shape != (512,):
        raise AssertionError("Legacy 512-D FaceNet512 embeddings changed")
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sample_bytes(image: Image.Image) -> bytes:
    io_file = io.BytesIO()
    image.save(io_file, format="JPEG", quality=92)
    return io_file.getvalue()


def evaluate(module, photo_bytes: bytes, strict: bool):
    # The counting wrapper is OBSERVATION ONLY. It calls the original detector
    # with unchanged arguments, and is restored even after an error.
    from deepface.modules import detection
    original = detection.extract_faces
    detector_calls = []
    def tracked(*args, **kwargs):
        detector_calls.append({
            "backend": kwargs.get("detector_backend", "unspecified"),
            "anti_spoofing": bool(kwargs.get("anti_spoofing", False)),
        })
        return original(*args, **kwargs)
    detection.extract_faces = tracked
    start = time.perf_counter()
    try:
        vec, score, jpeg = module._analyze_selfie_local(
            photo_bytes, require_liveness=strict, soft_liveness=False)
        return {
            "outcome": "accepted",
            "vector": vec,
            "liveness": score,
            "audit_bytes": jpeg,
            "seconds": round(time.perf_counter() - start, 4),
            "detector_calls": detector_calls,
        }
    except module.FaceError as exc:
        msg = str(exc).lower()
        category = "liveness" if "jonli" in msg or "spoof" in msg else "face_error"
        return {
            "outcome": category,
            "seconds": round(time.perf_counter() - start, 4),
            "detector_calls": detector_calls,
        }
    finally:
        detection.extract_faces = original


def main():
    app = types.ModuleType("app")
    app.__path__ = []
    core = types.ModuleType("app.core")
    core.__path__ = []
    cfg = types.ModuleType("app.core.config")
    cfg.settings = SimpleNamespace(
        face_detector="retinaface", face_model="Facenet512",
        face_threshold=0.30, max_upload_bytes=10_000_000)
    sys.modules.update({"app": app, "app.core": core, "app.core.config": cfg})

    original = load_face(ORIGINAL, "face_worker_original")
    optimized = load_face(CANDIDATE, "face_worker_reuse")
    base = Image.fromarray(astronaut()).convert("RGB").crop((70, 0, 360, 360))
    examples = [
        ("original_public", sample_bytes(base)),
        ("mirrored_public", sample_bytes(
            base.transpose(Image.Transpose.FLIP_LEFT_RIGHT))),
        ("rotated_public", sample_bytes(
            base.transpose(Image.Transpose.ROTATE_90))),
    ]

    # Preload both recognition and liveness models so paired timings are warm;
    # this is a functional comparison, not a production performance promise.
    evaluate(original, examples[0][1], strict=False)
    evaluate(original, examples[0][1], strict=True)
    rows = []
    accepts = 0
    for sample_name, payload in examples:
        for strict in (False, True):
            baseline = evaluate(original, payload, strict)
            candidate = evaluate(optimized, payload, strict)
            if baseline["outcome"] != candidate["outcome"]:
                raise AssertionError(f"Rejected/accepted mismatch: {sample_name}, strict={strict}")
            b_liveness = sum(int(x["anti_spoofing"]) for x in baseline["detector_calls"])
            c_liveness = sum(int(x["anti_spoofing"]) for x in candidate["detector_calls"])
            if b_liveness != c_liveness:
                raise AssertionError("Anti-spoofing detector call was changed")
            if strict and b_liveness < 1:
                raise AssertionError("Strict anti-spoofing was bypassed")
            baseline_retina_calls = sum(x["backend"] == "retinaface"
                                        for x in baseline["detector_calls"])
            optimized_retina_calls = sum(x["backend"] == "retinaface"
                                         for x in candidate["detector_calls"])
            if baseline["outcome"] == "accepted":
                accepts += 1
                disagreement = cosine_disagreement(
                    baseline["vector"], candidate["vector"])
                if abs(disagreement) > 1e-6:
                    raise AssertionError(f"Legacy embeddings differ ({disagreement:.8f})")
                if baseline["audit_bytes"] != candidate["audit_bytes"]:
                    raise AssertionError("Saved audit selfie changed")
                a,b = baseline["liveness"],candidate["liveness"]
                if (a is None) != (b is None):
                    raise AssertionError("Liveness score availability changed")
                if a is not None and abs(float(a) - float(b)) > 1e-6:
                    raise AssertionError("Liveness score changed")
            else:
                disagreement = None
            removed = baseline_retina_calls - optimized_retina_calls
            if removed != (1 if baseline["outcome"] == "accepted" else 0):
                raise AssertionError("Detector pass count mismatch: baseline={}, candidate={}, removed={}, outcome={}".format(baseline["detector_calls"], candidate["detector_calls"], removed, baseline["outcome"]))
            rows.append({
                "sample": sample_name,
                "strict": strict,
                "outcome": candidate["outcome"],
                "cosine_disagreement": round(disagreement, 9) if disagreement is not None else None,
                "original_seconds": baseline["seconds"],
                "optimized_seconds": candidate["seconds"],
                "original_retina_calls": baseline_retina_calls,
                "optimized_retina_calls": optimized_retina_calls,
                "strict_liveness_calls_preserved": b_liveness == c_liveness,
            })

    if accepts < 2:
        raise AssertionError("Too few accepted public tests; audit inconclusive")
    report = {
        "scope": "original deployed-worker ZIP vs undeployed RetinaFace crop reuse",
        "config": "RetinaFace/FaceNet512; original anti-spoofing preserved",
        "production_modified": False,
        "public_samples": len(examples),
        "accepted_test_cases": accepts,
        "rows": rows,
        "warning": "Technical equivalence on one public person's variations is "
                   "not enough to approve production biometric policy change.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print("RETINAFACE_WORKER_EQUIVALENCE " + json.dumps(report, ensure_ascii=False))
    print("PASS: RetinaFace worker face recognition equivalent on public tests, "
          "strict anti-spoof policy intact, no production changes")


if __name__ == "__main__":
    main()
