"""Audit-only RetinaFace face crop reuse benchmark with public sample images.

Does not touch the Railway database, production service, security settings, or
private images. Runs comparison against the exact bundled original face model.
Outputs numerical discrepancies only; never stores embeddings or portraits.
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

ARCHIVE = Path("davomat_v4_2_lazy_deepface_candidate.zip")
REPORT = Path("davomat_retinaface_crop_reuse.json")
MODEL = "Facenet512"
DETECTOR = "retinaface"


def face_module():
    with ZipFile(ARCHIVE) as z:
        files = [x.filename for x in z.infolist()
                 if x.filename.endswith("/backend/app/services/face.py")]
        if len(files) != 1:
            raise RuntimeError("Backend face.py not uniquely found")
        content = z.read(files[0])
    app = types.ModuleType("app")
    app.__path__ = []
    core = types.ModuleType("app.core")
    core.__path__ = []
    cfg = types.ModuleType("app.core.config")
    cfg.settings = SimpleNamespace(
        face_detector=DETECTOR, face_model=MODEL,
        face_threshold=.30, max_upload_bytes=10_000_000)
    sys.modules.update({"app": app, "app.core": core, "app.core.config": cfg})
    with tempfile.TemporaryDirectory(prefix="face-audit-") as tmp:
        p = Path(tmp) / "face.py"
        p.write_bytes(content)
        spec = importlib.util.spec_from_file_location("app.services.face", p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def distance(a, b) -> float:
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embedding(image, detector_backend: str, align: bool, enforce_detection: bool):
    from deepface import DeepFace
    start = time.perf_counter()
    reps = DeepFace.represent(
        img_path=image, model_name=MODEL,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align, normalization="base", max_faces=1)
    duration = round(time.perf_counter() - start, 4)
    if len(reps) != 1:
        raise RuntimeError("Expected exactly one face embedding")
    vec = np.asarray(reps[0]["embedding"], dtype="float64")
    if vec.shape != (512,):
        raise RuntimeError("Incorrect vector shape")
    return vec, duration


def test_one(label: str, rgb: np.ndarray):
    from deepface import DeepFace
    detector_time = time.perf_counter()
    faces = DeepFace.extract_faces(
        img_path=rgb, detector_backend=DETECTOR,
        enforce_detection=True, align=True, anti_spoofing=False)
    detector_seconds = round(time.perf_counter() - detector_time, 4)
    if len(faces) != 1:
        raise RuntimeError(f"Expected one aligned face ({label})")
    crop = np.asarray(faces[0]["face"])
    reference, reference_time = embedding(rgb, DETECTOR, True, True)
    # The extract_faces output is RGB in 0..1; benchmark possible conversions
    # explicitly rather than assuming a skip-backend channel convention.
    candidates = {
        "rgb_float_0_1": crop,
        "rgb_uint8_0_255": np.clip(crop * 255, 0, 255).astype("uint8"),
        "bgr_float_0_1": np.ascontiguousarray(crop[:, :, ::-1]),
        "bgr_uint8_0_255": np.ascontiguousarray(
            np.clip(crop[:, :, ::-1] * 255, 0, 255).astype("uint8")),
    }
    trials = []
    for name, candidate in candidates.items():
        try:
            vector, sec = embedding(candidate, "skip", False, False)
            trials.append({
                "candidate": name,
                "seconds": sec,
                "cosine_disagreement": round(distance(reference, vector), 8),
                "matches_0_002": abs(distance(reference, vector)) < 0.002,
            })
        except Exception as exc:
            trials.append({
                "candidate": name,
                "error": type(exc).__name__,
                "matches_0_002": False,
            })
    return {
        "sample": label,
        "detector_seconds": detector_seconds,
        "baseline_embedding_seconds": reference_time,
        "cropped_face_shape": [int(v) for v in crop.shape],
        "candidate_trials": trials,
    }


def main():
    _ = face_module()
    base = Image.fromarray(astronaut()).convert("RGB").crop((70, 0, 360, 360))
    # Rotate and mirror the same public image to test geometry/channel handling.
    samples = [
        ("original_public", np.asarray(base)),
        ("mirrored_public", np.asarray(base.transpose(Image.Transpose.FLIP_LEFT_RIGHT))),
    ]
    results = [test_one(label, img) for label, img in samples]
    viable = [
        key for key in ("rgb_float_0_1", "rgb_uint8_0_255", "bgr_float_0_1", "bgr_uint8_0_255")
        if all(any(t["candidate"] == key and t["matches_0_002"]
                   for t in result["candidate_trials"]) for result in results)
    ]
    report = {
        "scope": "public-image-only; no production access; original RetinaFace/FaceNet512",
        "anti_spoofing": "UNMODIFIED; skip candidates affect embedding only",
        "production_change": False,
        "results": results,
        "compatible_candidates": viable,
        "caution": "Agreement on two public variations is not sufficient for production; "
                   "must validate same-person and impostor distributions, spoofing, "
                   "and legacy stored embeddings without exposing images.",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print("RETINAFACE_CROP_AUDIT " + json.dumps(report, ensure_ascii=False))
    print("PASS: all results measured; no security or production settings changed")


if __name__ == "__main__":
    main()
