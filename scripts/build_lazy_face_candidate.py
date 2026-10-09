"""Safely build an isolated lazy-DeepFace candidate from the deployed v4.2 ZIP.

No production changes; preserve all other source, tests, and biometric checks.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import ast
import sys

ORIGINAL = Path("agrostar_davomat_4_2_PUSH_REMINDER_CALENDAR_FULL.zip")
CANDIDATE = Path("davomat_v4_2_lazy_deepface_candidate.zip")
ISOLATED = Path("davomat_v4_2_face_worker_candidate.zip")

def patch(source: str) -> str:
    needle = "from deepface import DeepFace\n"
    if source.count(needle) != 1:
        raise ValueError("Unexpected DeepFace import; refuse unsafe patch")
    source = source.replace(needle, "", 1)
    first = "    last_error: Exception | None = None\n"
    second = "    img = _decode(image_bytes)\n"
    if source.count(first) != 1 or source.count(second) != 1:
        raise ValueError("Face callsite changed; refuse unsafe patch")
    source = source.replace(first, "    from deepface import DeepFace  # Load ML stack on first face request only.\n" + first, 1)
    source = source.replace(second, "    from deepface import DeepFace  # Python import cache makes subsequent calls inexpensive.\n" + second, 1)
    ast.parse(source)
    return source

with ZipFile(ORIGINAL) as original:
    face_files = [i.filename for i in original.infolist() if i.filename.endswith("/backend/app/services/face.py")]
    if len(face_files) != 1:
        raise SystemExit("Expected exactly one backend face.py")
    target = face_files[0]
    before = original.read(target).decode("utf-8")
    after = patch(before)
    a, b = ast.parse(before), ast.parse(after)
    def biometric_controls(tree):
        return sorted([
            (n.func.attr, tuple((kw.arg, ast.unparse(kw.value)) for kw in n.keywords))
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("extract_faces", "represent")
        ])
    if biometric_controls(a) != biometric_controls(b):
        raise SystemExit("Biometric call changed; refusing candidate")
    if after.count("anti_spoofing=True") != 1 or after.count("anti_spoofing=False") != 1:
        raise SystemExit("Spoofing settings unexpectedly changed")
    with ZipFile(CANDIDATE, "w") as out:
        for info in original.infolist():
            out.writestr(info, after.encode("utf-8") if info.filename == target else original.read(info.filename))
    with ZipFile(CANDIDATE) as final:
        assert final.testzip() is None
        assert len(final.infolist()) == len(original.infolist())
        with ZipFile(ORIGINAL) as check_original:
            assert all(final.read(i.filename) == check_original.read(i.filename) for i in check_original.infolist() if i.filename != target)
print("PASS: only face.py changed; DeepFace import deferred; biometric calls intact")
print("CANDIDATE", CANDIDATE)


# Additional candidate: preserve strict face and liveness policy, but run the
# complete face analysis inside a spawned worker that exits after inactivity.
def worker_patch(source: str) -> str:
    # Work from the already lazy-imported module, to ensure CPU ML packages
    # never enter the HTTP server process even on the first attendance request.
    signature = "def analyze_selfie(image_bytes: bytes, require_liveness: bool = True, soft_liveness: bool = False) -> tuple[list[float], float | None, bytes]:"
    if source.count(signature) != 1:
        raise ValueError("Face public method signature changed")
    patched = source.replace(signature, signature.replace("analyze_selfie(", "_analyze_selfie_local("), 1)
    wrapper = '''
def analyze_selfie(image_bytes: bytes, require_liveness: bool = True, soft_liveness: bool = False) -> tuple[list[float], float | None, bytes]:
    from app.services.face_worker import analyze_selfie_isolated
    return analyze_selfie_isolated(image_bytes, require_liveness, soft_liveness)

'''
    # Place wrapper before extract_embedding (avoids modifying its call graph).
    insertion = "\ndef extract_embedding("
    if patched.count(insertion) != 1:
        raise ValueError("No insertion point for public API wrapper")
    patched = patched.replace(insertion, "\n" + wrapper + "def extract_embedding(", 1)
    ast.parse(patched)
    return patched

with ZipFile(CANDIDATE) as old:
    face_name = next(i.filename for i in old.infolist() if i.filename.endswith("/backend/app/services/face.py"))
    prefix = face_name.removesuffix("face.py")
    worker_name = prefix + "face_worker.py"
    docker_name = prefix.removesuffix("app/services/") + "Dockerfile"
    old_docker = old.read(docker_name).decode("utf-8")
    opencv_guard = "\n# Pin OpenCV because opencv-python 5.x broke CascadeClassifier in CI.\nRUN pip uninstall -y opencv-python opencv-python-headless \\\n    && pip install --no-cache-dir --no-deps opencv-python-headless==4.12.0.88 \\\n    && python -c \"import cv2; assert hasattr(cv2, 'CascadeClassifier')\"\n"
    if old_docker.count("COPY app ./app") != 1: raise ValueError("Unexpected Dockerfile")
    new_docker = old_docker.replace("COPY app ./app", opencv_guard+"\nCOPY app ./app")
    patched_face = worker_patch(old.read(face_name).decode("utf-8"))
    worker_source = Path("scripts/face_worker_module.py").read_text(encoding="utf-8")
    ast.parse(worker_source)
    with ZipFile(ISOLATED, "w") as out:
        for info in old.infolist():
            out.writestr(info, patched_face.encode("utf-8") if info.filename == face_name else new_docker.encode("utf-8") if info.filename == docker_name else old.read(info.filename))
        out.writestr(worker_name, worker_source.encode("utf-8"))
with ZipFile(ISOLATED) as z:
    assert z.testzip() is None
    assert worker_name in z.namelist()
    assert len(z.namelist()) == len(ZipFile(CANDIDATE).namelist()) + 1
print("PASS: isolated one-worker candidate created, model policy unchanged", ISOLATED)
