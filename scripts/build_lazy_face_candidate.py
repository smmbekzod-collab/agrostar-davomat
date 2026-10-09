"""Safely build an isolated lazy-DeepFace candidate from the deployed v4.2 ZIP.

No production changes; preserve all other source, tests, and biometric checks.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import ast
import sys

ORIGINAL = Path("agrostar_davomat_4_2_PUSH_REMINDER_CALENDAR_FULL.zip")
CANDIDATE = Path("davomat_v4_2_lazy_deepface_candidate.zip")

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
        assert all(final.read(i.filename) == original.read(i.filename) for i in original.infolist() if i.filename != target)
print("PASS: only face.py changed; DeepFace import deferred; biometric calls intact")
print("CANDIDATE", CANDIDATE)
