"""Build an UNDEPLOYED face worker candidate with RetinaFace crop reuse.

Preserves the production-derived worker, liveness checks, detector fallback,
all routes, auth code, and user-facing payloads. Changes only face.py.
"""
from __future__ import annotations

import ast
from pathlib import Path
from zipfile import ZipFile

SOURCE = Path("davomat_v4_2_face_worker_candidate.zip")
OUTPUT = Path("davomat_v4_2_retinaface_reuse_NOT_DEPLOYED.zip")


def exactly_once(s: str, old: str, new: str) -> str:
    if s.count(old) != 1:
        raise RuntimeError(f"Patch refused: expected exactly one match ({old[:75]!r})")
    return s.replace(old, new, 1)


with ZipFile(SOURCE) as archive:
    names = [info.filename for info in archive.infolist()
             if info.filename.endswith("/backend/app/services/face.py")]
    if len(names) != 1:
        raise RuntimeError("Cannot identify original face.py")
    name = names[0]
    before = archive.read(name).decode("utf-8")

after = before
after = exactly_once(after,
    "def _find_single_face_image(img: np.ndarray) -> tuple[np.ndarray, str]:",
    "def _find_single_face_image(img: np.ndarray) -> tuple[np.ndarray, str, np.ndarray]:")
after = exactly_once(after,
    "                    return candidate, detector",
    "                    return candidate, detector, np.ascontiguousarray(faces[0]['face'][:, :, ::-1])")
after = exactly_once(after,
    "    face_img, detector = _find_single_face_image(img)",
    "    face_img, detector, embedding_bgr = _find_single_face_image(img)")
after = exactly_once(after,
    'reps = DeepFace.represent(\n            img_path=face_img,\n'
    '            model_name=settings.face_model,\n            detector_backend=detector,',
    'reps = DeepFace.represent(\n            img_path=embedding_bgr,\n'
    '            model_name=settings.face_model,\n            detector_backend="skip",')

base = ast.parse(before)
candidate = ast.parse(after)

def calls(tree, fn_name: str):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == fn_name and
            isinstance(node.func.value, ast.Name) and node.func.value.id == "DeepFace"]

def keywords(nodes):
    return sorted([[(kw.arg, ast.unparse(kw.value)) for kw in n.keywords] for n in nodes],
                  key=repr)

if keywords(calls(base, "extract_faces")) != keywords(calls(candidate, "extract_faces")):
    raise RuntimeError("Liveness/detection API calls changed; refusing audit candidate")
rep_base = calls(base, "represent")
rep_new = calls(candidate, "represent")
if len(rep_base) != len(rep_new) or len(rep_new) != 1:
    raise RuntimeError("Unexpected represent call structure")
old_kw = {x.arg: ast.unparse(x.value) for x in rep_base[0].keywords}
new_kw = {x.arg: ast.unparse(x.value) for x in rep_new[0].keywords}
if {k:v for k,v in old_kw.items() if k not in ("img_path","detector_backend")} != {
    k:v for k,v in new_kw.items() if k not in ("img_path","detector_backend")
}:
    raise RuntimeError("Recognition model or parameters changed")
if old_kw["detector_backend"] != "detector" or new_kw["detector_backend"] != "'skip'":
    raise RuntimeError("Unexpected detector settings")
if new_kw["img_path"] != "embedding_bgr":
    raise RuntimeError("Candidate must only embed earlier aligned BGR crop")
if before.count("anti_spoofing=True") != 1 or after.count("anti_spoofing=True") != 1:
    raise RuntimeError("Strict anti-spoofing contract lost")

# Ensure no external callsites use the old helper's return arity.
helper_calls = [x for x in ast.walk(base)
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                and x.func.id == "_find_single_face_image"]
if len(helper_calls) != 1:
    raise RuntimeError("Unexpected helper callsites; refuse potentially breaking change")

with ZipFile(SOURCE) as basezip, ZipFile(OUTPUT, "w") as outzip:
    for info in basezip.infolist():
        outzip.writestr(info, after.encode("utf-8") if info.filename == name
                        else basezip.read(info.filename))
with ZipFile(SOURCE) as original, ZipFile(OUTPUT) as changed:
    assert changed.testzip() is None
    assert set(changed.namelist()) == set(original.namelist())
    assert all(original.read(i.filename) == changed.read(i.filename)
               for i in original.infolist() if i.filename != name)
print("PASS: original face worker preserved; anti-spoofing untouched; "
      "only redundant embedding RetinaFace detection bypassed in audit ZIP")
print("AUDIT_ONLY", OUTPUT)
