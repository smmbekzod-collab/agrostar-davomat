"""Measure original and lazy DeepFace imports in isolated Python processes.

GitHub Actions only, no server connection or user/selfie data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile


BASE = Path("agrostar_davomat_4_2_PUSH_REMINDER_CALENDAR_FULL.zip")
CANDIDATE = Path("davomat_v4_2_lazy_deepface_candidate.zip")
RESULT = Path("davomat_face_memory_benchmark.json")


def source_from_zip(path: Path) -> str:
    with ZipFile(path) as archive:
        members = [x.filename for x in archive.infolist()
                   if x.filename.endswith("/backend/app/services/face.py")]
        if len(members) != 1:
            raise ValueError(f"Exactly one face.py required: {path}")
        return archive.read(members[0]).decode("utf-8")


CHILD = r'''
import importlib.util
import json
import os
import sys
import time
import types
from types import SimpleNamespace
def rss_mb():
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])/1024
    return 0.0
pkg = types.ModuleType("app")
pkg.__path__ = []
core = types.ModuleType("app.core")
core.__path__ = []
config = types.ModuleType("app.core.config")
config.settings = SimpleNamespace(
    face_detector="retinaface", face_model="Facenet512",
    face_threshold=0.3, max_upload_bytes=10000000)
sys.modules.update({"app":pkg, "app.core":core, "app.core.config":config})
before = rss_mb()
started = time.monotonic()
spec = importlib.util.spec_from_file_location("app.services.face",sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cold_sec = time.monotonic() - started
after = rss_mb()
distance = module.cosine_distance([1.0,0.0],[1.0,0.0])
assert abs(distance)<1e-6
assert module.verify_embedding([1.0,0.0],[1.0,0.0])[0] is True
after_math = rss_mb()
imported_ml = "deepface" in sys.modules
if sys.argv[2] == "lazy-warm":
    t = time.monotonic()
    from deepface import DeepFace
    warm_sec = time.monotonic()-t
else:
    warm_sec = None
print("BENCHMARK_RESULT "+json.dumps({
    "mode":sys.argv[2], "rss_before_mb":before,
    "rss_after_import_mb":after, "rss_after_math_mb":after_math,
    "cold_import_seconds":cold_sec,
    "deepface_loaded_at_cold_start":imported_ml,
    "deferred_import_seconds":warm_sec,
    "rss_after_deferred_import_mb":rss_mb(),
}),flush=True)
'''

def run(mode: str, script: Path) -> dict:
    env = os.environ.copy()
    env.update({"TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2",
                "TF_NUM_INTRAOP_THREADS": "2"})
    process = subprocess.Popen([sys.executable,"-c",CHILD,str(script),mode],
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
    max_rss_mb = 0.0
    start = time.monotonic()
    while process.poll() is None:
        try:
            for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    max_rss_mb = max(max_rss_mb, int(line.split()[1])/1024)
                    break
        except OSError:
            pass
        if time.monotonic()-start > 180:
            process.kill()
            break
        time.sleep(0.04)
    stdout, stderr = process.communicate(timeout=10)
    if process.returncode:
        raise RuntimeError(f"{mode} failed ({process.returncode}): {stderr[-3000:]}")
    matched = [x[len("BENCHMARK_RESULT "):] for x in stdout.splitlines()
               if x.startswith("BENCHMARK_RESULT ")]
    if len(matched)!=1:
        raise RuntimeError(f"Missing/duplicate JSON for {mode}: {stdout[-1200:]} {stderr[-500:]}")
    data = json.loads(matched[0])
    data["peak_sampled_rss_mb"] = round(max_rss_mb,2)
    data["elapsed_seconds"] = round(time.monotonic()-start,3)
    return data


def main():
    if not BASE.exists() or not CANDIDATE.exists():
        raise FileNotFoundError("Build isolated candidate first")
    with tempfile.TemporaryDirectory(prefix="davomat-bench-") as tmp:
        scripts = {}
        for name,archive in (("baseline",BASE),("lazy",CANDIDATE)):
            p=Path(tmp)/(name+"_face.py")
            p.write_text(source_from_zip(archive),encoding="utf-8")
            scripts[name]=p
        results = [
            run("baseline",scripts["baseline"]),
            run("lazy-cold",scripts["lazy"]),
            run("lazy-warm",scripts["lazy"]),
        ]
    if not results[0]["deepface_loaded_at_cold_start"]:
        raise AssertionError("Baseline must import DeepFace at startup")
    if results[1]["deepface_loaded_at_cold_start"]:
        raise AssertionError("Lazy path must not import DeepFace at startup")
    reduction = (results[0]["rss_after_math_mb"] - results[1]["rss_after_math_mb"])
    summary = {
        "baseline_import_mb": results[0]["rss_after_math_mb"],
        "lazy_import_mb": results[1]["rss_after_math_mb"],
        "cold_ram_reduction_mb": round(reduction,2),
        "baseline_import_seconds": results[0]["cold_import_seconds"],
        "lazy_import_seconds": results[1]["cold_import_seconds"],
        "warm_first_ml_import_seconds": results[2]["deferred_import_seconds"],
        "note":"GitHub hosted-runner, model import only. No biometric-image inference and no production Railway billing measurement.",
        "runs": results,
    }
    RESULT.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("DAVOMAT_BENCHMARK "+json.dumps(summary,sort_keys=True))
    if reduction < 0:
        raise AssertionError("Candidate used more cold-start RAM")
    print("PASS: Baseline and lazy imports measured separately")


if __name__=="__main__":
    main()
