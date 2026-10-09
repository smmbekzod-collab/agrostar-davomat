"""Private-data-free, end-to-end baseline/lazy face inference benchmark.

Uses the public astronaut sample in scikit-image; never uses Davomat users or
production DB. Same face.py source, same FaceNet512/OpenCV settings for both.
Outputs aggregate metrics and pass/fail checks, never biometric embeddings.
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
CAND = Path("davomat_v4_2_lazy_deepface_candidate.zip")
REPORT = Path("davomat_face_inference_benchmark.json")

CHILD = r'''
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

app=types.ModuleType("app");app.__path__=[]
core=types.ModuleType("app.core");core.__path__=[]
cfg=types.ModuleType("app.core.config")
cfg.settings=SimpleNamespace(
    face_detector="opencv",face_model="Facenet512",
    face_threshold=.3,max_upload_bytes=10000000)
sys.modules.update({"app":app,"app.core":core,"app.core.config":cfg})

def rss():
    for ln in Path("/proc/self/status").read_text().splitlines():
        if ln.startswith("VmRSS:"):return int(ln.split()[1])/1024
    return 0

started=time.monotonic()
spec=importlib.util.spec_from_file_location("app.services.face",sys.argv[1])
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import_duration=time.monotonic()-started
idle_rss=rss()
samples=Path(sys.argv[2]).read_bytes()
out={"idle_mb":round(idle_rss,2),"import_s":round(import_duration,3),"runs":[]}
for require_liveness in (False,False,True):
    t=time.monotonic()
    try:
        embedding, score, _ = mod.analyze_selfie(samples, require_liveness=require_liveness)
        # Float embedding is not sent to CI logs, only to local parent process.
        result={"state":"accepted","n":len(embedding),
                "embedding":[float(x) for x in embedding],
                "liveness_score_available":score is not None}
    except Exception as exc:
        # No staff image or sensitive traceback. Same input to both versions.
        result={"state":"rejected","exception":type(exc).__name__,
                "detail":str(exc)[:400],  # public test image only; no user data
                "reason_type":("liveness" if "jonli yuz" in str(exc).lower()
                      or "spoof" in str(exc).lower() else "other")}
    result["require_liveness"]=require_liveness
    result["elapsed_s"]=round(time.monotonic()-t,3)
    result["rss_after_mb"]=round(rss(),2)
    out["runs"].append(result)
print("FACE_RESULT "+json.dumps(out),flush=True)
'''

def source_from_zip(z: Path):
    with ZipFile(z) as archive:
        names=[x.filename for x in archive.infolist()
               if x.filename.endswith("/backend/app/services/face.py")]
        assert len(names)==1
        return archive.read(names[0]).decode("utf-8")

def benchmark(mode:str,path:Path,photo:Path):
    env=os.environ.copy()
    env.update({"TF_CPP_MIN_LOG_LEVEL":"3","CUDA_VISIBLE_DEVICES":"-1",
                "OMP_NUM_THREADS":"2","TF_NUM_INTEROP_THREADS":"2",
                "TF_NUM_INTRAOP_THREADS":"2"})
    proc=subprocess.Popen([sys.executable,"-c",CHILD,str(path),str(photo)],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,env=env)
    peak=0
    start=time.monotonic()
    while proc.poll() is None:
        try:
            for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    peak=max(peak,int(line.split()[1])/1024)
                    break
        except OSError:
            pass
        if time.monotonic()-start>420:
            proc.kill()
            break
        time.sleep(.08)
    stdout,stderr=proc.communicate(timeout=20)
    if proc.returncode:
        raise RuntimeError(f"{mode} exited {proc.returncode}: {stderr[-2500:]}")
    out=[x[12:] for x in stdout.splitlines() if x.startswith("FACE_RESULT ")]
    if len(out)!=1:
        raise RuntimeError(f"Cannot get {mode} results: {stdout[-1200:]}; {stderr[-1000:]}")
    data=json.loads(out[0])
    data["peak_rss_mb"]=round(peak,2)
    data["mode"]=mode
    return data

def main():
    from skimage.data import astronaut
    from PIL import Image
    import io
    base=Image.fromarray(astronaut()).convert("RGB")
    # Upper-left crop focuses on the astronaut's face, preserving a fully
    # public, reproducible sample without accessing any user photographs.
    cropped=base.crop((70,0,360,360))
    cropped.thumbnail((640,640))
    with tempfile.TemporaryDirectory(prefix="davomat-face-e2e-") as t:
        temp=Path(t)
        photo=temp/"public_astronaut.jpg"
        cropped.save(photo,format="JPEG",quality=92)
        originals={}
        for key,archive in (("original",BASE),("lazy",CAND)):
            dest=temp/(key+"_face.py")
            dest.write_text(source_from_zip(archive),encoding="utf-8")
            originals[key]=dest
        a=benchmark("original",originals["original"],photo)
        b=benchmark("lazy",originals["lazy"],photo)
    for x in (a,b):
        assert len(x["runs"])==3, "Expected two repeats plus strict liveness"
    comparison={}
    for i,kind in enumerate(("first_inference","warmed_inference","strict_anti_spoof")):
        left,right=a["runs"][i],b["runs"][i]
        comparison[kind]={"original":left["state"],"candidate":right["state"],
                          "original_s":left["elapsed_s"],"candidate_s":right["elapsed_s"]}
        print("PUBLIC_SAMPLE_OUTCOME",kind,left["state"],right["state"],left.get("reason_type"),right.get("reason_type"))
        if left["state"]!=right["state"]:
            raise AssertionError(f"Candidate changed {kind} result: {comparison[kind]}")
        if left["state"]=="accepted":
            import numpy as np
            l=np.array(left["embedding"],dtype="float32")
            r=np.array(right["embedding"],dtype="float32")
            assert l.shape==r.shape and l.size>0
            dist=float(1-np.dot(l,r)/(np.linalg.norm(l)*np.linalg.norm(r)))
            comparison[kind]["cosine_disagreement"]=round(dist,8)
            assert abs(dist)<0.002, "Embeddings changed too much"
        elif kind=="strict_anti_spoof":
            if left["reason_type"]!=right["reason_type"]:
                raise AssertionError("Strict anti-spoof failure type changed")
    if a["runs"][0]["state"]!="accepted":
        print("PUBLIC_FACE_DIAG",repr(a["runs"][0]),repr(b["runs"][0]))
        raise AssertionError("Public face was not recognized, benchmark inconclusive")
    # Do not put raw biometric vectors in report artifact.
    for x in (a,b):
        for entry in x["runs"]:
            entry.pop("embedding",None)
    report={"source":"scikit-image public astronaut sample",
            "config":"Facenet512/OpenCV; separate Linux processes, CPU only",
            "production_changes":False,"original":a,"candidate":b,
            "comparison":comparison,
            "caution":"Public photo is not live-person liveness acceptance test; no production endpoint tested."}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("FACE_E2E_SUMMARY "+json.dumps({
        "baseline_idle_mb":a["idle_mb"],"candidate_idle_mb":b["idle_mb"],
        "baseline_peak_mb":a["peak_rss_mb"],"candidate_peak_mb":b["peak_rss_mb"],
        "comparisons":comparison},sort_keys=True))
    print("PASS: Public-photo inference results are equivalent (no production changes)")

if __name__=="__main__":
    main()
