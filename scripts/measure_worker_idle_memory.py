"""Read-only reproduction of AI worker child process RAM after idle timeout.

Run only on GitHub Actions with a public astronomy image; no private or
production data. Uses current production-derived FACE WORKER package.
"""
from __future__ import annotations
import json,os,pathlib,subprocess,sys,tempfile,zipfile

ARCHIVE="davomat_v4_2_face_worker_candidate.zip"
RUNNER=r'''
import os,time,io,json,gc
import psutil
from PIL import Image
from skimage.data import astronaut
from app.services.face import analyze_selfie
from app.services.face_worker import worker_running

def rss_mb():return round(psutil.Process().memory_info().rss/1024**2,2)
def children():
    p=psutil.Process()
    out=[]
    for x in p.children(recursive=True):
        try:
            if x.is_running() and x.status()!=psutil.STATUS_ZOMBIE:
                out.append({"rss_mb":round(x.memory_info().rss/1024**2,2),
                            "name":x.name()})
        except psutil.Error:
            pass
    return out

assert not worker_running()
base=Image.fromarray(astronaut()).convert("RGB").crop((70,0,360,360))
file=io.BytesIO();base.save(file,format="JPEG",quality=92)
before={"worker_running":worker_running(),"main_rss_mb":rss_mb(),"children":children()}
t=time.perf_counter()
vec,_,_=analyze_selfie(file.getvalue(),require_liveness=False)
assert len(vec)==512
elapsed=round(time.perf_counter()-t,3)
immediate={"worker_running":worker_running(),"main_rss_mb":rss_mb(),"children":children()}
time.sleep(39)
gc.collect()
time.sleep(3)
after={"worker_running":worker_running(),"main_rss_mb":rss_mb(),"children":children()}
print("WORKER_IDLE_METRICS "+json.dumps({
"public_only":True,
"production_changed":False,
"worker_idle_seconds":30,
"inference_seconds":elapsed,
"before":before,
"immediate":immediate,
"after_42_seconds":after,
},sort_keys=True),flush=True)
'''
with tempfile.TemporaryDirectory(prefix="worker-ram-test-") as tmp:
    with zipfile.ZipFile(ARCHIVE) as z:
        face_name=next(x for x in z.namelist() if x.endswith("/backend/app/services/face.py"))
        prefix=face_name.removesuffix("app/services/face.py")
        backend=pathlib.Path(tmp)/prefix
        for name in z.namelist():
            if name.startswith(prefix+"app/") and not name.endswith("/"):
                dest=pathlib.Path(tmp)/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(name))
    run_script=backend/"idle_worker_smoke.py"
    run_script.write_text(RUNNER,encoding="utf-8")
    env=os.environ.copy()
    env.update({"FACE_WORKER_IDLE_SECONDS":"30","TF_CPP_MIN_LOG_LEVEL":"3",
                "CUDA_VISIBLE_DEVICES":"-1","PYTHONPATH":str(backend)})
    child=subprocess.run([sys.executable,str(run_script)],cwd=backend,
                          env=env,capture_output=True,text=True,timeout=250)
    if child.returncode:
        print(child.stderr[-3500:])
        raise SystemExit(child.returncode)
    rows=[r.removeprefix("WORKER_IDLE_METRICS ") for r in child.stdout.splitlines()
          if r.startswith("WORKER_IDLE_METRICS ")]
    if len(rows)!=1:
        raise RuntimeError("Missing worker idle diagnostics")
    report=json.loads(rows[0])
    pathlib.Path("davomat_worker_idle_rss.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("WORKER_IDLE_METRICS "+json.dumps(report,sort_keys=True))
    main_growth=report["after_42_seconds"]["main_rss_mb"]-report["before"]["main_rss_mb"]
    child_high_rss=sum(p["rss_mb"] for p in report["after_42_seconds"]["children"] if p["rss_mb"]>250)
    print("IDLE_RSS_INTERPRETATION "+json.dumps({
        "main_rss_growth_mb":round(main_growth,2),
        "large_surviving_children_mb":round(child_high_rss,2),
        "worker_pool_retired":not report["after_42_seconds"]["worker_running"]
    }))
