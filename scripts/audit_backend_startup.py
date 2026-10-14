"""Read-only inspect FastAPI startup/scheduler and persistent activity by AST.

No execution of source; no secrets, literals, SQL, credentials or photos logged.
"""
import ast,json,zipfile
archive="agrostar_davomat_4_2_PUSH_REMINDER_CALENDAR_FULL.zip"
patterns=("main.py","work_scheduler.py","config.py","lifespan.py","face_worker.py","face.py")
with zipfile.ZipFile(archive) as z:
    files=[p for p in z.namelist() if "/backend/" in p and p.endswith(".py")]
    for path in files:
        if not path.endswith(patterns) and "/backend/app/api/" not in path:
            continue
        tree=ast.parse(z.read(path).decode("utf-8"))
        imports=[]
        defs=[]
        calls=[]
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                if isinstance(n,ast.Import):
                    imports += [a.name for a in n.names]
                elif n.module:
                    imports.append(n.module)
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                defs.append(n.name)
            if isinstance(n,(ast.Expr,ast.Assign,ast.AnnAssign)):
                for x in ast.walk(n):
                    if isinstance(x,ast.Call):
                        name=getattr(x.func,"attr",None) or getattr(x.func,"id",None)
                        if name and name in ("create_task","add_job","start","startup","include_router","add_event_handler","launch","on_event","run"):
                            calls.append(name)
        if path.endswith(("main.py","work_scheduler.py","config.py","lifespan.py")) or "scheduler" in path or "face" in path:
            print("STARTUP_AUDIT "+json.dumps({"path":path,"imports":sorted(set(imports))[:80],"definitions":defs[:70],"top_level_calls":calls[:30]},ensure_ascii=False))
    # Print only bounded function-call structure around scheduler, never
    # full source or any identifier values loaded from configuration.
    for path in files:
        if "scheduler" not in path and not path.endswith("/backend/app/main.py"):
            continue
        tree=ast.parse(z.read(path).decode("utf-8"))
        for n in ast.walk(tree):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                calls=[]
                for child in ast.walk(n):
                    if isinstance(child,ast.Call):
                        c=child.func
                        name=getattr(c,"attr",None) or getattr(c,"id",None)
                        if name and name in {"sleep","add_job","start","shutdown","create_task","run","execute","scalars","commit","refresh","list","select","wait_for","fetchall","fetchone","connect","execute","schedule","get_db","session"}:
                            calls.append(name)
                print("TASK_LOOP_AUDIT "+json.dumps({"path":path,"fn":n.name,"calls":calls[:50]}))
