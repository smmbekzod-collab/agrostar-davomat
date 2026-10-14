"""Static read-only audit of production-derived backend ML import paths.

Reads only Python source in the bundled ZIP, never imports or executes it,
never connects to any Railway API/database, and prints no environment values.
"""
import ast
import json
from pathlib import Path
from zipfile import ZipFile

ARCHIVE = Path("agrostar_davomat_4_2_PUSH_REMINDER_CALENDAR_FULL.zip")
LIBS = {"tensorflow","torch","deepface","retinaface","cv2","keras","tf_keras"}
INTERESTING = {"analyze_selfie","extract_embedding","get_embedding",
               "create_app","lifespan","startup","run","main"}
with ZipFile(ARCHIVE) as z:
    files = [n for n in z.namelist() if "/backend/app/" in n
             and n.endswith(".py") and "/__pycache__/" not in n]
    report = []
    for path in files:
        source = z.read(path).decode("utf-8", "replace")
        try:
            root = ast.parse(source)
        except SyntaxError:
            continue
        imports = []
        calls = []
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in LIBS:
                        imports.append((node.lineno, alias.name))
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in LIBS:
                    imports.append((node.lineno, node.module))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("represent","extract_faces","ProcessPoolExecutor","shutdown","Timer"):
                    calls.append((node.lineno, node.func.attr))
        if imports or calls:
            report.append({
                "path":path,"imports":sorted(set(imports)),
                "calls":sorted(set(calls))[:60],
            })
    print("ZIP_BACKEND_FILES",len(files))
    print("ML_IMPORT_CALL_AUDIT",json.dumps(report,ensure_ascii=False))
