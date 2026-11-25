"""Execute the generated notebook's code cells in order (pip lines stripped) with fake rtmlib/onnxruntime.

Usage: python run_nb.py NOTEBOOK  (cwd = folder that contains dataset/)
Env:   MOBILEFOG_OUTPUT_ROOT (required), E2E_MAX_FRAMES (optional override after the config cell)
Prints "ALL CELLS OK" on success; exits 1 at the first failing cell.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

FAKES = Path(__file__).resolve().parent / "fakepkgs"
sys.path.insert(0, str(FAKES))
os.environ["PYTHONPATH"] = str(FAKES) + os.pathsep + os.environ.get("PYTHONPATH", "")

nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ns = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith("!"))
    t0 = time.time()
    print(f"######## CELL {i}", flush=True)
    try:
        exec(compile(src, f"cell{i}", "exec"), ns)
    except Exception:
        traceback.print_exc()
        print(f"######## CELL {i} FAILED", flush=True)
        sys.exit(1)
    print(f"######## CELL {i} ok ({time.time() - t0:.1f}s)", flush=True)
    if "DATASET_ROOT = find_dataset_root" in src:
        import torch
        torch.cuda.device_count = lambda: 2  # simulate 2x T4
        if os.environ.get("E2E_MAX_FRAMES"):
            ns["MAX_FRAMES"] = int(os.environ["E2E_MAX_FRAMES"])
print("ALL CELLS OK")
