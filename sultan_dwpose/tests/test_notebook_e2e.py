"""Run the whole generated notebook top-to-bottom with fake rtmlib/onnxruntime on a synthetic dataset.

~1-2 minutes. Needs dataset/ (only for the real trial ids/splits).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sultan_dwpose import build_notebook as bn
from sultan_dwpose import extract_worker as ew
from sultan_dwpose.tests.e2e.make_dataset import N_FRAMES, make_dataset

REPO = Path(__file__).resolve().parents[2]
E2E = Path(__file__).resolve().parent / "e2e"

pytestmark = pytest.mark.skipif(not (REPO / "dataset" / "train" / "videos").exists(), reason="needs dataset/")


def _run(nb, cwd, out, max_frames=None):
    env = dict(os.environ, MOBILEFOG_OUTPUT_ROOT=str(out))
    env.pop("E2E_MAX_FRAMES", None)
    if max_frames:
        env["E2E_MAX_FRAMES"] = str(max_frames)
    p = subprocess.run([sys.executable, str(E2E / "run_nb.py"), str(nb)], cwd=cwd, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    return p.stdout + p.stderr


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    assert make_dataset(REPO / "dataset", root / "dataset") == 71
    nb = root / "nb.ipynb"
    bn.build(nb)
    out = root / "out"
    log1 = _run(nb, root, out, max_frames=100)   # a "test run" with MAX_FRAMES
    log2 = _run(nb, root, out)                   # the real run in the same output folder
    return out, log1, log2


def test_guard_rejects_truncated_test_run(e2e):
    out, log1, _ = e2e
    assert "ALL CELLS OK" not in log1
    assert "belum diekstraksi" in log1


def test_full_run_reextracts_and_finishes(e2e):
    out, _, log2 = e2e
    assert "ALL CELLS OK" in log2, log2[-3000:]
    raws = sorted((out / "01_extracted_poses_raw" / "dwpose_m").glob("*/*.npz"))
    assert len(raws) == 71
    assert all(ew.is_complete(p) for p in raws)
    assert all(np.load(p)["kps"].shape[0] == N_FRAMES for p in raws)
    d = np.load(out / "02_processed_scenarios" / "D_windowed_stgcn_cls" / "from_B_denoised_hampel_accel_15hz" / "train.npz")
    assert d["X"].shape[0] >= 47 * 15


def test_qc_outputs_zipped_and_file_count(e2e):
    out, _, _ = e2e
    qc = out / "03_qc"
    df = pd.read_csv(qc / "qc_per_trial.csv")
    assert len(df) == 71
    assert (qc / "signals.zip").exists() and (qc / "clips.zip").exists()
    assert not (qc / "signals").exists() and not (qc / "clips").exists()
    assert (qc / "qc_manual_review.csv").exists()
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    assert n_files < 500, n_files
