"""Generate MobileFOG_Sultan_DWPose_Extraction_QC.ipynb (self-contained Kaggle notebook).

Usage:  python sultan_dwpose/build_notebook.py
Spec:   docs/superpowers/specs/2026-11-22-dwpose-extraction-qc-design.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
sys.path.insert(0, str(REPO))

from sultan_dwpose.rhama_cells import (  # noqa: E402
    RHAMA_CELL_IDS,
    load_rhama_cells,
    rhama_constants_block,
    transform_rhama_source,
)

RHAMA_NB = REPO / "parkinsonfp-biomed.ipynb"
OUT_NB = REPO / "MobileFOG_Sultan_DWPose_Extraction_QC.ipynb"
DET_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip"


def _strip_inline(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not l.rstrip().endswith("# inline-strip")) + "\n"


def _read(name: str) -> str:
    return (PKG / name).read_text(encoding="utf-8")


def helpers_source() -> str:
    return _strip_inline(_read("nb_helpers.py"))


def worker_source() -> str:
    return _strip_inline(_read("select_person.py")) + "\n\n" + _strip_inline(_read("extract_worker.py"))


def qc_source() -> str:
    return _strip_inline(_read("qc_metrics.py")) + "\n\n" + _strip_inline(_read("qc_visual.py"))


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def _code(text: str, verbatim: bool = False) -> dict:
    src = text if verbatim else text.strip("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


TITLE_MD = """
# Mobile-FOG (Bagian Sultan): Ekstraksi `DWPose-m` + Preprocessing Scenario A / B / D + QC Pose
**Target Lingkungan:** Kaggle Notebook — **Accelerator: GPU T4 x2** (`2× NVIDIA T4`) & **Internet: ON**

Notebook ini padanan `parkinsonfp-biomed.ipynb` milik Rhama, dengan estimator **DWPose-m** (133 keypoint COCO-WholeBody):
1. **Ekstraksi pose paralel (`2× T4`)**: detektor YOLOX-m + **DWPose-m** (bukan RTMW), **setiap frame (30 fps)** & resolusi asli, ke `01_extracted_poses_raw/dwpose_m_sultan/`.
2. **Preprocessing Scenario A / B / D**: kode **disalin persis** dari notebook Rhama (hanya nama folder `yolo26n_lb6_` → `dwpose_m_lb6_`), sehingga hasilnya bisa langsung dipakai `MobileFOG_Rhama_Training_4Models_Kaggle.ipynb` dengan mengganti `DATA_DIR`.
3. **QC pose (hanya melapor, tidak mengubah data)**: metrik per trial (frame valid, confidence rendah, lonjakan, swap kiri-kanan, salah orang, kecocokan jumlah frame), plot sinyal, klip overlay 15 detik untuk dicek manual, dan lembar `qc_manual_review.csv`.

Kode notebook ini dihasilkan oleh `sultan_dwpose/build_notebook.py` (modul + unit test ada di repo). Jangan edit sel kode Rhama di sini, ubah di sumbernya.
"""

INSTALL = """
# rtmlib mewajibkan paket 'onnxruntime' (CPU) yang bentrok dengan 'onnxruntime-gpu' -> pasang --no-deps
!pip uninstall -y -q onnxruntime
!pip install -q --no-deps rtmlib==0.0.16
!pip install -q onnxruntime-gpu tqdm

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch  # import torch dulu: memuat library CUDA/cuDNN yang juga dipakai onnxruntime-gpu
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

{helpers}

print(f"PyTorch     : {{torch.__version__}}")
print(f"onnxruntime : {{ort.__version__}} | providers: {{ort.get_available_providers()}}")
print(f"CUDA avail  : {{torch.cuda.is_available()}} | GPU count: {{torch.cuda.device_count()}}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {{i}}: {{props.name}} ({{props.total_memory / 1024**3:.1f}} GB VRAM)")
require_cuda_provider(ort.get_available_providers())
"""

CONFIG = """
# ==============================================================================
# PATH (auto-detect; bisa juga dijalankan lokal dari root repo)
# ==============================================================================
DATASET_ROOT = find_dataset_root([Path("/kaggle/input"), Path("dataset"), Path("Data/Train Val Test")])
if os.environ.get("MOBILEFOG_OUTPUT_ROOT"):
    OUTPUT_ROOT = Path(os.environ["MOBILEFOG_OUTPUT_ROOT"])
elif Path("/kaggle/working").exists():
    OUTPUT_ROOT = Path("/kaggle/working/mobilefog_sultan")
else:
    OUTPUT_ROOT = Path("output/mobilefog_sultan")
RAW_POSE_DIR = OUTPUT_ROOT / "01_extracted_poses_raw" / "dwpose_m_sultan"
QC_DIR = OUTPUT_ROOT / "03_qc"

# ==============================================================================
# MODEL DWPose-m eksplisit (BUKAN kelas Wholebody rtmlib: semua mode bawaannya = RTMW)
# ==============================================================================
DET_URL = "{det_url}"
POSE_URL = "{pose_url}"

# Bagian yang diekstraksi di sesi ini. Jika estimasi > 9 jam: mis. ["train"] dulu,
# lalu ["val", "test"] di sesi berikutnya (lampirkan output sesi sebelumnya sebagai Input).
RUN_SPLITS = ["train", "val", "test"]
MAX_FRAMES = None          # None = seluruh frame (hanya diubah untuk uji coba)
BENCHMARK_FRAMES = 300     # jumlah frame untuk mengukur kecepatan sebelum ekstraksi penuh

# ==============================================================================
# KONSTANTA PREPROCESSING (disalin persis dari notebook Rhama, cell 2)
# ==============================================================================
{constants}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"DATASET_ROOT : {{DATASET_ROOT}}")
print(f"OUTPUT_ROOT  : {{OUTPUT_ROOT}}")
"""

RESTORE = """
# ==============================================================================
# LANJUTKAN DARI SESI SEBELUMNYA + SKEMA SENDI
# ==============================================================================
RAW_POSE_DIR.mkdir(parents=True, exist_ok=True)
n_restored = restore_previous_outputs(Path("/kaggle/input"), RAW_POSE_DIR)
n_done = len(list(RAW_POSE_DIR.glob("*/*.npz"))) - len(list(RAW_POSE_DIR.glob("*/*.tmp.npz")))
print(f"Dipulihkan dari Input: {n_restored} file | total trial sudah diekstraksi: {n_done}/{len(df_manifest)}")

joint_schema = {
    "estimator": "DWPose-m (rtmpose-m ucoco dw, 256x192) + YOLOX-m HumanArt detector",
    "det_url": DET_URL,
    "pose_url": POSE_URL,
    "num_keypoints": 133,
    "layout": "COCO-WholeBody: 0-16 body (COCO-17), 17-22 feet, 23-90 face, 91-132 hands",
    "coords": "pixel (resolusi asli video), kolom ke-3 = skor keypoint",
    "lb6_indices": LB6_INDICES,
    "lb6_names": LB6_NAMES,
    "feet_indices": {"L_BigToe": 17, "L_SmallToe": 18, "L_Heel": 19, "R_BigToe": 20, "R_SmallToe": 21, "R_Heel": 22},
    "person_selection": "argmax(area * mean_body17_score * (1 - 0.3*|cx-W/2|/(W/2))) per frame, tanpa tracking",
}
with open(RAW_POSE_DIR.parent / "joint_schema.json", "w") as f:
    json.dump(joint_schema, f, indent=2)
"""

WORKER = """
# ==============================================================================
# TULIS SKRIP WORKER EKSTRAKSI (dijalankan 1 proses per GPU, pola notebook Rhama)
# ==============================================================================
WORKER_SRC = r'''{worker}'''
WORKER_PATH = OUTPUT_ROOT / "_dwpose_extract_worker.py"
WORKER_PATH.write_text(WORKER_SRC, encoding="utf-8")
print(f"Worker ditulis ke {{WORKER_PATH}}")
"""

BENCHMARK = """
# ==============================================================================
# UJI KECEPATAN (sekaligus mengunduh model sekali sebelum worker paralel berjalan)
# ==============================================================================
wk = {"__name__": "dwpose_worker"}
exec(compile(WORKER_SRC, str(WORKER_PATH), "exec"), wk)

pending = [r for r in df_manifest.itertuples()
           if r.split in RUN_SPLITS and not wk["is_complete"](RAW_POSE_DIR / r.split / f"{r.trial_id}.npz")]
n_gpus = max(1, torch.cuda.device_count())
if pending:
    det_fn, pose_fn = wk["build_models"](DET_URL, POSE_URL, "cuda")
    wk["check_providers"](det_fn.providers, "cuda")
    wk["check_providers"](pose_fn.providers, "cuda")
    bench = wk["extract_video"](pending[0].video_path, det_fn, pose_fn, max_frames=BENCHMARK_FRAMES)
    ms = float(bench["ms_per_frame"])
    total_frames = sum(int(cv2.VideoCapture(r.video_path).get(cv2.CAP_PROP_FRAME_COUNT)) for r in pending)
    eta = eta_hours(ms, total_frames, n_gpus)
    print(f"Benchmark {pending[0].trial_id}: {ms:.1f} ms/frame | valid={bench['valid'].mean()*100:.1f}% | "
          f"orang/frame rata-rata={bench['n_persons'].mean():.2f}")
    print(f"{len(pending)} trial tersisa, {total_frames} frame, {n_gpus} GPU -> estimasi {eta:.2f} jam")
    if eta > 9:
        print("PERINGATAN: estimasi > 9 jam (batas sesi Kaggle ~12 jam). Ubah RUN_SPLITS di sel konfigurasi "
              "(mis. ['train'] dulu), jalankan ulang, lalu lanjutkan split lain di sesi berikutnya.")
    del det_fn, pose_fn, bench
else:
    print("Semua trial pada RUN_SPLITS sudah diekstraksi.")
"""

LAUNCH = """
# ==============================================================================
# TAHAP 1: EKSTRAKSI DWPose-m PARALEL PADA 2x T4 GPU (SUBPROCESS ISOLATED)
# ==============================================================================
tasks_list = [{"split": r.split, "trial_id": r.trial_id, "video_path": r.video_path}
              for r in df_manifest.itertuples() if r.split in RUN_SPLITS]
cfg_dict = {"out_dir": str(RAW_POSE_DIR), "det_url": DET_URL, "pose_url": POSE_URL,
            "max_frames": MAX_FRAMES, "device": "cuda"}

t0 = time.time()
shards = [tasks_list[i::n_gpus] for i in range(n_gpus)]
procs = []
try:
    for gid in range(n_gpus):
        job_file = OUTPUT_ROOT / f"_dwpose_job_gpu{gid}.json"
        job_file.write_text(json.dumps({"gpu_label": gid, "cfg": cfg_dict, "tasks": shards[gid]}), encoding="utf-8")
        env = os.environ.copy()
        if torch.cuda.is_available():
            env["CUDA_VISIBLE_DEVICES"] = str(gid)
        p = subprocess.Popen([sys.executable, str(WORKER_PATH), str(job_file)], env=env)
        procs.append((gid, p))
    rcs = {gid: p.wait() for gid, p in procs}   # tunggu SEMUA worker
finally:
    for gid, p in procs:                         # kernel di-interrupt / error -> jangan tinggalkan worker hidup
        if p.poll() is None:
            p.terminate()
assert all(rc == 0 for rc in rcs.values()), f"Worker gagal: {rcs}"
print(f"Ekstraksi {len(tasks_list)} video selesai dalam {time.time() - t0:.1f} detik.")
"""

GUARD = """
# Preprocessing butuh SEMUA 71 trial LENGKAP (scaler di-fit pada seluruh train).
# File hasil uji coba MAX_FRAMES / terpotong / rusak tidak dihitung selesai.
missing = [f"{r.split}/{r.trial_id}" for r in df_manifest.itertuples()
           if not wk["is_complete"](RAW_POSE_DIR / r.split / f"{r.trial_id}.npz")]
assert not missing, (f"{len(missing)} trial belum diekstraksi lengkap (mis. {missing[:3]}). Pastikan MAX_FRAMES = None, "
                     "ubah RUN_SPLITS ke split yang tersisa, lalu jalankan ulang notebook "
                     "(lampirkan output sesi sebelumnya sebagai Input).")
print("Semua 71 trial sudah diekstraksi. Lanjut ke preprocessing (kode Rhama).")
"""

PREPROC_MD = """
## Preprocessing Scenario A / B / D (kode Rhama, disalin persis)
Tiga sel berikut diambil otomatis dari `parkinsonfp-biomed.ipynb` (commit `aeb2126`). Satu-satunya perubahan:
`yolo26n_lb6_` → `dwpose_m_lb6_` dan `mobilefog_rhama` → `mobilefog_sultan`. Jika menemukan bug di sini,
laporkan ke tim agar diperbaiki untuk semua estimator, jangan diubah sepihak.
"""

WINDOW_CHECK = """
# Pemeriksaan kewajaran: jumlah window DWPose vs YOLO Rhama (boleh beda sedikit; beda besar = ada masalah)
RHAMA_WINDOWS = {"train": 5535, "val": 1169, "test": 1652}
for src_scen in ["A_baseline_norm", "B_denoised_hampel_accel"]:
    for hz in (15, 30):
        d_root = SCEN_DIR / "D_windowed_stgcn_cls" / f"from_{src_scen}_{hz}hz"
        counts = {sp: int(np.load(d_root / f"{sp}.npz")["y_maj"].shape[0]) for sp in SPLITS}
        diff = {sp: counts[sp] - RHAMA_WINDOWS[sp] for sp in SPLITS}
        print(f"{src_scen} @ {hz}Hz: DWPose={counts} | selisih vs Rhama={diff}")
"""

QC_MD = """
## QC Pose DWPose (hanya melapor)
Metrik dihitung dari pose mentah 30 fps. Ambang flag hanya untuk **memprioritaskan cek manual**, tidak membuang data.
Ambang boleh disetel berdasarkan train/val, **bukan** test. Warna overlay: sendi **kiri = biru**, **kanan = merah**
(swap kiri-kanan terlihat sebagai warna yang bertukar).
"""

QC_RUN = """
# ==============================================================================
# QC: METRIK PER TRIAL, PLOT SINYAL, KLIP OVERLAY, LEMBAR CEK MANUAL
# ==============================================================================
from IPython.display import Image, display

QC_DIR.mkdir(parents=True, exist_ok=True)
qc_rows, qc_cache = [], {}
for r in df_manifest.itertuples():
    z = np.load(RAW_POSE_DIR / r.split / f"{r.trial_id}.npz")
    lbl30 = pd.read_csv(r.label30_path)
    inside = lbl30["inside_imu_mask"].to_numpy() == 1
    y30 = lbl30["y_fog"].to_numpy()
    n_video = int(cv2.VideoCapture(r.video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    n_persons = z["n_persons"] if "n_persons" in z.files else None
    row = qc_trial(z["kps"], z["valid"], z["boxes"], n_persons, float(z["fps"]), LB6_INDICES,
                   n_label30=len(lbl30), n_video=n_video, info_joints=joint_schema["feet_indices"])
    row = {"split": r.split, "trial_id": r.trial_id,
           "fog_pct": 100.0 * float(y30[inside].mean()) if inside.any() else 0.0, **row}
    qc_rows.append(row)
    ev = event_frames(z["kps"], z["valid"], z["boxes"], LB6_INDICES)
    qc_cache[r.trial_id] = (r, ev)
    plot_signals(z["kps"], z["valid"], y30, ev, LB6_INDICES,
                 f"{r.trial_id} [{r.split}] flags={row['flags'] or 'OK'}", QC_DIR / "signals" / f"{r.trial_id}.png")

df_qc = pd.DataFrame(qc_rows)
df_qc.to_csv(QC_DIR / "qc_per_trial.csv", index=False)
flagged = df_qc[df_qc["flags"] != ""]
print(f"QC selesai: {len(df_qc)} trial, {len(flagged)} ter-flag. Tabel: {QC_DIR / 'qc_per_trial.csv'}")
display(flagged[["split", "trial_id", "fog_pct", "flags", "valid_pct", "spike_pct", "lr_swap_per_min",
                 "person_switch_per_min", "multi_person_pct"]])
display(df_qc.describe().T)

import matplotlib.pyplot as plt
metrics = ["valid_pct", "spike_pct", "lr_swap_per_min", "person_switch_per_min", "multi_person_pct",
           "lowconf_pct_L_Ankle", "lowconf_pct_R_Ankle", "lowconf_pct_L_Hip"]
fig, axes = plt.subplots(2, 4, figsize=(18, 7), dpi=100)
for ax, m in zip(axes.ravel(), metrics):
    ax.hist(df_qc[m].dropna(), bins=20, color="#3b82f6", edgecolor="#1e293b")
    ax.set_title(m)
fig.tight_layout()
fig.savefig(QC_DIR / "qc_histograms.png")
plt.close(fig)
display(Image(filename=str(QC_DIR / "qc_histograms.png")))

# Klip overlay 15 detik: semua trial ter-flag + 6 sampel train/val (seed 42)
sample_ids = pick_sample_trials(df_qc[["trial_id", "split", "fog_pct"]], seed=42)
clip_ids = sorted(set(flagged["trial_id"]) | set(sample_ids))
review_rows = []
for tid in clip_ids:
    r, ev = qc_cache[tid]
    z = np.load(RAW_POSE_DIR / r.split / f"{tid}.npz")
    y30 = pd.read_csv(r.label30_path)["y_fog"].to_numpy()
    fps = float(z["fps"])
    for s in clip_windows(len(z["valid"]), fps, y30, ev, seed=42):
        out = QC_DIR / "clips" / f"{tid}_{int(s / fps)}s.mp4"
        render_clip(r.video_path, z["kps"], z["boxes"], z["valid"], y30, s, int(round(15 * fps)), out, LB6_INDICES)
        review_rows.append((tid, out.name))
created = write_review_template(QC_DIR / "qc_manual_review.csv", review_rows)
print(f"{len(review_rows)} klip dibuat untuk {len(clip_ids)} trial (sampel: {sample_ids}).")
print("Lembar cek manual:", QC_DIR / "qc_manual_review.csv", "(baru)" if created else "(sudah ada, tidak ditimpa)")

# Kaggle membatasi jumlah file output (~500): gabungkan plot & klip ke zip
for sub in ("signals", "clips"):
    if (QC_DIR / sub).exists():
        shutil.make_archive(str(QC_DIR / sub), "zip", root_dir=QC_DIR / sub)
        shutil.rmtree(QC_DIR / sub)
n_out_files = sum(1 for p in OUTPUT_ROOT.rglob("*") if p.is_file())
print(f"Plot & klip di-zip: {QC_DIR / 'signals.zip'}, {QC_DIR / 'clips.zip'} | total file output: {n_out_files}")
"""

NEXT_MD = """
## Langkah berikutnya
1. **Simpan output**: jalankan *Save Version → Save & Run All* agar `/kaggle/working/mobilefog_sultan` tersimpan, lalu jadikan output ini Kaggle Dataset untuk notebook training.
2. **Cek manual QC**: unduh `03_qc/clips.zip`, `03_qc/signals.zip`, `03_qc/qc_per_trial.csv`, dan `03_qc/qc_manual_review.csv`. Tonton klip, isi CSV, lalu simpan CSV yang sudah diisi ke repo (`qc/qc_manual_review.csv`). Klip tidak di-commit.
3. **Training**: di `MobileFOG_Rhama_Training_4Models_Kaggle.ipynb`, isi `DATA_DIR` secara manual ke folder `mobilefog_sultan` (auto-detect di sana mencari `mobilefog_rhama`).
"""


def build(out_path: Path = OUT_NB) -> None:
    cells = load_rhama_cells(RHAMA_NB)
    rh = {k: transform_rhama_source(cells[i])[0] for k, i in RHAMA_CELL_IDS.items()}
    worker = worker_source()
    assert "'''" not in worker and not worker.rstrip("\n").endswith("\\"), "worker source tidak aman untuk r'''...'''"

    def fill(template: str, **parts: str) -> str:
        # unescape the template's own {{ }} first, then insert sources untouched
        text = template.replace("{{", "{").replace("}}", "}")
        for key, value in parts.items():
            text = text.replace("{" + key + "}", value)
        return text

    nb_cells = [
        _md(TITLE_MD),
        _code(fill(INSTALL, helpers=helpers_source())),
        _code(fill(CONFIG, det_url=DET_URL, pose_url=POSE_URL, constants=rhama_constants_block(cells[2]))),
        _code(rh["manifest"], verbatim=True),
        _code(RESTORE),
        _code(fill(WORKER, worker=worker)),
        _code(BENCHMARK),
        _code(LAUNCH),
        _code(GUARD),
        _md(PREPROC_MD),
        _code(rh["preprocess_fns"], verbatim=True),
        _code(rh["scenarios"], verbatim=True),
        _code(WINDOW_CHECK),
        _code(rh["smoke"], verbatim=True),
        _md(QC_MD),
        _code(qc_source()),
        _code(QC_RUN),
        _md(NEXT_MD),
    ]
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    Path(out_path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build(OUT_NB)
    print(f"Notebook ditulis: {OUT_NB}")
