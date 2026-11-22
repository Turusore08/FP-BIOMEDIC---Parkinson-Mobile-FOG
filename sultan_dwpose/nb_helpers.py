"""Notebook helpers: GPU provider check, dataset auto-detect, cross-session resume, ETA."""
import os
import shutil
from pathlib import Path

SKIP_WALK = {"videos", "imu_128hz", "labels_15hz", "labels_30hz"}


def require_cuda_provider(providers):
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            f"onnxruntime tidak punya CUDAExecutionProvider (providers={providers}). "
            "Pastikan paket 'onnxruntime-gpu' yang terpasang (bukan versi CPU): "
            "!pip uninstall -y onnxruntime && !pip install onnxruntime-gpu, lalu restart kernel. "
            "Accelerator Kaggle juga harus 'GPU T4 x2'."
        )


def find_dataset_root(candidates):
    for cand in candidates:
        cand = Path(cand)
        if not cand.exists():
            continue
        for d, dirs, _ in os.walk(cand):
            p = Path(d)
            if (p / "train" / "videos").is_dir() and (p / "val" / "videos").is_dir():
                return p
            dirs[:] = sorted(x for x in dirs if x not in SKIP_WALK)
    raise FileNotFoundError(
        f"Folder dataset (berisi train/videos & val/videos) tidak ditemukan di: {[str(c) for c in candidates]}. "
        "Tambahkan dataset 'Train Val Test' sebagai Kaggle Input."
    )


def restore_previous_outputs(input_root, raw_dir):
    """Copy DWPose raw npz from an attached previous output into raw_dir (resume across sessions)."""
    input_root, raw_dir = Path(input_root), Path(raw_dir)
    if not input_root.exists():
        return 0
    n = 0
    for f in sorted(input_root.rglob("01_extracted_poses_raw/dwpose_m_sultan/*/*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        dst = raw_dir / f.parent.name / f.name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp.npz")
        shutil.copy2(f, tmp)
        os.replace(tmp, dst)  # atomic: an interrupted copy never looks like a finished file
        n += 1
    return n


def eta_hours(ms_per_frame, total_frames, n_gpus):
    return ms_per_frame * total_frames / 1000.0 / 3600.0 / max(int(n_gpus), 1)
