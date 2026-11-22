"""Synthetic 71-trial dataset with the real trial ids/splits (600 frames, 160x120 each)."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

N_FRAMES = 600


def make_dataset(real_root: Path, out_root: Path) -> int:
    n_tr = 0
    for vp in sorted(Path(real_root).glob("*/videos/*.mp4")):
        split, tid = vp.parent.parent.name, vp.stem
        for sub in ("videos", "labels_30hz", "labels_15hz", "imu_128hz"):
            (out_root / split / sub).mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(out_root / split / "videos" / f"{tid}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 29.978, (160, 120))
        for i in range(N_FRAMES):
            vw.write(np.full((120, 160, 3), (i * 3) % 255, np.uint8))
        vw.release()
        n30 = N_FRAMES - (1 if n_tr % 14 == 0 else 0)  # like the real data: some labels 1 row short
        y = np.zeros(n30, int)
        if n_tr % 3 == 0:
            y[200:320] = 1
        elif n_tr % 3 == 1:
            y[250:265] = 1
        mask = np.zeros(n30, int)
        mask[30:570] = 1
        pd.DataFrame({"frame_idx": np.arange(n30), "time_s": np.arange(n30) / 30.0, "y_fog": y,
                      "inside_imu_mask": mask}).to_csv(out_root / split / "labels_30hz" / f"{tid}.csv", index=False)
        h = n30 // 2
        pd.DataFrame({"frame_idx": np.arange(h), "time_s": np.arange(h) / 15.0, "y_fog": y[::2][:h],
                      "inside_imu_mask": mask[::2][:h]}).to_csv(out_root / split / "labels_15hz" / f"{tid}.csv", index=False)
        n_tr += 1
    return n_tr
