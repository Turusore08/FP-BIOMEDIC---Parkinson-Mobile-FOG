"""Pull Rhama's preprocessing cells verbatim from `parkinsonfp-biomed.ipynb`.

The DWPose notebook must preprocess exactly like Rhama's YOLO notebook, so its cells
are copied programmatically instead of re-typed. Only two string replacements are
allowed (see REPLACEMENTS); everything else stays byte-identical.
"""
from __future__ import annotations

import json
from pathlib import Path

RHAMA_CELL_IDS = {"manifest": 3, "preprocess_fns": 5, "scenarios": 6, "smoke": 7}

REPLACEMENTS = {
    "yolo26n_lb6_": "dwpose_m_lb6_",
    "mobilefog_rhama": "mobilefog_sultan",
}

CONSTANT_NAMES = [
    "LB6_INDICES", "LB6_NAMES", "CONF_VALID_THRES", "HAMPEL_WINDOW", "HAMPEL_SIGMA",
    "BUTTER_CUTOFF_HZ", "WINDOW_SEC", "STRIDE_SEC", "RANDOM_SEED",
]


def load_rhama_cells(nb_path: Path) -> list[str]:
    """Source of every cell, indexed by notebook cell index ('' for non-code cells)."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    return ["".join(c["source"]) if c["cell_type"] == "code" else "" for c in nb["cells"]]


def transform_rhama_source(src: str) -> tuple[str, dict[str, int]]:
    counts = {}
    for old, new in REPLACEMENTS.items():
        counts[old] = src.count(old)
        src = src.replace(old, new)
    return src, counts


def rhama_constants_block(cell2_src: str) -> str:
    """Verbatim definition lines (+ their directly preceding comment lines) of CONSTANT_NAMES."""
    lines = cell2_src.splitlines()
    out: list[str] = []
    for name in CONSTANT_NAMES:
        idx = next(i for i, l in enumerate(lines) if l.startswith(f"{name} ="))
        start = idx
        while start > 0 and lines[start - 1].startswith("#") and not lines[start - 1].startswith("# ==="):
            start -= 1
        if out and start > 0 and lines[start - 1].strip() == "":
            out.append("")
        out.extend(lines[start: idx + 1])
    return "\n".join(out) + "\n"
