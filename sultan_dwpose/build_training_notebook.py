"""Generate MobileFOG_DWPose_Training_4Models_Kaggle.ipynb from the team training notebook.

Every cell is copied VERBATIM except an exact, counted list of string replacements:
team member names removed, paths pointed at the DWPose PatientLock output, and the LOSO scenario
fixed to "B_15hz" (decided without looking at test scores; "AUTO_BEST" picks by test PR-AUC).

Usage:  python sultan_dwpose/build_training_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEAM_NB = REPO / "MobileFOG_Rhama_Training_4Models_Kaggle.ipynb"
OUT_NB = REPO / "MobileFOG_DWPose_Training_4Models_Kaggle.ipynb"

# Applied in order (longest/most specific first).
REPLACEMENTS = {
    "# Mobile-FOG (Bagian Rhama): Training": "# Mobile-FOG (DWPose-m + PatientLock): Training",
    "Notebook ini langsung memakai output ekstraksi `YOLO26n-pose` yang sudah ada (**TIDAK perlu ekstraksi YOLO ulang**)":
        "Notebook ini langsung memakai output ekstraksi `DWPose-m` + PatientLock yang sudah ada (**TIDAK perlu ekstraksi ulang**)",
    "dengan Sultan & Faiz.": "dengan estimator pose lain (YOLO26n, MediaPipe).",
    "mobilefog_rhama_patientlock": "mobilefog_dwpose_patientlock",
    "mobilefog-rhama-extracted/mobilefog_rhama": "mobilefog-dwpose-extracted/mobilefog_dwpose_patientlock",
    "mobilefog-rhama-training": "mobilefog-dwpose-training",
    "mobilefog_rhama": "mobilefog_dwpose_patientlock",
    "rhama_4models_comparison.png": "dwpose_4models_comparison.png",
    'LOSO_SCENARIO = "AUTO_BEST"  # "AUTO_BEST" (otomatis pilih skenario terbaik Tahap A) atau mis. "B_30hz"':
        'LOSO_SCENARIO = "B_15hz"     # ditetapkan tanpa melihat skor test, sama untuk semua estimator '
        '("AUTO_BEST" memilih berdasarkan test PR-AUC)',
}
EXPECTED_COUNTS = {
    "# Mobile-FOG (Bagian Rhama): Training": 1,
    "Notebook ini langsung memakai output ekstraksi `YOLO26n-pose` yang sudah ada (**TIDAK perlu ekstraksi YOLO ulang**)": 1,
    "dengan Sultan & Faiz.": 1,
    "mobilefog_rhama_patientlock": 1,
    "mobilefog-rhama-extracted/mobilefog_rhama": 1,
    "mobilefog-rhama-training": 1,
    "mobilefog_rhama": 3,
    "rhama_4models_comparison.png": 1,
    'LOSO_SCENARIO = "AUTO_BEST"  # "AUTO_BEST" (otomatis pilih skenario terbaik Tahap A) atau mis. "B_30hz"': 1,
}


def build(out_path: Path = OUT_NB) -> dict[str, int]:
    nb = json.loads(TEAM_NB.read_text(encoding="utf-8"))
    counts = {k: 0 for k in REPLACEMENTS}
    for c in nb["cells"]:
        src = "".join(c["source"])
        for old, new in REPLACEMENTS.items():
            counts[old] += src.count(old)
            src = src.replace(old, new)
        c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    assert counts == EXPECTED_COUNTS, f"Notebook training tim berubah, periksa REPLACEMENTS: {counts}"
    Path(out_path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return counts


if __name__ == "__main__":
    build(OUT_NB)
    print(f"Notebook ditulis: {OUT_NB}")
