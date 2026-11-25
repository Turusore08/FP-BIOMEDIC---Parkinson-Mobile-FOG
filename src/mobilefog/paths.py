"""Centralized path registry for Mobile-FOG scripts, notebooks, and tests."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Tracked repository directories
DATA_DIR = REPO_ROOT / "data"
METADATA_CSV = DATA_DIR / "metadata" / "PDFEinfo.csv"
RAW_VIDEOS_DIR = Path(
    os.environ.get("MOBILEFOG_DATA_DIR", str(REPO_ROOT / "Data" / "Train Val Test"))
)
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SCRIPTS_DIR = REPO_ROOT / "scripts"
RESULTS_METRICS_DIR = REPO_ROOT / "results" / "metrics"
DOCS_DIR = REPO_ROOT / "docs"
REFERENCES_DIR = REPO_ROOT / "references"

# Local gitignored artifacts & heavy outputs
OUTPUTS_DIR = Path(
    os.environ.get("MOBILEFOG_OUTPUTS_DIR", str(REPO_ROOT / "outputs"))
)
POSE_EXTRACTION_DIR = OUTPUTS_DIR / "pose_extraction"
TRAINING_BBOX_DIR = OUTPUTS_DIR / "training" / "bbox_35subj"
TRAINING_CLEAN_DIR = OUTPUTS_DIR / "training" / "clean_32subj"
DEMO_VIDEOS_DIR = OUTPUTS_DIR / "demo_videos"
SLIDES_OUTPUT_DIR = OUTPUTS_DIR / "slides"
