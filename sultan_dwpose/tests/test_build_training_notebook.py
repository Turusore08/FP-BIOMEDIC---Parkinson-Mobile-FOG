import ast
import difflib
import json
import re
from pathlib import Path

import pytest

from sultan_dwpose import build_training_notebook as bt

REPO = Path(__file__).resolve().parents[2]
NAMES = re.compile(r"(?<![a-z])(rhama|sultan|faiz|panji|arya|hamdi|jailani|daulay|hizbullah)(?![a-z])", re.I)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("tr") / "nb.ipynb"
    counts = bt.build(out)
    return json.loads(out.read_text(encoding="utf-8")), counts


def _src(nb):
    return ["".join(c["source"]) for c in nb["cells"]]


def test_replacement_counts_exact(built):
    _, counts = built
    assert counts == bt.EXPECTED_COUNTS


def test_no_personal_names(built):
    nb, _ = built
    hits = [l for s in _src(nb) for l in s.splitlines() if NAMES.search(l)]
    assert hits == []
    assert not NAMES.search(bt.OUT_NB.name)


def test_only_allowed_lines_changed(built):
    nb, _ = built
    team = json.loads(bt.TEAM_NB.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == len(team["cells"])
    for ours, theirs in zip(_src(nb), _src(team)):
        removed = [l[2:] for l in difflib.ndiff(theirs.splitlines(), ours.splitlines()) if l.startswith("- ")]
        for r in removed:
            assert any(old in r for old in bt.REPLACEMENTS), r


def test_config_points_to_dwpose_and_fixed_loso(built):
    nb, _ = built
    cfg = _src(nb)[2]
    assert 'DATA_DIR = Path("/kaggle/input/GANTI_DENGAN_SLUG_OUTPUT_ANDA/mobilefog_dwpose_patientlock")' in cfg
    assert re.search(r'^LOSO_SCENARIO = "B_15hz"', cfg, re.M)
    joined = "\n".join(_src(nb))
    assert "dwpose_4models_comparison.png" in joined


def test_code_cells_parse_and_no_outputs(built):
    nb, _ = built
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))))
