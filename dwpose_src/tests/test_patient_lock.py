import inspect
import json
import re
from pathlib import Path

import numpy as np

from dwpose_src import patient_lock as pl

REPO = Path(__file__).resolve().parents[2]


def _team_worker_src():
    nb = json.loads((REPO / "yolo_notebooks" / "01_extract_yolo26_patientlock.ipynb").read_text(encoding="utf-8"))
    return "".join(nb["cells"][4]["source"])


def _team_func(name):
    src = _team_worker_src()
    m = re.search(rf"^def {name}\(.*?(?=^def |^if __name__)", src, re.S | re.M)
    assert m, name
    return m.group(0).rstrip() + "\n"


def test_crop_helpers_are_verbatim_copies_of_team_worker():
    for name in ("_box_iou_one_to_many", "_crop_and_mask_patient"):
        ours = inspect.getsource(getattr(pl, name)).rstrip() + "\n"
        assert ours == _team_func(name), name


def _frame(h=720, w=1280):
    return np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3).astype(np.uint8)


def test_crop_pads_and_clips():
    crop, off, core, pbox = pl._crop_and_mask_patient(_frame(), np.array([0.4, 0.1, 0.6, 0.9]), np.empty((0, 4)), 0.15, True)
    assert off == (473, 0)
    assert crop.shape[:2] == (720, 807 - 473)
    np.testing.assert_allclose(core, [512 - 473, 72, 768 - 473, 648])
    np.testing.assert_allclose(pbox, [512, 72, 768, 648])


def test_mask_supervisors_keeps_core():
    frame = _frame()
    sup = np.array([[0.35, 0.0, 0.45, 1.0]])  # overlaps the crop's left padding AND the patient core
    crop, off, core, _ = pl._crop_and_mask_patient(frame, np.array([0.4, 0.1, 0.6, 0.9]), sup, 0.15, True)
    x0, y0, x1, y1 = core.astype(int)
    np.testing.assert_array_equal(crop[y0:y1, x0:x1], frame[off[1] + y0:off[1] + y1, off[0] + x0:off[0] + x1])
    assert (crop[300, 0] == 114).all()          # padding area under the supervisor box is masked
    crop2, *_ = pl._crop_and_mask_patient(frame, np.array([0.4, 0.1, 0.6, 0.9]), sup, 0.15, False)
    np.testing.assert_array_equal(crop2[300, 0], frame[300, off[0]])


def test_pick_box_in_crop():
    core = np.array([40, 70, 300, 650], np.float32)
    boxes = np.array([[0, 0, 60, 100], [45, 80, 290, 640], [200, 60, 330, 660]], np.float32)
    assert pl.pick_box_in_crop(boxes, core, 0.20) == 1
    assert pl.pick_box_in_crop(np.array([[0, 0, 60, 100]], np.float32), core, 0.20) is None
    assert pl.pick_box_in_crop(np.zeros((0, 4), np.float32), core, 0.20) is None


def test_frame_supervisor_boxes_csr():
    flat = np.array([[0.1, 0.1, 0.2, 0.9, 0.8], [0.7, 0.1, 0.9, 0.9, 0.7], [0.0, 0.0, 0.1, 0.5, 0.6]], np.float32)
    off = np.array([0, 2, 2, 3])
    np.testing.assert_allclose(pl.frame_supervisor_boxes(flat, off, 0), flat[:2, :4])
    assert pl.frame_supervisor_boxes(flat, off, 1).shape == (0, 4)
    np.testing.assert_allclose(pl.frame_supervisor_boxes(flat, off, 2), flat[2:3, :4])
