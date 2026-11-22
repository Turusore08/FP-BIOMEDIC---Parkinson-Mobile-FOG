import numpy as np

from sultan_dwpose.select_person import select_main_person

W, H = 1280, 720


def _person(x0, x1, y0=100, y1=700, body=0.9, rest=0.5):
    box = np.array([x0, y0, x1, y1], np.float32)
    kp = np.zeros((133, 2), np.float32)
    kp[:, 0] = np.linspace(x0, x1, 133)
    kp[:, 1] = np.linspace(y0, y1, 133)
    sc = np.full(133, rest, np.float32)
    sc[:17] = body
    return box, kp, sc


def _stack(*persons):
    boxes, kps, scores = zip(*persons)
    return np.stack(boxes), np.stack(kps), np.stack(scores)


def test_select_empty():
    kp, box, valid, n = select_main_person(
        np.zeros((0, 4), np.float32), np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32), W, H)
    assert kp.shape == (133, 3) and kp.dtype == np.float32 and not kp.any()
    assert box.shape == (4,) and not box.any()
    assert valid is False
    assert n == 0


def test_select_single():
    b, k, s = _person(500, 800)
    kp, box, valid, n = select_main_person(*_stack((b, k, s)), W, H)
    np.testing.assert_allclose(kp[:, :2], k)
    np.testing.assert_allclose(kp[:, 2], s)
    np.testing.assert_allclose(box, b)
    assert valid is True
    assert n == 1


def test_select_prefers_central_over_larger_edge():
    edge = _person(0, 480)        # area 1.2x central, far from center
    central = _person(440, 840)   # centered at x=640
    kp, box, valid, n = select_main_person(*_stack(edge, central), W, H)
    np.testing.assert_allclose(box, central[0])
    assert n == 2 and valid is True


def test_select_uses_body_scores_only():
    a = _person(440, 840, body=0.3, rest=0.99)
    b = _person(440, 840, body=0.8, rest=0.1)
    kp, box, valid, n = select_main_person(*_stack(a, b), W, H)
    np.testing.assert_allclose(kp[:, 2], b[2])
