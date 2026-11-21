import numpy as np

from mmft import calibration as cal
from mmft.evaluate import summarize


def _sample(n=20000, k=4, scale=2.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k)) * scale
    p = cal.softmax(z)
    y = np.array([rng.choice(k, p=row) for row in p])
    return z, y


def test_calibrated_model_has_low_ece():
    z, y = _sample()
    p = cal.softmax(z)
    conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
    assert cal.ece(conf, correct) < 0.015


def test_overconfidence_is_detected_and_temperature_recovers_it():
    z, y = _sample()
    hot = z * 3.0  # 3x overconfident
    p = cal.softmax(hot)
    assert cal.ece(p.max(1), (p.argmax(1) == y).astype(float)) > 0.1
    t = cal.fit_temperature(hot, y)
    assert abs(t - 3.0) < 0.15
    fixed = cal.softmax(hot, t)
    assert cal.nll(fixed, y) < cal.nll(p, y)
    assert cal.ece(fixed.max(1), (fixed.argmax(1) == y).astype(float)) < 0.02


def test_missing_options_are_masked():
    logits = np.array([[2.0, 1.0, -np.inf, -np.inf], [0.0, 0.0, 0.0, 0.0]])
    p = cal.softmax(logits)
    assert np.allclose(p.sum(1), 1) and p[0, 2] == 0
    assert np.isfinite(cal.fit_temperature(logits, np.array([0, 3])))


def test_brier_and_nll_reference_values():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    y = np.array([0, 1])
    assert cal.brier(p, y) == 0.25  # (0 + 0.5) / 2
    assert np.isclose(cal.nll(p, y), np.log(2) / 2)


def test_summarize_groups_and_worst_group():
    rows = [
        {"qid": "1", "group": "a", "answer": 0, "logits": [5.0, 0.0, 0.0, 0.0]},
        {"qid": "2", "group": "a", "answer": 1, "logits": [0.0, 5.0, 0.0, 0.0]},
        {"qid": "3", "group": "b", "answer": 2, "logits": [5.0, 0.0, 0.0, 0.0]},
        {"qid": "4", "group": "b", "answer": 0, "logits": [5.0, 0.0, 0.0]},
    ]
    s = summarize(rows)
    assert s["accuracy"] == 0.75
    assert s["per_group"] == {"a": {"n": 2, "acc": 1.0}, "b": {"n": 2, "acc": 0.5}}
    assert s["worst_group"] == "b" and s["worst_group_acc"] == 0.5
