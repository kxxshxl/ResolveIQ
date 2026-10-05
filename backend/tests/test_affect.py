"""Severity/sentiment affect model: combiner maths (no model needed) + an integration check when the NLI model is cached."""
import json

import numpy as np
import pytest

from app.classification.affect import CUES, AffectModel, cue_hash
from app.core.config import Settings


def _artifact(tmp_path, model="m"):
    n = len(CUES)
    coef = np.zeros((2, n))
    coef[0, list(CUES).index("angry")] = 5.0      # class "angry" fires on the angry cue
    coef[1, list(CUES).index("calm")] = 5.0       # class "neutral" fires on the calm cue
    art = {"version": 1, "model": model, "cue_hash": cue_hash(), "cues": list(CUES),
           "dimensions": {"sentiment": {"classes": ["angry", "neutral"], "coef": coef.tolist(), "intercept": [0.0, 0.0]}}}
    p = tmp_path / "affect.json"
    p.write_text(json.dumps(art), encoding="utf-8")
    return p


def test_combine_is_a_softmax_over_cue_scores(tmp_path):
    m = AffectModel(Settings(affect_model="m", affect_artifact=str(_artifact(tmp_path))))
    m.artifact = json.loads(m.artifact_path.read_text())
    x = np.zeros((2, len(CUES)), dtype=np.float32)
    x[0, list(CUES).index("angry")] = 1.0
    x[1, list(CUES).index("calm")] = 1.0
    out = m.combine(x)
    assert out[0]["sentiment"]["angry"] > 0.99 and out[1]["sentiment"]["neutral"] > 0.99
    for row in out:
        assert abs(sum(row["sentiment"].values()) - 1.0) < 1e-9


def test_artifact_for_a_different_model_or_cue_set_is_rejected_not_silently_used(tmp_path):
    m = AffectModel(Settings(affect_model="other-model", affect_artifact=str(_artifact(tmp_path, model="m"))))
    m.load()
    assert m.available is False  # degrade to the rule/kNN fallback rather than serve mismatched weights


def test_missing_artifact_degrades_gracefully(tmp_path):
    m = AffectModel(Settings(affect_artifact=str(tmp_path / "nope.json")))
    m.load()
    assert m.available is False


def test_disabled_by_configuration():
    m = AffectModel(Settings(affect_enabled=False))
    m.load()
    assert m.available is False


@pytest.fixture(scope="module")
def real_model():
    m = AffectModel(Settings())
    m.load()
    if not m.available:
        pytest.skip("NLI model or data/models/affect_v1.json not available (run scripts/train_affect.py)")
    return m


def test_real_model_separates_angry_urgent_from_calm_minor(real_model):
    out = real_model.predict_sync([
        "This is a disgrace. I run a business from home and have lost customers all week. Fix it NOW or I'm cancelling.",
        "Hi, just a heads up that the wifi is a little slow in the evenings. No rush at all, thanks.",
    ])
    assert out[0]["sentiment"]["angry"] > out[1]["sentiment"]["angry"]
    sev = lambda r: r["severity"]["high"] + r["severity"]["critical"]  # noqa: E731
    assert sev(out[0]) > sev(out[1])
    assert max(out[1]["severity"], key=out[1]["severity"].get) in ("low", "medium")
