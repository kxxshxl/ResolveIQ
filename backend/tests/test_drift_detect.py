"""Drift detection: the statistics, the explainable report, and the deterministic injection demo.

The pure tests use synthetic Gaussian-blob embeddings with fixed seeds (fast, no model). The demo test uses the real embedding model on the project's
labelled complaints and asserts only what the design promises: injected changes of a clear size are found and attributed correctly, an unchanged
window is not flagged. Detection RATES at small effect sizes are reported by `python -m app.evaluation.drift_demo`, not asserted here.
"""
import json

import numpy as np
import pytest

from app.drift import stats
from app.drift.detect import DriftConfig, Window, _cluster_weights, analyze, holm, sample_for_embedding

D, TOPICS = 32, 6
_rng0 = np.random.default_rng(0)
CENTERS = _rng0.normal(size=(TOPICS, D))
CENTERS /= np.linalg.norm(CENTERS, axis=1, keepdims=True)
NEW_CENTER = _rng0.normal(size=D)
NEW_CENTER /= np.linalg.norm(NEW_CENTER)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def window(n, rng, weights=None, new_topic=0, abstain=0.0, evidence=0.85, severity=("medium",), tag="r"):
    w = np.ones(TOPICS) / TOPICS if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    k = rng.choice(TOPICS, size=n, p=w)
    X = _unit(CENTERS[k] + rng.normal(0, 0.12, (n, D)))
    rows = [{"request_id": f"{tag}{i}", "complaint": f"complaint {tag}{i} topic {k[i]}", "intent": f"intent_{k[i]}", "product": "mobile",
             "severity": severity[i % len(severity)], "sentiment": "neutral", "evidence": float(np.clip(rng.normal(evidence, 0.05), 0, 1)),
             "status": "abstained" if rng.random() < abstain else "resolved"} for i in range(n)]
    if new_topic:
        X = np.vstack([X, _unit(NEW_CENTER + rng.normal(0, 0.12, (new_topic, D)))])
        rows += [{"request_id": f"{tag}n{i}", "complaint": f"brand new topic {tag}n{i}", "intent": "intent_0", "product": "mobile", "severity": "medium",
                  "sentiment": "neutral", "evidence": 0.4, "status": "abstained"} for i in range(new_topic)]
    return Window(rows=rows, emb_rows=rows, X=X.astype(np.float32))


CFG = DriftConfig(permutations=200)


# ------------------------------------------------------------------ statistics
def test_psi_is_zero_for_identical_mixes_and_grows_with_shift():
    same = {"a": 50, "b": 50}
    assert stats.psi(same, same)[0] == pytest.approx(0.0, abs=1e-9)
    small, _ = stats.psi({"a": 50, "b": 50}, {"a": 55, "b": 45})
    big, contrib = stats.psi({"a": 50, "b": 50}, {"a": 85, "b": 15})
    assert small < 0.10 < 0.25 < big and set(contrib) == {"a", "b"}


def test_psi_is_finite_when_a_category_is_absent_from_one_window():
    score, _ = stats.psi({"a": 100}, {"a": 60, "new": 40})
    assert np.isfinite(score) and score > 0.25


def test_chi_square_pools_rare_categories_and_skips_degenerate_tables():
    r = stats.chi2_homogeneity({"a": 100, "b": 100, "rare": 1}, {"a": 100, "b": 100, "rare2": 1})
    assert set(r["pooled"]) == {"rare", "rare2"} and r["p_value"] > 0.5
    assert stats.chi2_homogeneity({"a": 10}, {"a": 10})["p_value"] == 1.0
    assert stats.chi2_homogeneity({}, {"a": 10})["p_value"] == 1.0


def test_chi_square_detects_a_real_shift():
    assert stats.chi2_homogeneity({"a": 200, "b": 200}, {"a": 60, "b": 20})["p_value"] < 0.01
    assert stats.chi2_homogeneity({"a": 200, "b": 200}, {"a": 41, "b": 39})["p_value"] > 0.5


def test_category_changes_flags_new_vanished_and_moved_categories():
    rows = {c["label"]: c for c in stats.category_changes({"a": 100, "b": 100, "gone": 20}, {"a": 40, "b": 160, "fresh": 12})}
    assert rows["fresh"]["flag"] == "new" and rows["gone"]["flag"] == "vanished"
    assert rows["a"]["flag"] == "down" and rows["b"]["flag"] == "up"
    assert rows["a"]["delta"] < 0 < rows["b"]["delta"] and rows["a"]["psi_share"] > 0


def test_proportion_test_uses_exact_test_for_small_counts_and_handles_zero():
    assert stats.proportion_test(0, 50, 0, 50) == 1.0
    assert stats.proportion_test(1, 20, 12, 20) < 0.01
    assert stats.proportion_test(0, 0, 3, 10) == 1.0


def test_ks_detects_a_shifted_distribution_not_a_resampled_one():
    rng = np.random.default_rng(1)
    a = rng.normal(0.8, 0.05, 300)
    assert stats.ks_test(a, rng.normal(0.8, 0.05, 100))["p_value"] > 0.01
    shifted = stats.ks_test(a, rng.normal(0.7, 0.05, 100))
    assert shifted["p_value"] < 1e-6 and shifted["d"] > 0.5
    assert stats.ks_test([0.5], [0.6])["p_value"] == 1.0


def test_centroid_shift_permutation_p_value_separates_shifted_from_same():
    rng = np.random.default_rng(2)
    same = stats.centroid_shift(window(200, rng).X, window(80, rng).X, 200, np.random.default_rng(3))
    moved = stats.centroid_shift(window(200, rng).X, window(80, rng, weights=[8, 1, 1, 1, 1, 1]).X, 200, np.random.default_rng(3))
    assert same["p_value"] > 0.01 and moved["p_value"] <= 0.01 and moved["distance"] > same["distance"]


def test_holm_controls_the_family_and_is_never_weaker_than_bonferroni():
    p = {"a": 0.0004, "b": 0.003, "c": 0.2, "d": 0.5}
    assert holm(p, 0.01) == {"a", "b"}            # b: 0.003 <= 0.01/3 once "a" is out of the way; Bonferroni at 0.01/4 = 0.0025 would keep only "a"
    assert holm({"a": 0.0004, "b": 0.009, "c": 0.2, "d": 0.5}, 0.01) == {"a"}
    assert holm({"a": 0.02, "b": 0.5}, 0.01) == set()


# ------------------------------------------------------------------ the report
def test_unchanged_traffic_raises_no_alert_across_many_seeds():
    alarms = 0
    for s in range(40):
        rng = np.random.default_rng(100 + s)
        alarms += analyze(window(100, rng, tag="r"), window(300, rng, tag="b"), CFG)["status"] == "alert"
    assert alarms <= 1   # target 1% per report; 40 trials, so one chance event is within tolerance


def test_category_surge_is_detected_and_attributed():
    rng = np.random.default_rng(5)
    r = analyze(window(120, rng, weights=[5, 1, 1, 1, 1, 1], tag="r"), window(400, rng, tag="b"), CFG)
    assert r["status"] == "alert"
    alert = next(a for a in r["alerts"] if a["signal"] == "intent_distribution")
    assert "intent_0" in alert["message"] and alert["effect"]["psi"] >= 0.10
    top = max(r["distributions"]["intent"]["categories"], key=lambda c: c["delta"])
    assert top["label"] == "intent_0" and top["flag"] == "up"


def test_abstention_and_evidence_degradation_alert_but_a_tiny_change_does_not():
    rng = np.random.default_rng(6)
    base = window(400, rng, tag="b", abstain=0.05)
    bad = analyze(window(120, rng, tag="r", abstain=0.35, evidence=0.6), base, CFG)
    assert {"abstention_rate", "evidence_confidence"} <= {a["signal"] for a in bad["alerts"]}
    mild = analyze(window(120, rng, tag="r", abstain=0.08, evidence=0.84), base, CFG)
    assert mild["status"] == "ok"


def test_severity_mix_shift_is_detected():
    rng = np.random.default_rng(7)
    r = analyze(window(120, rng, tag="r", severity=("high", "high", "critical", "medium")), window(400, rng, tag="b", severity=("low", "medium", "medium", "high")), CFG)
    assert "severity_distribution" in {a["signal"] for a in r["alerts"]}


def test_emerging_cluster_is_found_and_described():
    rng = np.random.default_rng(8)
    r = analyze(window(100, rng, new_topic=20, tag="r"), window(400, rng, tag="b"), CFG)
    sig = [c for c in r["emerging_clusters"] if c["significant"]]
    assert sig and r["status"] == "alert"
    c = max(sig, key=lambda c: c["size"])
    assert c["size"] >= 15 and c["kind"] == "new_topic" and c["covered_by_corpus"] is False and c["p_value"] <= CFG.alpha
    assert all(i.startswith("rn") for i in c["request_ids"][:5]) and c["examples"] and c["mean_evidence"] < 0.6
    assert any(a["signal"] == "emerging_cluster" and a["cluster_id"] == c["cluster_id"] for a in r["alerts"])


def test_cluster_weights_sum_to_one_and_lean_towards_unexplained_clusters():
    w = _cluster_weights([True, False, False, False], 0.75)
    assert sum(w) == pytest.approx(1.0) and w[0] == pytest.approx(0.75) and w[1] == pytest.approx(0.25 / 3)
    assert _cluster_weights([True, True], 0.75) == [0.5, 0.5] and _cluster_weights([False] * 4, 0.75) == [0.25] * 4
    assert _cluster_weights([True, False, False], 0.0) == pytest.approx([1 / 3] * 3)   # 0 = the plain Bonferroni split


def test_weighting_finds_a_small_unexplained_topic_that_the_equal_split_misses():
    """4 new-topic complaints among 100: their exact p-value (about 0.004) is significant at 1% alone but not after an equal split over the ~7
    clusters tested. Leaning the budget towards the cluster the corpus does not explain finds it; nothing else changes."""
    from dataclasses import replace
    found = {0.0: 0, 0.75: 0}
    for seed in range(5):
        for share in found:
            rng = np.random.default_rng(100 + seed)
            r = analyze(window(100, rng, new_topic=4, tag="r"), window(300, rng, tag="b"), replace(CFG, unexplained_weight=share))
            found[share] += any(c["significant"] and c["kind"] == "new_topic" for c in r["emerging_clusters"])
    assert found == {0.0: 0, 0.75: 5}


def test_a_surge_of_a_known_topic_is_labelled_a_surge_not_a_new_topic():
    rng = np.random.default_rng(9)
    r = analyze(window(150, rng, weights=[1, 1, 1, 1, 1, 12], tag="r"), window(450, rng, tag="b"), CFG)
    kinds = {c["kind"] for c in r["emerging_clusters"] if c["significant"]}
    assert "new_topic" not in kinds


def test_too_little_traffic_is_reported_and_never_alerted():
    rng = np.random.default_rng(10)
    r = analyze(window(12, rng, abstain=1.0, tag="r"), window(400, rng, tag="b"), CFG)
    assert r["status"] == "insufficient_data" and r["alerts"] == [] and "need at least" in r["note"]


def test_embedding_tests_are_skipped_without_embeddings_but_label_tests_still_run():
    rng = np.random.default_rng(11)
    rec, base = window(120, rng, weights=[6, 1, 1, 1, 1, 1], tag="r"), window(400, rng, tag="b")
    rec.X = base.X = None
    r = analyze(rec, base, CFG)
    assert r["embedding"]["status"] == "skipped" and r["emerging_clusters"] == []
    assert "intent_distribution" in {a["signal"] for a in r["alerts"]}


def test_report_is_json_serialisable_and_deterministic():
    def go():
        rng = np.random.default_rng(12)
        return analyze(window(100, rng, new_topic=15, tag="r"), window(300, rng, tag="b"), CFG)

    a, b = go(), go()
    assert json.dumps(a) == json.dumps(b)   # plain types only (it is stored as JSONB) and a fixed seed reproduces it
    assert a["tests"]["method"] == "Holm-Bonferroni" and a["tests"]["n"] == len(a["tests"]["p_values"]) >= 7


def test_sample_for_embedding_collapses_duplicates_and_thins_evenly():
    rows = [{"request_id": str(i), "complaint": f"  Same   Text {i % 5} "} for i in range(50)]
    assert len(sample_for_embedding(rows, 100)) == 5
    distinct = [{"request_id": str(i), "complaint": f"text {i}"} for i in range(100)]
    thinned = sample_for_embedding(distinct, 10)
    assert len(thinned) == 10 and int(thinned[-1]["request_id"]) > 80


# ------------------------------------------------------------------ the injection demo (real embeddings, real labelled complaints)
@pytest.fixture(scope="module")
def traffic(embedder):
    from app.evaluation import drift_demo as dd

    return dd, dd.load_traffic(embedder)


def test_demo_unchanged_windows_are_not_flagged(traffic):
    dd, t = traffic
    rep = dd.run_control(t, 30, DriftConfig(novelty_threshold=t.novelty_threshold))
    assert rep["false_alarms"] <= 2, rep   # 1% target; the interval on 30 trials is wide, so only a gross regression fails


@pytest.mark.parametrize("name", ["intent_shift_40", "severity_shift_55", "vocabulary_shift"])
def test_demo_clear_injected_shifts_are_detected(traffic, name):
    dd, t = traffic
    sc = next(s for s in dd.SCENARIOS if s.name == name)
    res = dd.run_scenario(t, sc, 12, DriftConfig(novelty_threshold=t.novelty_threshold))
    assert res["detection_rate"] >= 0.75, res
    if name == "intent_shift_40":
        assert res["correct_attribution"] == res["trials"]   # the boosted intent is always the top mover


def test_demo_new_topic_is_recovered_and_links_to_a_discovery_proposal(traffic):
    dd, t = traffic
    cfg = DriftConfig(novelty_threshold=t.novelty_threshold)
    sc = next(s for s in dd.SCENARIOS if s.name == "new_topic_10")
    res = dd.run_scenario(t, sc, 12, cfg)
    assert res["detected"] >= 4 and res["unmatched_significant_clusters_per_report"] <= 0.1, res
    ex = dd.illustrate(t, cfg, 20)["new_topics_3x8"]
    assert ex["trial"] is not None and any("new topic" in a for a in ex["alerts"])
    assert ex["discovery_link"]["links"], "the drift cluster must overlap a proposal built by the discovery code from the same window"
