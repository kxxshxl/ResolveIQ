from app.observability.drift import DriftThresholds, compare, distribution, js_divergence, summarize


def rows(n, abstain=0.0, ev=0.85, intent="billing_dispute", severity="medium", sentiment="neutral"):
    out = []
    for i in range(n):
        out.append({"status": "abstained" if i < n * abstain else "resolved", "intent": intent, "severity": severity,
                    "sentiment": sentiment, "evidence": ev, "latency_ms": 1000 + i})
    return out


def test_js_divergence_bounds_and_symmetry():
    a, b = {"x": 1.0}, {"y": 1.0}
    assert js_divergence(a, a) == 0.0 and js_divergence(a, b) == 1.0
    p, q = {"x": 0.7, "y": 0.3}, {"x": 0.4, "y": 0.6}
    assert js_divergence(p, q) == js_divergence(q, p) and 0 < js_divergence(p, q) < 0.2
    assert js_divergence({}, {}) == 0.0


def test_stable_traffic_raises_no_alert():
    r = compare(summarize(rows(100)), summarize(rows(400)), None)
    assert r["status"] == "ok" and r["alerts"] == []


def test_abstention_surge_and_evidence_drop_and_mix_shift_all_alert():
    recent = summarize(rows(60, abstain=0.4, ev=0.6, intent="esim_activation"))
    r = compare(recent, summarize(rows(400, abstain=0.05)), 0.5)
    signals = {a["signal"] for a in r["alerts"]}
    assert {"abstention_rate", "mean_evidence", "intent_distribution", "negative_feedback"} <= signals and r["status"] == "alert"


def test_too_little_traffic_is_reported_not_alerted():
    r = compare(summarize(rows(5, abstain=1.0)), summarize(rows(400)), None, DriftThresholds(min_requests=30))
    assert r["status"] == "insufficient_data" and r["alerts"] == []


def test_distribution_normalises():
    assert distribution(["a", "a", "b"]) == {"a": 2 / 3, "b": 1 / 3}
