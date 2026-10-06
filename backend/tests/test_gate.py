from app.evaluation.gate import evaluate, lookup

RESULTS = {"classification": {"strategies": {"ensemble": {"gold": {"intent": {"accuracy": 0.95}}}}}, "x": {"flag": True, "lat": 12.0}}


def test_lookup_walks_dotted_paths_and_ignores_non_numbers():
    assert lookup(RESULTS, "classification.strategies.ensemble.gold.intent.accuracy") == 0.95
    assert lookup(RESULTS, "classification.nope") is None
    assert lookup(RESULTS, "x.flag") is None          # bool is not a metric


def test_floor_ceiling_and_missing_metrics():
    base = "classification.strategies.ensemble.gold.intent.accuracy"
    res = evaluate(RESULTS, [{"path": base, "min": 0.9}, {"path": base, "min": 0.99}, {"path": "x.lat", "max": 20}, {"path": "x.lat", "max": 5},
                             {"path": "absent.metric", "min": 0.1}])
    assert [r.ok for r in res] == [True, False, True, False, False]
    assert res[-1].value is None and res[-1].bound == "measured"   # an unmeasured check fails: a silently skipped suite must not pass CI


def test_lookup_indexes_into_lists_and_rejects_bad_indexes():
    res = {"sweep": [{"purity": 0.5}, {"purity": 0.8, "inner": [{"v": 3}]}]}
    assert lookup(res, "sweep.1.purity") == 0.8 and lookup(res, "sweep.0.purity") == 0.5
    assert lookup(res, "sweep.1.inner.0.v") == 3
    assert lookup(res, "sweep.2.purity") is None and lookup(res, "sweep.x.purity") is None


def test_every_floor_in_the_repository_thresholds_file_is_measured_in_the_recorded_run():
    import json

    from app.core.config import REPO_ROOT

    results = json.loads((REPO_ROOT / "data/eval/results/latest.json").read_text(encoding="utf-8"))
    checks = json.loads((REPO_ROOT / "data/eval/thresholds.json").read_text(encoding="utf-8"))["checks"]
    failed = [r.path for r in evaluate(results, checks) if not r.ok]
    assert failed == [], f"the committed results violate (or do not contain) these floors: {failed}"
