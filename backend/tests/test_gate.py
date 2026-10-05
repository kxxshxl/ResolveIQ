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
