"""The recorded-results reader behind the Evaluation view: reads only files that exist, labels them with path and write time, never invents data."""
import json

from app.evaluation.summary import FILES, load_results


def _write(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def test_missing_files_are_omitted_not_fabricated(tmp_path):
    out = load_results(tmp_path)
    assert out == {"files": {}}


def test_present_files_are_included_with_path_and_write_time(tmp_path):
    _write(tmp_path, FILES["pgvector_scale"], {"rows": 1000})
    _write(tmp_path, FILES["latest"], {"results": {"classification": {"accuracy": 0.9}}, "meta_by_run": {"x": 1}})
    out = load_results(tmp_path)
    assert out["pgvector_scale"] == {"rows": 1000} and out["latest"]["results"]["classification"]["accuracy"] == 0.9
    assert "meta_by_run" not in out["latest"], "bulky per-run metadata stays out of the console payload"
    assert out["files"]["pgvector_scale"]["path"] == FILES["pgvector_scale"] and out["files"]["pgvector_scale"]["written"].endswith("+00:00")
    assert set(out["files"]) == {"pgvector_scale", "latest"}


def test_corrupt_file_is_skipped(tmp_path):
    p = tmp_path / FILES["drift_demo"]
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    assert load_results(tmp_path) == {"files": {}}


def test_endpoint_serves_the_repository_results(client):
    r = client.get("/api/v1/evaluation/results")
    assert r.status_code == 200
    body = r.json()
    assert "files" in body and set(body["files"]) <= set(FILES)
    for key, meta in body["files"].items():
        assert key in body and meta["path"] == FILES[key]
