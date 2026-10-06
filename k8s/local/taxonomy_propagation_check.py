"""Multi-replica check: a taxonomy change made through ONE API replica must become visible on the OTHER replica and the worker.

    python k8s/local/taxonomy_propagation_check.py [--wait 40]

Adds one throw-away intent through replica A, then polls replica B's GET /api/v1/taxonomy until it reports the new version
(or --wait seconds pass). Exit code 0 = propagated. Run against the local kind cluster only: it changes the cluster's taxonomy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_test import Forward, call, kubectl, secret  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=40)
    a = ap.parse_args()
    key = {"X-API-Key": secret("API_KEYS").split(",")[0]}
    pods = sorted(kubectl("get", "pods", "-l", "app=backend", "-o", "jsonpath={.items[*].metadata.name}").split())
    assert len(pods) >= 2, "needs two backend replicas"
    with Forward(f"pod/{pods[0]}", 18001, 8000) as pa, Forward(f"pod/{pods[1]}", 18002, 8000) as pb:
        def version(f):
            _, body, _, _ = call(f.url, "GET", "/api/v1/taxonomy", headers=key)
            return body["version"], len(body["labels"]["intent"])
        before_a, before_b = version(pa), version(pb)
        label = "propagation_check_" + uuid.uuid4().hex[:6]
        s, body, _, _ = call(pa.url, "POST", "/api/v1/taxonomy/labels", {"dimension": "intent", "label_id": label, "description": "Throw-away class used to test taxonomy propagation between replicas.",
                                                                  "keywords": ["propagation"], "examples": ["propagation check complaint"]}, key)
        print(f"added intent {label} through {pods[0]}: HTTP {s}")
        t0 = time.time()
        while True:
            va, vb = version(pa), version(pb)
            if vb[0] == va[0] and vb != before_b:
                print(f"{pods[1]} sees the change after {time.time() - t0:.1f} s: A={va} B={vb} (before: A={before_a} B={before_b})")
                return 0
            if time.time() - t0 > a.wait:
                print(f"NOT propagated after {a.wait:.0f} s: A={va} B={vb} (before: A={before_a} B={before_b})")
                return 1
            time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
