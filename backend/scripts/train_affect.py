"""Train the severity/sentiment combiner on the *synthetic* tickets and write data/models/affect_v1.json.

Only ~14 NLI cue scores per ticket are the features, so the learned part is tiny (14 x 4 weights per dimension); the language
understanding comes from the pretrained NLI model. Hand-written gold sets are never used for training (see docs/evaluation.md).

    python scripts/train_affect.py [--out data/models/affect_v1.json] [--c 2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from app.classification.affect import CUES, AffectModel, cue_hash  # noqa: E402
from app.core.config import REPO_ROOT, get_settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--c", type=float, default=2.0, help="inverse L2 strength")
    args = ap.parse_args()
    s = get_settings()
    out = Path(args.out) if args.out else s.data_dir / "models" / "affect_v1.json"
    rows = [json.loads(l) for l in (REPO_ROOT / "data" / "processed" / "tickets.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    model = AffectModel(s)
    model.load(require_artifact=False)
    if not model.available:
        raise SystemExit("NLI model could not be loaded")
    X = model.cue_scores([r["complaint_text"] for r in rows])
    art: dict = {"version": 1, "model": s.affect_model, "cue_hash": cue_hash(), "cues": list(CUES), "trained_on": f"{len(rows)} synthetic tickets",
                 "trained_at": datetime.now(timezone.utc).isoformat(), "sklearn": sklearn.__version__, "C": args.c, "dimensions": {}}
    for dim in ("severity", "sentiment"):
        y = [r[dim] for r in rows]
        clf = LogisticRegression(C=args.c, max_iter=5000, class_weight="balanced").fit(X, y)
        art["dimensions"][dim] = {"classes": [str(c) for c in clf.classes_], "coef": clf.coef_.round(5).tolist(),
                                  "intercept": clf.intercept_.round(5).tolist()}
        print(f"{dim}: train acc {np.mean(clf.predict(X) == np.array(y)):.3f} classes={list(clf.classes_)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
