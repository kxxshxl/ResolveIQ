"""Deterministic drift demonstration: inject known distribution changes into otherwise unchanged traffic and measure whether the monitor notices.

    python -m app.evaluation.drift_demo [--trials 30] [--aa-trials 200]      (no database, no LLM; needs only the embedding model)

Design, so the result cannot be flattering by construction:
  * The traffic is the project's 301 labelled hand-written complaints (queries + gold + blind), not generated for this purpose.
    Every trial splits them into two DISJOINT halves and draws the baseline from one and the recent window from the other (with
    replacement, as real traffic repeats itself; exact duplicates are collapsed before any embedding test, exactly as in production).
  * Each scenario changes exactly one thing about the recent window and nothing else. The control scenario changes nothing and is
    run many times: its alert rate is the measured false-alarm rate.
  * Detection counts only when the RIGHT signal fires (and, where there is ground truth, points at the right thing: the boosted
    intent, the injected topic).
  * Seeds are fixed, so a re-run reproduces the numbers on the same embedding model and library versions.
Not simulated: the real pipeline. "Evidence" here is the top-1 cosine similarity of a complaint to the resolved-ticket corpus (a stand-in
for the pipeline's evidence confidence), and the "classifier" is the ground-truth label (novel complaints get their nearest ticket's intent,
as the live classifier would). The statistics are the production code (app.drift.detect.analyze); the traffic is the stand-in.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import REPO_ROOT, Settings
from app.discovery.clustering import DiscoveryParams, Neighbors, build_proposals
from app.drift.detect import DriftConfig, Window, analyze, sample_for_embedding
from app.evaluation.datasets import load_jsonl

N_BASELINE, N_RECENT = 300, 100   # default window sizes; the *_large scenarios use 4x
STALE_INTENTS = ("slow_speed", "plan_change", "service_outage")   # intents whose tickets "disappear" from the corpus in the stale-KB scenario
BOOST_INTENT = "billing_dispute"
NOVEL_CLASSES = ("number_porting", "voicemail_issue", "tv_streaming_app")   # the three classes the discovery parameters were never tuned on (10 complaints each)
RETURNING_INTENTS = ("billing_dispute", "device_issue", "slow_speed", "service_outage")   # known intents for the held-out "returns after an absence" check

_CHAT = {"please": "pls", "because": "cuz", "you": "u", "your": "ur", "thanks": "thx", "thank": "thx", "internet": "net", "connection": "conn",
         "message": "msg", "messages": "msgs", "customer": "cust", "service": "svc", "and": "n", "with": "w", "have": "hv", "been": "bn",
         "since": "sinc", "again": "agn", "about": "abt", "tomorrow": "tmrw", "problem": "prob", "account": "acct", "number": "no"}


def chat_style(text: str) -> str:
    """The same complaint as it might arrive from a chat widget: lower case, abbreviated, no punctuation. Fixed rules, no randomness."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return " ".join(_CHAT.get(w, w) for w in words) + " thx"


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


# ------------------------------------------------------------------ traffic
@dataclass
class Traffic:
    items: list[dict]            # in-domain complaints: text, intent, product, severity, sentiment, vec, ev, ev_stale, chat_vec, chat_ev
    novel: dict[str, list[dict]]
    corpus_X: np.ndarray
    corpus_intent: list[str]
    corpus_product: list[str]
    corpus_text: list[str]
    abstain_below: float         # proxy cut-off: the 5th percentile of in-domain evidence, i.e. what a 5%-abstention system would refuse
    novelty_threshold: float     # proxy stand-in for DISCOVERY_NOVELTY_THRESHOLD: the in-domain median
    marginals: dict[str, tuple[list[str], np.ndarray]]


def load_traffic(embedder, data_dir: Path | None = None) -> Traffic:
    d = data_dir or REPO_ROOT / "data"
    pool = []
    for name in ("queries", "gold", "gold_v2", "gold_blind"):
        pool += load_jsonl(d / "eval" / f"{name}.jsonl")
    seen, items = set(), []
    for r in pool:
        k = " ".join(r["text"].lower().split())
        if k not in seen:
            seen.add(k)
            items.append({k2: r[k2] for k2 in ("text", "intent", "product", "severity", "sentiment")})
    tickets = load_jsonl(d / "processed" / "tickets.jsonl")
    cX = embedder.encode_sync([t["complaint_text"] for t in tickets])
    cint = [t["intent"] for t in tickets]
    keep = np.array([i not in STALE_INTENTS for i in cint])
    X = embedder.encode_sync([i["text"] for i in items])
    Xc = embedder.encode_sync([chat_style(i["text"]) for i in items])
    for it, v, ev, ev_s, vc, evc in zip(items, X, (X @ cX.T).max(1), (X @ cX[keep].T).max(1), Xc, (Xc @ cX.T).max(1)):
        it.update(vec=v, ev=float(ev), ev_stale=float(ev_s), chat_vec=vc, chat_ev=float(evc))
    novel_rows = load_jsonl(d / "eval" / "novel_stream.jsonl") + load_jsonl(d / "eval" / "evolving_queries.jsonl")
    NX = embedder.encode_sync([r["text"] for r in novel_rows])
    novel: dict[str, list[dict]] = {}
    for r, v in zip(novel_rows, NX):
        sims = cX @ v
        nn = int(sims.argmax())
        novel.setdefault(r["intent"], []).append({"text": r["text"], "intent": cint[nn], "product": tickets[nn]["product"], "vec": v,
                                                  "ev": float(sims.max()), "truth": r["intent"]})
    ev_all = np.array([i["ev"] for i in items])
    marg = {k: (sorted({i[k] for i in items}), np.array([sum(i[k] == v for i in items) for v in sorted({i[k] for i in items})], dtype=float)) for k in ("severity", "sentiment")}
    marg = {k: (v[0], v[1] / v[1].sum()) for k, v in marg.items()}
    return Traffic(items, novel, cX, cint, [t["product"] for t in tickets], [t["complaint_text"] for t in tickets],
                   float(np.quantile(ev_all, 0.05)), float(np.median(ev_all)), marg)


def _window(prefix: str, rows_in: list[dict], t: Traffic) -> Window:
    rows = [{"request_id": f"{prefix}-{i}", "complaint": it["text"], "intent": it["intent"], "product": it["product"], "severity": it["severity"],
             "sentiment": it["sentiment"], "evidence": it["ev"], "status": "abstained" if it["ev"] < t.abstain_below else "resolved", "truth": it.get("truth"),
             "_vec": it["vec"]} for i, it in enumerate(rows_in)]
    emb = sample_for_embedding(rows, 10**9)   # distinct complaints only, as the service does
    return Window(rows=rows, emb_rows=emb, X=np.stack([r["_vec"] for r in emb]).astype(np.float32))


# ------------------------------------------------------------------ scenarios
def _split(t: Traffic, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """Two disjoint halves with (almost) identical label mix, so the two windows can differ only through the injected change and sampling noise.

    Complaints are ordered by (intent, product, severity, sentiment), shuffled within each identical-label cell, and dealt alternately to the halves:
    every label combination splits evenly, odd leftovers alternate. A purely random split would give the halves different label mixes of their own,
    which the tests would (correctly) report as drift between two different populations.
    """
    cells: dict[tuple, list[dict]] = {}
    for i in t.items:
        cells.setdefault((i["intent"], i["product"], i["severity"], i["sentiment"]), []).append(i)
    a, b, to_a = [], [], bool(rng.integers(2))
    for key in sorted(cells):
        cell = cells[key]
        for j in rng.permutation(len(cell)):
            (a if to_a else b).append(cell[j])
            to_a = not to_a
    return a, b


def _draw(pool: list[dict], n: int, rng: np.random.Generator, replace: bool = True) -> list[dict]:
    if not replace:
        return [pool[j] for j in rng.permutation(len(pool))[:n]]
    return [pool[j] for j in rng.integers(len(pool), size=n)]


def _completed(novel_item: dict, t: Traffic, rng: np.random.Generator) -> dict:
    """A novel complaint as the live system would log it: nearest-ticket intent/product, severity and sentiment drawn from the traffic's own mix."""
    out = dict(novel_item)
    for k, (vals, p) in t.marginals.items():
        out[k] = str(rng.choice(vals, p=p))
    return out


# Each builder returns (recent complaints, what was injected). Signature: (traffic, rng, pool for the recent window, window size)
def scenario_no_drift(t, rng, rec, n):
    return _draw(rec, n, rng), {}


def scenario_no_drift_distinct(t, rng, rec, n):
    """Control without repeated complaints: every request in the window is a different complaint."""
    return _draw(rec, n, rng, replace=False), {}


def scenario_intent_shift(share: float):
    def build(t, rng, rec, n):
        k = round(n * share)
        target = [i for i in rec if i["intent"] == BOOST_INTENT]
        rest = [i for i in rec if i["intent"] != BOOST_INTENT]
        return _draw(target, k, rng) + _draw(rest, n - k, rng), {"target": BOOST_INTENT, "share": share}
    return build


def scenario_severity_shift(share: float):
    def build(t, rng, rec, n):
        k = round(n * share)
        hi = [i for i in rec if i["severity"] in ("high", "critical")]
        rest = [i for i in rec if i["severity"] not in ("high", "critical")]
        return _draw(hi, k, rng) + _draw(rest, n - k, rng), {"target": "high/critical", "share": share}
    return build


def scenario_new_topic(count: int):
    def build(t, rng, rec, n):
        cls = NOVEL_CLASSES[int(rng.integers(len(NOVEL_CLASSES)))]
        extra = [_completed(x, t, rng) for x in _draw(t.novel[cls], count, rng, replace=False)]
        return _draw(rec, n - len(extra), rng) + extra, {"targets": [cls]}
    return build


def scenario_new_topics(per_class: int):
    def build(t, rng, rec, n):
        extra = [_completed(x, t, rng) for cls in NOVEL_CLASSES for x in _draw(t.novel[cls], per_class, rng, replace=False)]
        return _draw(rec, n - len(extra), rng) + extra, {"targets": list(NOVEL_CLASSES)}
    return build


def scenario_known_intent_returns(count: int):
    """Held-out check of the cluster-test weighting: an intent the corpus DOES explain is absent from the baseline and `count` of its complaints arrive.
    The weighting leans away from such a cluster, so this scenario shows its cost. No setting was chosen on it."""
    def build(t, rng, rec, n):
        target = RETURNING_INTENTS[int(rng.integers(len(RETURNING_INTENTS)))]
        extra = [{**x, "truth": target} for x in _draw([i for i in rec if i["intent"] == target], count, rng, replace=False)]
        return _draw([i for i in rec if i["intent"] != target], n - len(extra), rng) + extra, {"targets": [target], "exclude_from_baseline": target}
    return build


def scenario_vocabulary_shift(t, rng, rec, n):
    return [{**i, "vec": i["chat_vec"], "ev": i["chat_ev"], "text": chat_style(i["text"])} for i in _draw(rec, n, rng)], {}


def scenario_stale_kb(t, rng, rec, n):
    return [{**i, "ev": i["ev_stale"]} for i in _draw(rec, n, rng)], {}


def _signals(rep: dict) -> list[str]:
    return sorted({a["signal"] for a in rep["alerts"]})


def _recovered(rep: dict, rows_by_id: dict[str, dict], targets: list[str]) -> tuple[dict[str, dict], int]:
    """Which injected topics have a significant cluster dominated (>= 70%) by their complaints, and how many significant clusters match none."""
    hit, matched = {}, 0
    for c in rep["emerging_clusters"]:
        if not c["significant"]:
            continue
        cls, n = Counter(rows_by_id[i]["truth"] for i in c["request_ids"]).most_common(1)[0]
        if cls in targets and n / c["size"] >= 0.7:
            matched += 1
            if cls not in hit or c["size"] > hit[cls]["size"]:
                hit[cls] = {"size": c["size"], "purity": round(n / c["size"], 3), "p_value": c["p_value"], "keywords": c["keywords"][:4]}
    return hit, sum(1 for c in rep["emerging_clusters"] if c["significant"]) - matched


# detect(report, meta, recent window) -> (detected, detail)
def _detect_intent(rep, meta, win):
    top = max(rep["distributions"]["intent"]["categories"], key=lambda c: c["delta"])["label"]
    return "intent_distribution" in _signals(rep), {"correct_attribution": top == meta["target"]}


def _detect_severity(rep, meta, win):
    return "severity_distribution" in _signals(rep), {}


def _detect_topic(rep, meta, win):
    hit, spurious = _recovered(rep, {r["request_id"]: r for r in win.rows}, meta["targets"])
    return len(hit) >= 1, {"recovered": len(hit), "spurious": spurious, "clusters": hit}


def _detect_returning(rep, meta, win):
    """Either detector counts: a recovered cluster, or an intent-mix alert whose top mover is the returning intent (correct_attribution)."""
    hit, d = _detect_topic(rep, meta, win)
    top = max(rep["distributions"]["intent"]["categories"], key=lambda c: c["delta"])["label"]
    mix = "intent_distribution" in _signals(rep) and top == meta["targets"][0]
    return hit or mix, d | {"correct_attribution": mix}


def _detect_vocab(rep, meta, win):
    return bool(set(_signals(rep)) & {"embedding_centroid", "evidence_confidence", "emerging_cluster"}), {}


def _detect_stale(rep, meta, win):
    return bool(set(_signals(rep)) & {"evidence_confidence", "abstention_rate"}), {}


@dataclass
class Scenario:
    name: str
    description: str
    build: object
    detect: object
    n_baseline: int = N_BASELINE
    n_recent: int = N_RECENT


SCENARIOS = [
    Scenario("intent_shift_25", f"{BOOST_INTENT} rises from ~16% to 25% of recent traffic", scenario_intent_shift(0.25), _detect_intent),
    Scenario("intent_shift_33", f"{BOOST_INTENT} rises to 33%", scenario_intent_shift(0.33), _detect_intent),
    Scenario("intent_shift_40", f"{BOOST_INTENT} rises to 40%", scenario_intent_shift(0.40), _detect_intent),
    Scenario("intent_shift_25_large", f"{BOOST_INTENT} rises to 25%, with 4x the traffic (1,200 baseline / 400 recent requests)", scenario_intent_shift(0.25), _detect_intent,
             4 * N_BASELINE, 4 * N_RECENT),
    Scenario("severity_shift_40", "high/critical severity rises from ~30% to 40%", scenario_severity_shift(0.40), _detect_severity),
    Scenario("severity_shift_55", "high/critical severity rises to 55%", scenario_severity_shift(0.55), _detect_severity),
    Scenario("new_topic_4", "4 of 100 recent complaints (4%) are about a topic the corpus has never seen", scenario_new_topic(4), _detect_topic),
    Scenario("new_topic_6", "6 of 100 (6%) are a new topic", scenario_new_topic(6), _detect_topic),
    Scenario("new_topic_10", "10 of 100 (10%) are a new topic", scenario_new_topic(10), _detect_topic),
    Scenario("new_topics_3x8", "three new topics at once, 8 complaints each (24% of recent traffic); detected = at least one recovered", scenario_new_topics(8), _detect_topic),
    Scenario("known_intent_returns_10", "a known intent, absent from the baseline, returns with 10 of 100 complaints (held-out check of the weighting; detected = cluster or intent-mix alert)",
             scenario_known_intent_returns(10), _detect_returning),
    Scenario("vocabulary_shift", "every recent complaint arrives in chat-widget style (abbreviated, lower case)", scenario_vocabulary_shift, _detect_vocab),
    Scenario("stale_knowledge_base", "tickets for 3 of 9 intents vanish from the evidence corpus", scenario_stale_kb, _detect_stale),
]


def _trial(t: Traffic, seed: int, build, n_baseline: int, n_recent: int, distinct: bool = False):
    rng = np.random.default_rng(seed)
    a, b = _split(t, rng)
    base_items = _draw(a, n_baseline, rng, replace=not distinct)
    items, meta = build(t, rng, b, n_recent)
    if meta.get("exclude_from_baseline"):   # drawn separately so every other scenario keeps its random stream (and its recorded numbers)
        base_items = _draw([i for i in a if i["intent"] != meta["exclude_from_baseline"]], n_baseline, rng, replace=not distinct)
    return _window(f"b{seed}", base_items, t), _window(f"r{seed}", items, t), meta


def run_scenario(t: Traffic, sc: Scenario, trials: int, cfg: DriftConfig, seed0: int = 1000) -> dict:
    ok, fired, attrib, recovered, spurious = 0, Counter(), 0, 0, 0
    for s in range(trials):
        base, rec, meta = _trial(t, seed0 + s, sc.build, sc.n_baseline, sc.n_recent)
        rep = analyze(rec, base, cfg)
        hit, d = sc.detect(rep, meta, rec)
        ok += hit
        attrib += bool(d.get("correct_attribution"))
        recovered += d.get("recovered", 0)
        spurious += d.get("spurious", 0)
        fired.update(_signals(rep))
    out = {"description": sc.description, "n_baseline": sc.n_baseline, "n_recent": sc.n_recent, "trials": trials, "detected": ok,
           "detection_rate": round(ok / trials, 4), "ci95": wilson(ok, trials), "signals_fired": dict(fired.most_common())}
    if sc.name.startswith(("intent_shift", "known_intent")):
        out["correct_attribution"] = attrib
    if sc.name.startswith(("new_topic", "known_intent")):
        out["topics_recovered_per_report"] = round(recovered / trials, 3)
        out["unmatched_significant_clusters_per_report"] = round(spurious / trials, 3)
    return out


def run_control(t: Traffic, trials: int, cfg: DriftConfig, n_baseline: int = N_BASELINE, n_recent: int = N_RECENT, distinct: bool = False, seed0: int = 5000) -> dict:
    alarms, fired = 0, Counter()
    for s in range(trials):
        base, rec, _ = _trial(t, seed0 + s, scenario_no_drift_distinct if distinct else scenario_no_drift, n_baseline, n_recent, distinct)
        rep = analyze(rec, base, cfg)
        alarms += rep["status"] == "alert"
        fired.update(_signals(rep))
    return {"n_baseline": n_baseline, "n_recent": n_recent, "distinct_complaints": distinct, "trials": trials, "false_alarms": alarms, "false_alarm_rate": round(alarms / trials, 4),
            "ci95": wilson(alarms, trials), "signals_fired": dict(fired.most_common()), "design_target": cfg.alpha}


def illustrate(t: Traffic, cfg: DriftConfig, trials: int, seed0: int = 1000) -> dict:
    """The alerts a reviewer would read, taken from the FIRST trial of each scenario that detected the change (labelled, with how many trials it took);
    for new topics, also how the drift clusters line up with discovery proposals built from the same window. Detection rates are in the table above."""
    out = {}
    for sc in SCENARIOS:
        for s in range(trials):
            base, rec, meta = _trial(t, seed0 + s, sc.build, sc.n_baseline, sc.n_recent)
            rep = analyze(rec, base, cfg)
            if sc.detect(rep, meta, rec)[0]:
                entry = {"description": sc.description, "trial": s + 1, "seed": seed0 + s, "status": rep["status"], "alerts": [a["message"] for a in rep["alerts"]]}
                if sc.name == "new_topics_3x8":
                    entry["discovery_link"] = _link_to_discovery(t, rep, rec)
                out[sc.name] = entry
                break
        else:
            out[sc.name] = {"description": sc.description, "trial": None, "status": "not detected in any trial", "alerts": []}
    return out


def _proposals(t: Traffic, rec: Window) -> list[dict]:
    """The production proposal builder (discovery) on the recent window's low-evidence or abstained complaints, selected as the discovery job selects them."""
    cands = [i for i, r in enumerate(rec.emb_rows) if r["evidence"] < t.novelty_threshold or r["status"] == "abstained"]
    if not cands:
        return []
    X = rec.X[cands]
    neighbors = []
    for v in X:
        sims = t.corpus_X @ v
        top = np.argsort(-sims)[:5]
        neighbors.append(Neighbors([t.corpus_intent[j] for j in top], [t.corpus_product[j] for j in top], float(sims[top[0]])))
    labels = {i: None for i in sorted(set(t.corpus_intent))}
    return build_proposals([rec.emb_rows[i]["complaint"] for i in cands], X, neighbors, [rec.emb_rows[i]["evidence"] for i in cands], t.corpus_text,
                           DiscoveryParams(), labels, request_ids=[rec.emb_rows[i]["request_id"] for i in cands])


def run_small_topics(t: Traffic, cfg: DriftConfig, trials: int, counts: tuple[int, ...] = (4, 6), seed0: int = 1000) -> dict:
    """Small new topics: alarm versus review queue. The drift alarm has a 1% false-alarm budget, so a cluster of 4 to 6 complaints is largely below what
    it can certify; discovery has no such budget and puts proposals in front of a person instead. Measured on the SAME windows as the new_topic_*
    scenarios: a topic counts as surfaced when one proposal is at least 70% its complaints. The no-change control gives the review load (proposals per
    window with nothing injected), which is the price of that sensitivity."""
    out = {}
    for n in counts:
        sc = scenario_new_topic(n)
        alarm = surfaced = other = 0
        for s in range(trials):
            base, rec, meta = _trial(t, seed0 + s, sc, N_BASELINE, N_RECENT)
            rows = {r["request_id"]: r for r in rec.rows}
            alarm += _detect_topic(analyze(rec, base, cfg), meta, rec)[0]
            hit = False
            for p in _proposals(t, rec):
                cls, k = Counter(rows[i]["truth"] for i in p["member_request_ids"]).most_common(1)[0]
                if cls in meta["targets"] and k / len(p["member_request_ids"]) >= 0.7:
                    hit = True
                else:
                    other += 1
            surfaced += hit
        out[f"new_topic_{n}"] = {"injected": n, "trials": trials, "drift_alarm": alarm, "discovery_proposal": surfaced, "discovery_ci95": wilson(surfaced, trials),
                                 "other_proposals_per_window": round(other / trials, 2)}
    ctl = [len(_proposals(t, _trial(t, 5000 + s, scenario_no_drift, N_BASELINE, N_RECENT)[1])) for s in range(trials)]
    out["no_change"] = {"trials": trials, "proposals_per_window": round(float(np.mean(ctl)), 2), "windows_with_a_proposal": int(sum(c > 0 for c in ctl))}
    return out


def _link_to_discovery(t: Traffic, rep: dict, rec: Window) -> dict:
    """Run the production proposal builder on the recent low-evidence complaints and report how much of each drift cluster it covers."""
    props = _proposals(t, rec)
    links = []
    for c in rep["emerging_clusters"]:
        if not c["significant"]:
            continue
        ids = set(c["request_ids"])
        for p in props:
            ov = len(ids & set(p["member_request_ids"]))
            if ov:
                links.append({"cluster": c["cluster_id"], "cluster_size": c["size"], "cluster_keywords": c["keywords"][:3], "proposal_recommendation": p["recommendation"],
                              "proposal_label": p["label_id"], "proposal_keywords": p["keywords"][:4], "shared_complaints": ov, "share_of_cluster": round(ov / c["size"], 3)})
    return {"proposals_built": len(props), "links": links}


def run_demo(embedder, trials: int = 30, aa_trials: int = 100, data_dir: Path | None = None) -> dict:
    t0 = time.perf_counter()
    import scipy
    t = load_traffic(embedder, data_dir)
    cfg = DriftConfig(novelty_threshold=t.novelty_threshold)
    out = {
        "meta": {"embedding_model": embedder.model_name, "numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0],
                 "in_domain_complaints": len(t.items), "novel_classes": {k: len(v) for k, v in t.novel.items()}, "corpus_tickets": len(t.corpus_text),
                 "trials_per_scenario": trials,
                 "evidence_proxy": {"abstain_below": round(t.abstain_below, 4), "novelty_threshold": round(t.novelty_threshold, 4)}},
        "config": cfg.public(),
        "control": [run_control(t, aa_trials, cfg), run_control(t, aa_trials, cfg, 140, 70, distinct=True)],
        "scenarios": {sc.name: run_scenario(t, sc, trials, cfg) for sc in SCENARIOS},
        "small_topics": run_small_topics(t, cfg, trials),
    }
    out["examples"] = illustrate(t, cfg, trials)
    out["meta"]["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return out


def render_markdown(r: dict) -> str:
    m = r["meta"]
    lines = ["# Drift monitoring: deterministic injection demo", "",
             "Generated by `python -m app.evaluation.drift_demo`; see [drift.md](drift.md) for the method and its limits. "
             f"Embedding model `{m['embedding_model']}`; {m['in_domain_complaints']} labelled complaints as the traffic; {m['trials_per_scenario']} trials per scenario "
             f"with fixed seeds. Took {m['elapsed_s']} s.", "",
             "## False alarms (nothing changed)", "",
             "| Baseline / recent requests | Reports with an alert | Rate | 95% interval | Signals that fired |", "|---|---|---|---|---|"]
    for c in r["control"]:
        lines.append(f"| {c['n_baseline']} / {c['n_recent']}{' (no repeated complaints)' if c.get('distinct_complaints') else ''} | {c['false_alarms']} of {c['trials']} | {c['false_alarm_rate']:.1%} | {c['ci95'][0]:.1%} to {c['ci95'][1]:.1%} | {c['signals_fired'] or 'none'} |")
    lines += ["", f"Design target: at most {r['control'][0]['design_target']:.0%} of reports.", "",
              "## Detection of injected changes", "",
              "| Scenario | What changed in the recent window | Detected | 95% interval | Notes |", "|---|---|---|---|---|"]
    for name, s in r["scenarios"].items():
        note = []
        if "correct_attribution" in s:
            note.append(f"top mover correct in {s['correct_attribution']}/{s['trials']}")
        if "topics_recovered_per_report" in s:
            note.append(f"{s['topics_recovered_per_report']} injected topics recovered per report; {s['unmatched_significant_clusters_per_report']} unmatched significant clusters per report")
        fired = ", ".join(f"{k} ({v})" for k, v in list(s["signals_fired"].items())[:4])
        lines.append(f"| `{name}` | {s['description']} | {s['detected']}/{s['trials']} ({s['detection_rate']:.0%}) | {s['ci95'][0]:.0%} to {s['ci95'][1]:.0%} | {'; '.join(note + ([f'fired: {fired}'] if fired else []))} |")
    if r.get("small_topics"):
        st = r["small_topics"]
        lines += ["", "## Small new topics: alarm versus review queue", "",
                  "The same windows as `new_topic_4` and `new_topic_6`. The drift alarm has a 1% false-alarm budget, and a pure cluster of 4 recent complaints cannot "
                  "reach p < 1% at these window sizes even before any correction. Discovery (`build_proposals`, the production code) has no alarm budget: it puts "
                  "proposals in front of a person, so its sensitivity is paid for in review load (last column, and the control row).", "",
                  "| Injected topic | Drift alarm | Discovery proposal for the topic | 95% interval | Other proposals per window |", "|---|---|---|---|---|"]
        for k, v in st.items():
            if k.startswith("new_topic"):
                lines.append(f"| {v['injected']} of 100 complaints | {v['drift_alarm']}/{v['trials']} | {v['discovery_proposal']}/{v['trials']} | "
                             f"{v['discovery_ci95'][0]:.0%} to {v['discovery_ci95'][1]:.0%} | {v['other_proposals_per_window']} |")
        nc = st["no_change"]
        lines.append(f"| nothing (control) | - | - | - | {nc['proposals_per_window']} (a proposal in {nc['windows_with_a_proposal']} of {nc['trials']} windows) |")
    lines += ["", "## What the alerts say", "",
              "From the first trial of each scenario that detected the change (the table above gives how often that happens). The text is exactly what the API returns.", ""]
    for name, e in r["examples"].items():
        lines.append(f"**{name}**: {e['description']}. " + (f"Detected in trial {e['trial']} (seed {e['seed']})." if e["trial"] else "Not detected in any trial."))
        for a in e["alerts"]:
            lines.append(f"- {a}")
        if e.get("discovery_link"):
            d = e["discovery_link"]
            lines.append(f"- Discovery link: the proposal builder produced {d['proposals_built']} proposal(s) from the same window. " +
                         ("; ".join(f"cluster {l['cluster']} ({l['cluster_size']} complaints, {', '.join(l['cluster_keywords'])}) shares {l['shared_complaints']} complaints with a "
                                    f"`{l['proposal_recommendation']}` proposal `{l['proposal_label']}` ({', '.join(l['proposal_keywords'])})" for l in d["links"]) or "none overlapped a drift cluster"))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--aa-trials", type=int, default=200)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "eval" / "results" / "drift_demo.json"))
    ap.add_argument("--report", default=str(REPO_ROOT / "docs" / "drift_demo_results.md"))
    a = ap.parse_args()
    from app.services.embedding import EmbeddingService

    emb = EmbeddingService(Settings())
    emb.warmup()
    res = run_demo(emb, a.trials, a.aa_trials)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    Path(a.report).write_text(render_markdown(res), encoding="utf-8")
    print(render_markdown(res))
    print(f"\nwrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
