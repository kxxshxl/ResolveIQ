"""Pure feedback analytics and the improvement report. Same rows in, same report out (no clocks, no randomness, ties broken by name)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

MIN_N = 3          # fewer ratings than this for an intent or source: shown, never ranked or reported as a finding
Z = 1.96


def wilson_lower(k: int, n: int, z: float = Z) -> float:
    """Lower end of the 95% Wilson interval for a proportion: ranks a 3-of-3 below a 30-of-40 instead of above it."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d)


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _cited_ids(row: dict) -> list[str]:
    return [c["source_id"] for c in (row.get("citations") or []) if isinstance(c, dict) and "source_id" in c]


def analyze(feedback: list[dict], requests: list[dict], days: int) -> dict:
    """feedback: joined feedback + request rows; requests: every logged request in the window (status, intent, product, ...)."""
    n_req = len(requests)
    helpful = [f for f in feedback if f["rating"] == "helpful"]
    rejected = [f for f in feedback if f["rating"] == "not_helpful"]
    rated_requests = {f["request_id"] for f in feedback}
    totals = {"requests": n_req, "feedback": len(feedback), "requests_with_feedback": len(rated_requests), "coverage": _rate(len(rated_requests), n_req),
              "helpful": len(helpful), "not_helpful": len(rejected), "helpful_rate": _rate(len(helpful), len(feedback)), "rejection_rate": _rate(len(rejected), len(feedback)),
              "edited_by_agent": sum(1 for f in feedback if f.get("edited")), "with_corrected_intent": sum(1 for f in feedback if f.get("corrected_intent"))}

    # ---- by intent
    by_intent: dict[str, dict] = defaultdict(lambda: {"requests": 0, "feedback": 0, "helpful": 0, "not_helpful": 0, "abstained": 0, "degraded": 0, "unreliable": 0})
    for r in requests:
        e = by_intent[r.get("intent") or "unknown"]
        e["requests"] += 1
        for st in ("abstained", "degraded", "unreliable"):
            e[st] += r["status"] == st
    for f in feedback:
        e = by_intent[f.get("intent") or "unknown"]
        e["feedback"] += 1
        e["helpful" if f["rating"] == "helpful" else "not_helpful"] += 1
    intents = []
    for name, e in by_intent.items():
        intents.append({"intent": name, **e, "rejection_rate": _rate(e["not_helpful"], e["feedback"]), "rejection_lower_bound": round(wilson_lower(e["not_helpful"], e["feedback"]), 4),
                        "abstention_rate": _rate(e["abstained"], e["requests"])})
    intents.sort(key=lambda x: (-x["rejection_lower_bound"], -x["feedback"], x["intent"]))
    problem_intents = [i for i in intents if i["feedback"] >= MIN_N and i["not_helpful"] > 0][:8]

    # ---- sources: how often a source was cited in a rejected vs helpful resolution (cited = shown to the agent as support)
    src: dict[str, dict] = defaultdict(lambda: {"in_rejected": 0, "in_helpful": 0, "explicit_rejections": 0, "intents": Counter()})
    for f in feedback:
        for sid in set(_cited_ids(f)):
            s = src[sid]
            s["in_rejected" if f["rating"] == "not_helpful" else "in_helpful"] += 1
            if f.get("intent"):
                s["intents"][f["intent"]] += 1
        for sid in set(f.get("rejected_sources") or []):
            src[sid]["explicit_rejections"] += 1
    sources = []
    for sid, s in src.items():
        n = s["in_rejected"] + s["in_helpful"]
        sources.append({"source_id": sid, "type": "article" if sid.startswith("KB-") else "ticket", "cited_in_rejected": s["in_rejected"], "cited_in_helpful": s["in_helpful"],
                        "rejection_share": _rate(s["in_rejected"], n), "rejection_lower_bound": round(wilson_lower(s["in_rejected"], n), 4),
                        "explicit_rejections": s["explicit_rejections"], "intents": [i for i, _ in s["intents"].most_common(3)]})
    sources.sort(key=lambda x: (-x["explicit_rejections"], -x["rejection_lower_bound"], -x["cited_in_rejected"], x["source_id"]))
    rejected_sources = [s for s in sources if s["explicit_rejections"] > 0 or (s["cited_in_rejected"] >= 1 and s["cited_in_rejected"] + s["cited_in_helpful"] >= MIN_N)][:12]
    weak_articles = [s for s in rejected_sources if s["type"] == "article" and (s["explicit_rejections"] >= 2 or (s["cited_in_rejected"] + s["cited_in_helpful"] >= MIN_N and (s["rejection_share"] or 0) >= 0.5))]

    # ---- abstention
    ab = [r for r in requests if r["status"] == "abstained"]
    reasons = Counter((r.get("abstention_reason") or "unknown").split(":")[0] for r in ab)
    abstention = {"count": len(ab), "rate": _rate(len(ab), n_req), "by_reason": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
                  "by_intent": dict(Counter(r.get("intent") or "unknown" for r in ab).most_common(8)), "by_product": dict(Counter(r.get("product") or "unknown" for r in ab).most_common(6))}

    # ---- failure patterns
    patterns = []

    def add(name, count, detail):
        if count:
            patterns.append({"pattern": name, "count": count, "share": _rate(count, n_req), "detail": detail})

    val = [r.get("validation") or {} for r in requests]
    add("unreliable: citation or grounding checks failed", sum(r["status"] == "unreliable" for r in requests), "the answer was withheld from automation and escalated")
    add("invented citation ids removed", sum(bool(v.get("invalid_citations")) for v in val), "the model cited ids that were never retrieved")
    add("steps without a valid citation", sum(bool(v.get("uncited_steps")) for v in val), "the model produced steps it did not ground")
    add("steps not supported by the cited text", sum(bool(v.get("unsupported_steps")) for v in val), "cited, but the cited text does not back the step")
    fb_reasons = Counter(r.get("fallback_reason") or "unspecified" for r in requests if r["status"] == "degraded")
    add("degraded: evidence-only answer", sum(r["status"] == "degraded" for r in requests), "; ".join(f"{k} x{v}" for k, v in fb_reasons.most_common(3)))
    add("answered on weak evidence", sum(1 for r in requests if r["status"] in ("resolved", "degraded") and r.get("evidence") is not None and r["evidence"] < 0.65), "evidence confidence below 0.65")
    add("escalated", sum(1 for r in requests if r.get("escalated")), "the resolution recommends escalation")
    patterns.sort(key=lambda p: (-p["count"], p["pattern"]))
    reasons_tags = Counter(t for f in feedback for t in (f.get("reasons") or []))
    conf = Counter((f["intent"], f["corrected_intent"]) for f in feedback if f.get("corrected_intent") and f.get("intent") and f["corrected_intent"] != f["intent"])
    confusions = [{"predicted": a, "corrected": b, "count": c} for (a, b), c in sorted(conf.items(), key=lambda kv: (-kv[1], kv[0]))][:8]
    return {"window_days": days, "totals": totals, "by_intent": intents, "problem_intents": problem_intents, "rejected_sources": rejected_sources, "weak_articles": weak_articles,
            "abstention": abstention, "failure_patterns": patterns, "feedback_reasons": dict(reasons_tags.most_common()), "intent_confusions": confusions}


def findings(summary: dict) -> list[dict]:
    """Deterministic improvement findings. Each cites the numbers behind it; each needs a minimum sample (MIN_N) before it is raised."""
    t = summary["totals"]
    out: list[dict] = []

    def add(kind, severity, title, evidence, action):
        out.append({"kind": kind, "severity": severity, "title": title, "evidence": evidence, "suggested_action": action})

    if t["feedback"] < 10:
        add("data_sufficiency", "info", f"Only {t['feedback']} rating(s) in the last {summary['window_days']} days", {"feedback": t["feedback"], "requests": t["requests"]},
            "Treat everything below as a hint, not a conclusion; encourage agents to rate resolutions.")
    elif t["coverage"] is not None and t["coverage"] < 0.05:
        add("data_sufficiency", "info", f"Only {t['coverage']:.0%} of requests are rated", {"coverage": t["coverage"]}, "Ratings may not represent typical traffic.")
    for i in summary["problem_intents"]:
        if i["feedback"] >= 5 and (i["rejection_rate"] or 0) >= 0.4 and i["rejection_lower_bound"] >= 0.2:
            top = [s["source_id"] for s in summary["rejected_sources"] if i["intent"] in s["intents"]][:3]
            add("problem_intent", "high" if i["rejection_lower_bound"] >= 0.4 else "medium",
                f"Intent '{i['intent']}': {i['not_helpful']} of {i['feedback']} ratings are 'not helpful'",
                {"rejection_rate": i["rejection_rate"], "rejection_lower_bound": i["rejection_lower_bound"], "n": i["feedback"], "sources_often_cited": top},
                "Review the tickets and KB articles behind this intent for outdated or missing steps" + (f"; start with {', '.join(top)}." if top else "."))
    for s in summary["weak_articles"]:
        add("weak_article", "high" if s["explicit_rejections"] >= 3 else "medium", f"KB article {s['source_id']} is cited in rejected resolutions",
            {k: s[k] for k in ("cited_in_rejected", "cited_in_helpful", "explicit_rejections", "rejection_share")}, "Review, rewrite or deprecate the article (POST /api/v1/articles/{id}/deprecate).")
    for s in summary["rejected_sources"]:
        if s["type"] == "ticket" and (s["explicit_rejections"] >= 2 or (s["cited_in_rejected"] + s["cited_in_helpful"] >= MIN_N and (s["rejection_share"] or 0) >= 0.6)):
            add("rejected_ticket", "medium", f"Past ticket {s['source_id']} keeps being rejected", {k: s[k] for k in ("cited_in_rejected", "cited_in_helpful", "explicit_rejections")},
                "Check whether this resolution is still correct; archive it if not so it stops being retrieved.")
    for c in summary["intent_confusions"]:
        if c["count"] >= 2:
            add("intent_confusion", "medium", f"Agents corrected '{c['predicted']}' to '{c['corrected']}' {c['count']} times", c,
                f"Add keywords or examples to '{c['corrected']}' (or review the proposal workflow) so the classifier separates the two.")
    ab = summary["abstention"]
    for name, n in ab["by_intent"].items():
        row = next((i for i in summary["by_intent"] if i["intent"] == name), None)
        if row and n >= 5 and (row["abstention_rate"] or 0) >= 0.3:
            add("abstention_cluster", "medium", f"{n} abstentions for intent '{name}' ({row['abstention_rate']:.0%} of its requests)", {"abstained": n, "rate": row["abstention_rate"]},
                "The corpus does not cover these complaints well: add resolved tickets or an article, or run class discovery.")
    for p in summary["failure_patterns"]:
        if p["pattern"].startswith("unreliable") and (p["share"] or 0) >= 0.05:
            add("grounding_failures", "high", f"{p['share']:.0%} of answers failed the citation/grounding checks", p, "Inspect the cases (Case Replay), then the prompt or model.")
        if p["pattern"].startswith("degraded") and (p["share"] or 0) >= 0.10:
            add("llm_degradation", "high", f"{p['share']:.0%} of answers were evidence-only", p, "Check the LLM backend capacity and circuit breaker; users are getting the fallback.")
    order = {"high": 0, "medium": 1, "info": 2}
    out.sort(key=lambda f: (order[f["severity"]], f["kind"], f["title"]))
    return out


def to_markdown(summary: dict, items: list[dict]) -> str:
    t = summary["totals"]
    lines = [f"# Feedback improvement report (last {summary['window_days']} days)", "",
             f"{t['requests']} requests, {t['feedback']} ratings on {t['requests_with_feedback']} of them"
             + (f" ({t['coverage']:.1%} coverage)" if t["coverage"] is not None else "") + ".",
             f"Helpful {t['helpful']}, not helpful {t['not_helpful']}" + (f" (rejection rate {t['rejection_rate']:.0%})." if t["rejection_rate"] is not None else "."), ""]
    if not items:
        lines.append("No findings: nothing crossed the minimum-evidence thresholds.")
    for sev, name in (("high", "High priority"), ("medium", "Medium"), ("info", "Notes")):
        group = [f for f in items if f["severity"] == sev]
        if group:
            lines += [f"## {name}", ""]
            for f in group:
                lines += [f"- **{f['title']}**", f"  - evidence: {', '.join(f'{k}={v}' for k, v in f['evidence'].items())}", f"  - suggested action: {f['suggested_action']}"]
            lines.append("")
    lines += ["_Advisory only: this report does not retrain any model or change any data. Findings require at least " + str(MIN_N) + " ratings per item._"]
    return "\n".join(lines)
