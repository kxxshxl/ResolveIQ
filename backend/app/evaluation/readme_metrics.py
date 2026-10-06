"""Render the README's and docs/evaluation.md's result tables from the recorded result files, so that no number in them is typed by hand.

    python scripts/render_readme_metrics.py            # rewrite the generated blocks
    python scripts/render_readme_metrics.py --check    # exit 1 when a block is out of date (the test suite runs this)

A block lives between ``<!-- METRICS:START ... -->`` and ``<!-- METRICS:END -->``. Only values present in the files are rendered; a missing
file or key renders as "not measured" and never as a made-up number. Nothing here reads the clock, so the output depends only on the files.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import REPO_ROOT
from app.evaluation.summary import load_results

START = "<!-- METRICS:START"
END = "<!-- METRICS:END -->"
TARGETS = {"README.md": "readme", "docs/evaluation.md": "evaluation"}

STRATEGY_NAMES = {"lexical": "keyword (Postgres FTS)", "bm25": "keyword (BM25, in memory)", "dense": "dense (pgvector)", "hybrid": "hybrid (RRF)",
                  "hybrid_reranked": "hybrid + cross-encoder", "adaptive": "adaptive"}


def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pc(x, nd=0):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def ms(x):
    return "n/a" if x is None else (f"{x / 1000:.1f} s" if x >= 10000 else f"{x:,.0f} ms" if x >= 100 else f"{x:.1f} ms")


def table(head: list[str], rows: list[list[str]], right: int = 1) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * right + ["---:"] * (len(head) - right)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def _bold_best(values: list[float | None], i: int, text: str) -> str:
    best = max((v for v in values if v is not None), default=None)
    return f"**{text}**" if best is not None and values[i] == best else text


# ------------------------------------------------------------------ sections
def run_info(res: dict) -> list[str]:
    m = g(res, "latest", "meta", default={})
    if not m:
        return ["No evaluation run is recorded (`data/eval/results/latest.json` is missing): run `python -m app.evaluation.run`.", ""]
    c = m.get("corpus", {})
    sets = m.get("eval_sets", {})
    return [f"Recorded on the final code (suites were run on 2026-10-06; the newest write is `{m.get('timestamp', 'unknown')}`): {c.get('tickets')} tickets, {c.get('articles')} active articles (+ {c.get('deprecated_articles')} retired); embedding `{m.get('embedding_model')}`, reranker `{m.get('reranker')}`, "
            f"LLM `{', '.join(m.get('llm', []))}`. Query sets: " + ", ".join(f"{k} {v}" for k, v in sets.items()) + ".", ""]


def retrieval(res: dict, detail: bool) -> list[str]:
    ad = g(res, "latest", "adaptive", "results")
    out: list[str] = []
    if not ad:
        return ["Retrieval comparison: not measured in the recorded run.", ""]
    splits = [("test", "held-out paraphrases (templated), tickets"), ("gold", "hand-written gold, tickets"), ("blind", "hand-written blind, tickets")]
    for split, title in splits if detail else splits[:1] + splits[1:2]:
        block = g(ad, split, "ticket")
        if not block:
            continue
        n = g(block, "dense", "n")
        out.append(f"**{title}** (n={n})")
        out.append("")
        order = [s for s in ("lexical", "bm25", "dense", "hybrid", "hybrid_reranked", "adaptive") if s in block]
        cols = {k: [g(block, s, k) for s in order] for k in ("hit@1", "hit@5", "mrr", "ndcg@10")}
        rows = []
        for i, s in enumerate(order):
            rows.append([STRATEGY_NAMES[s]] + [_bold_best(cols[k], i, f(cols[k][i])) for k in cols] + [ms(g(block, s, "latency", "p50_ms")), ms(g(block, s, "latency", "p95_ms"))])
        out += table(["strategy", "Hit@1", "Hit@5", "MRR", "nDCG@10", "p50", "p95"], rows)
    out += ["Quality columns are deterministic and reproduce exactly between runs. Latencies are single-process timings on a shared laptop: the p50s are stable, "
            "but a p95 of a few milliseconds moves several-fold between runs with whatever else the machine is doing, so compare strategies by p50.", ""]
    art = g(ad, "gold", "article")
    if detail and art:
        out.append(f"**KB articles, hand-written gold** (n={g(art, 'dense', 'n')})")
        out.append("")
        order = [s for s in ("lexical", "bm25", "dense", "hybrid", "hybrid_reranked", "adaptive") if s in art]
        cols = {k: [g(art, s, k) for s in order] for k in ("hit@1", "hit@3", "mrr")}
        out += table(["strategy", "Hit@1", "Hit@3", "MRR"], [[STRATEGY_NAMES[s]] + [_bold_best(cols[k], i, f(cols[k][i])) for k in cols] for i, s in enumerate(order)])
    return out


def adaptive(res: dict, detail: bool) -> list[str]:
    ad = g(res, "latest", "adaptive")
    if not ad:
        return []
    cfg = ad.get("config", {})
    out = [f"**Adaptive retrieval** (margins: tickets {cfg.get('margin_ticket')}, articles {cfg.get('margin_article')}; rerank rung {'on' if cfg.get('rerank_gap') else 'off'}, MMR {'on' if cfg.get('mmr') else 'off'}; "
           f"chosen on {cfg.get('tuned_on')}, judged on {', '.join(cfg.get('held_out', []))}). Share of queries that stopped at each rung, and the paired-bootstrap difference to dense retrieval (MRR, 95% interval):", ""]
    rows = []
    for split in ("gold", "blind"):
        for kind in ("ticket", "article"):
            lad = g(ad, "ladder", split, kind)
            diff = g(ad, "paired_differences", split, kind, "adaptive_minus_dense", "mrr")
            dis = g(ad, "disagreements", split, kind)
            if not lad:
                continue
            rows.append([f"{split} {kind}s", pc(lad.get("stopped_at_dense")), pc(lad.get("stopped_at_hybrid")), pc(lad.get("stopped_at_rerank")),
                         (f"{diff['mean_diff']:+.3f} ({diff['ci95'][0]:+.3f} to {diff['ci95'][1]:+.3f})" if diff and "mean_diff" in diff else "n/a"),
                         (f"{dis['adaptive_better']} / {dis['adaptive_worse']}" if dis else "n/a")])
    out += table(["split", "stopped at dense", "at hybrid", "at rerank", "adaptive minus dense", "queries adaptive better / worse"], rows)
    return out


def classification(res: dict, detail: bool) -> list[str]:
    strat = g(res, "latest", "classification", "strategies")
    if not strat:
        return ["Classification: not measured in the recorded run.", ""]
    out = ["**Classification accuracy** (`gold` = hand-written, `blind` = hand-written after the affect model was frozen; `before` = rules + kNN for every dimension):", ""]
    rows = []
    for split in ("test", "gold", "blind") if detail else ("gold", "blind"):
        for key, name in (("ensemble_legacy", "before"), ("ensemble", "shipped")):
            e = g(strat, key, split)
            if e:
                rows.append([f"{split} ({g(e, 'intent', 'n')})", name] + [f(g(e, d, "accuracy")) for d in ("intent", "product", "severity", "sentiment")])
    out += table(["split", "classifier", "intent", "product", "severity", "sentiment"], rows, right=2)
    if detail:
        out.append("The `test` rows are templated text that shares its vocabulary with the generator, so the `before` severity and sentiment scores there are circular; the hand-written splits are the honest ones.")
        out.append("")
    return out


def robustness(res: dict) -> list[str]:
    p = g(res, "latest", "robustness", "perturbations")
    if not p:
        return []
    rows = [[k.replace("_", " "), f(v.get("intent")), f(v.get("severity")), f(v.get("sentiment")), f(v.get("ticket_hit@1")), f(v.get("ticket_hit@5")), f(v.get("article_hit@3"))] for k, v in p.items()]
    return [f"**Robustness** ({g(res, 'latest', 'robustness', 'n')} hand-written queries under deterministic perturbations):", ""] + table(
        ["perturbation", "intent", "severity", "sentiment", "ticket Hit@1", "ticket Hit@5", "article Hit@3"], rows)


def rag(res: dict) -> list[str]:
    r = g(res, "latest", "rag")
    e = g(res, "latest", "e2e")
    out: list[str] = []
    if r and "primary" in r:
        p, x = r["primary"], r.get("extractive_no_llm", {})
        out += [f"**Grounded generation** ({p.get('generator')}, {p.get('answered')} answers; evidence-only baseline {x.get('answered')} answers):", ""]
        rows = [["step faithfulness", f(p.get("step_faithfulness"), 4), f(x.get("step_faithfulness"), 4)], ["response hallucination rate", pc(p.get("response_hallucination_rate"), 1), pc(x.get("response_hallucination_rate"), 1)],
                ["citation validity", f(p.get("citation_validity")), f(x.get("citation_validity"))], ["citation coverage", f(p.get("citation_coverage")), f(x.get("citation_coverage"))],
                ["citation precision vs gold scenario", f(p.get("citation_precision_vs_gold")), f(x.get("citation_precision_vs_gold"))], ["gold-step recall", f(p.get("gold_step_recall")), f(x.get("gold_step_recall"))]]
        j = r.get("llm_judge")
        if j:
            rows.append([f"LLM judge, {j.get('n')} answers (1 to 5; a {j.get('judge_model')} self-judge)", f"faithfulness {j.get('faithfulness_1to5')}, relevance {j.get('relevance_1to5')}", "n/a"])
        out += table(["metric", "LLM", "evidence only"], rows)
    if e and "in_domain" in e:
        i, o = e["in_domain"], e.get("out_of_domain", {})
        lat = e.get("latency_total", {})
        out += [f"**End to end** ({i.get('n')} in-domain + {o.get('n')} out-of-domain complaints through the full pipeline):", ""]
        out += table(["metric", "value"], [["resolved", pc(i.get("successful_resolution_rate"))], ["resolved with a source from the right scenario", pc(i.get("correct_resolution_rate"))],
                                           ["false abstention (in-domain)", pc(i.get("false_abstention_rate"), 1)], ["out-of-domain complaints abstained", pc(o.get("correct_abstention_rate"))],
                                           ["pipeline errors", str(g(e, "failures", "pipeline_errors"))], ["latency p50 / p95 (LLM generation dominates)", f"{ms(lat.get('p50_ms'))} / {ms(lat.get('p95_ms'))}"]]
                     + ([[f"LLM tokens per generated answer, prompt / completion (mean; p95), {tk['n']} answers",
                          f"{tk['prompt']['mean']:,.0f} / {tk['completion']['mean']:,.0f} ({tk['prompt']['p95']:,.0f} / {tk['completion']['p95']:,.0f})"]]
                        if (tk := e.get("tokens")) and tk.get("prompt") and tk.get("completion") else []))
    return out or ["Generation and end-to-end: not measured in the recorded run.", ""]


def clustering(res: dict, detail: bool) -> list[str]:
    c = g(res, "latest", "clustering")
    if not c:
        return []
    sweep = c.get("sweep", [])
    out = [f"**Recurring-complaint clusters** ({c.get('complaints')} labelled complaints from {c.get('scenarios')} root causes in {c.get('intents')} intents; average linkage on the stored query embeddings). "
           "Purity is the share of complaints in a cluster that agree with the cluster's majority label; a root cause counts as recovered when one cluster is at least 70% that cause and holds at least half of its complaints. "
           "The application uses distance 0.55.", ""]
    rows = []
    for e in sweep:
        v, i = e["vs_scenario"], e["vs_intent"]
        rows.append([f"{e['distance']}", str(e["clusters"]), pc(e["coverage"]), f(e["mean_cluster_size"], 1), f(i.get("purity")), f(v.get("purity")), f(v.get("ari")), f"{v.get('recovered')} of {v.get('of')}"])
    out += table(["cosine distance", "clusters", "complaints clustered", "mean size", "purity vs intent", "purity vs root cause", "ARI vs root cause", "root causes recovered"], rows)
    u = c.get("with_unseen_topics")
    if u:
        out += [f"With {u.get('novel_complaints')} complaints from {u.get('of')} never-seen classes mixed in (distance 0.55): {u.get('recovered')} of {u.get('of')} of those classes came out as their own cluster.", ""]
    return out


def evolving_and_discovery(res: dict, detail: bool) -> list[str]:
    ev = g(res, "latest", "evolving")
    dc = g(res, "latest", "discovery")
    out = []
    if ev:
        b, a = ev.get("before_ingestion", {}), ev.get("after_ingestion", {})
        out.append(f"**Evolving data** ({a.get('n')} queries about 2 new intents; no restart): Hit@5 {f(b.get('hit@5'))} before ingestion, {f(a.get('hit@5'))} after ingesting 2 articles and a batch of 12 tickets in "
                   f"{g(ev, 'timings', 'ingest_2_articles_and_12_tickets_s')} s.")
        out.append("")
    if dc:
        cs, pr, af = dc.get("candidate_selection", {}), dc.get("proposals", {}), dc.get("after_acceptance", {})
        out.append(f"**Emerging-class discovery** ({g(dc, 'stream', 'novel')} complaints from 3 never-seen classes among {g(dc, 'stream', 'known')} known): the abstention gate alone catches {pc(cs.get('abstention_only_novel_recall'))} of the novel complaints, "
                   f"the candidate filter {pc(cs.get('novel_recall'))}; clustering recovered {pr.get('recovered_classes')} of {pr.get('of')} classes, proposal precision {f(pr.get('precision'))}; after a person accepts the proposals, held-out complaints are routed "
                   f"correctly {pc(af.get('routed_correctly_before'))} → {pc(af.get('routed_correctly_after'))} with no resolved tickets for those classes.")
        out.append("")
    return out


def drift(res: dict, detail: bool) -> list[str]:
    d = res.get("drift_demo")
    if not d:
        return []
    ctrl = d.get("control", [])
    sc = d.get("scenarios", {})
    out = [f"**Drift detection** (injection demo, {g(d, 'meta', 'trials_per_scenario')} trials per scenario, no database, deterministic seeds; `python -m app.evaluation.drift_demo`):", ""]
    rows = [[f"no change, {c['n_baseline']} / {c['n_recent']} requests", f"{c['false_alarms']} of {c['trials']} reports", f"{pc(c['false_alarm_rate'], 1)} (95% {pc(c['ci95'][0], 1)} to {pc(c['ci95'][1], 1)})"] for c in ctrl]
    keys = ["intent_shift_25", "intent_shift_33", "intent_shift_40", "severity_shift_40", "severity_shift_55", "new_topic_4", "new_topic_6", "new_topic_10", "new_topics_3x8",
            "known_intent_returns_10", "vocabulary_shift", "stale_knowledge_base"]
    for k in keys if detail else ["intent_shift_33", "intent_shift_40", "new_topic_6", "new_topic_10", "new_topics_3x8", "vocabulary_shift"]:
        v = sc.get(k)
        if v:
            rows.append([v["description"], f"{v['detected']} of {v['trials']} reports", f"{pc(v['detection_rate'])} (95% {pc(v['ci95'][0])} to {pc(v['ci95'][1])})"])
    out += table(["scenario", "alerts", "rate"], rows)
    st = d.get("small_topics")
    if st and st.get("no_change"):
        rows = [[f"{v['injected']} of 100 complaints are a new topic", f"{v['drift_alarm']} of {v['trials']}", f"{v['discovery_proposal']} of {v['trials']} ({pc(v['discovery_ci95'][0])} to {pc(v['discovery_ci95'][1])})",
                 f"{v['other_proposals_per_window']}"] for k, v in st.items() if k.startswith("new_topic")]
        nc = st["no_change"]
        rows.append(["nothing changed", "-", "-", f"{nc['proposals_per_window']}"])
        out += ["**Small new topics: alarm versus review queue** (same windows). A pure cluster of 4 recent complaints cannot reach p < 1% at this window size, so the alarm, "
                "with its 1% false-alarm budget, cannot certify it; discovery has no alarm budget and puts a proposal in front of a person instead, at the cost of review load:", ""]
        out += table(["scenario", "drift alarm", "discovery proposal for the topic (95%)", "other proposals per window"], rows)
    return out


def scale(res: dict, detail: bool) -> list[str]:
    s = res.get("pgvector_scale")
    out: list[str] = []
    if s:
        st, ix, sto, se = s["setup"], s["index_build"], s["storage"], s["search"]
        ef = "ef_search=100"      # the application's setting (HNSW_EF_SEARCH)
        row = se.get(ef) or {}
        out += [f"**pgvector at scale (SYNTHETIC vectors, not a production benchmark):** {st['vectors']:,} x {st['dimensions']} vectors, HNSW m={st['hnsw']['m']}, ef_construction={st['hnsw']['ef_construction']}: build {ix['seconds']} s "
                f"({'serial' if not ix['parallel_workers'] else str(ix['parallel_workers']) + ' workers'}), {sto['total_mb']:.0f} MB on disk ({sto['hnsw_index_mb']:.0f} MB index); at {ef.replace('=', ' ')}: p50 {row.get('p50_ms')} ms, "
                f"p95 {row.get('p95_ms')} ms, p99 {row.get('p99_ms')} ms, recall@10 {row.get('recall@10')} versus exact search. {', '.join(s.get('caveats', [])[:2])}.", ""]
    ib = res.get("ingest_benchmark")
    if ib:
        a, b = ib.get("per_ticket_transactions", {}), ib.get("one_batched_transaction", {})
        out += [f"**Ingestion at the database layer** ({ib.get('tickets_per_path')} tickets with random embeddings, so the embedding model is excluded): {a.get('tickets_per_second')} tickets/s with one transaction per ticket versus "
                f"{b.get('tickets_per_second')} tickets/s in one batched transaction ({ib.get('speedup')}x); re-running the same batch updates in place ({g(ib, 'idempotent_rerun', 'updated')} updated, {g(ib, 'idempotent_rerun', 'created')} created).", ""]
    return out


def load(res: dict, detail: bool) -> list[str]:
    l = res.get("load_test")
    if not l:
        return ["**Load test:** no summary is recorded (`loadtest/results/latest_summary.json`).", ""]
    names = {"no_llm": "no LLM (evidence-only mode)", "llm": "full pipeline, local LLM", "llm_down": "LLM unreachable"}
    rows = [[names.get(r["scenario"], r["scenario"]), str(r["users"]), f"{r['rps']}", ms(r.get("p50")), ms(r.get("p95")), ms(r.get("p99")), str(r.get("errors")), str(r.get("evidence_only", "n/a"))] for r in l.get("rows", [])]
    dirty = "; the working tree had uncommitted changes outside the request path" if l.get("dirty") else ""
    rec = f", recorded as `{l['commit_recorded']}` before the history was rewritten" if l.get("commit_recorded") else ""
    return [f"**Load test** (suite `{l.get('suite')}`, commit `{l.get('commit')}`{rec}{dirty}; the request path has not changed since): {l.get('description')}", ""] + table(
        ["scenario", "users", "OK req/s", "p50", "p95", "p99", "errors", "answered from evidence only"], rows) + [l.get("caveat", ""), ""]


def failures(res: dict, detail: bool) -> list[str]:
    if not detail:
        return []
    fa = res.get("failure_analysis")
    e2e = g(res, "latest", "e2e", "failure_cases")
    out: list[str] = []
    if fa:
        r, c, q = fa["retrieval"], fa["classification"], fa["queries"]
        n = q["gold"] + q["blind"]
        wrong = sum(r["top1_wrong"].values())
        out += [f"**Retrieval misses** (dense, {n} hand-written complaints): the top-1 ticket is from another root cause for {wrong} ({r['top1_wrong'].get('gold', 0)} gold, {r['top1_wrong'].get('blind', 0)} blind); "
                f"for {r['top1_wrong_but_same_intent']} of them it is a ticket of the same intent, i.e. an adjacent root cause. Most frequent confusions:", ""]
        out += table(["true root cause", "retrieved root cause", "queries"], [[x["true"], x["predicted"], str(x["count"])] for x in r["confusions_true_to_retrieved_scenario"][:5]], right=2)
        rows = [[d, str(c[d]["errors"]), "; ".join(f"{x['true']} → {x['predicted']} ({x['count']})" for x in c[d]["confusions"][:3])] for d in ("intent", "product", "severity", "sentiment")]
        out += [f"**Classifier errors** on the same {n} complaints (shipped ensemble):", ""] + table(["dimension", "errors", "most frequent confusions (true → predicted)"], rows, right=2)
    if e2e is not None:
        out += [f"**End-to-end failures** ({len(e2e)} of the in-domain complaints did not end in a resolution that cites the right root cause):", ""]
        rows = []
        for x in e2e[:14]:
            what = x["status"] if x["status"] != "resolved" else "resolved, wrong source"
            rows.append([str(x.get("qid")), what, x.get("true_scenario") or "", ", ".join(x.get("cited_scenarios", [])) or "none", "yes" if x.get("true_scenario_in_top3") else "no", f(x.get("evidence_confidence"))])
        out += table(["query", "outcome", "true root cause", "cited root cause(s)", "true cause in top-3 tickets", "evidence confidence"], rows, right=3)
    return out


def headline(res: dict) -> list[str]:
    ad = g(res, "latest", "adaptive", "results", "gold", "ticket", default={})
    cl = g(res, "latest", "classification", "strategies", default={})
    e2e = g(res, "latest", "e2e", default={})
    rag_p = g(res, "latest", "rag", "primary", default={})
    sc = g(res, "pgvector_scale", default={})
    ds = g(res, "drift_demo", "scenarios", default={})
    ctrl = g(res, "drift_demo", "control", default=[])
    ld = {(r["scenario"], r["users"]): r for r in g(res, "load_test", "rows", default=[])}
    out = []
    if ad.get("dense"):
        out.append(f"* **Semantic retrieval beats keyword search** on hand-written complaints: Hit@1 {f(ad['dense']['hit@1'])} (dense) versus {f(ad['lexical']['hit@1'])} (Postgres full-text) and {f(ad['bm25']['hit@1'])} (BM25).")
    b, a = g(cl, "ensemble_legacy", "blind"), g(cl, "ensemble", "blind")
    if a and b:
        out.append(f"* **Severity and sentiment** (blind hand-written set): accuracy {f(g(b, 'severity', 'accuracy'))} to {f(g(a, 'severity', 'accuracy'))} and {f(g(b, 'sentiment', 'accuracy'))} to {f(g(a, 'sentiment', 'accuracy'))} with the NLI cue model; intent {f(g(a, 'intent', 'accuracy'))}.")
    if e2e.get("in_domain"):
        out.append(f"* **End to end** ({e2e['in_domain']['n']} in-domain, {e2e['out_of_domain']['n']} out-of-domain complaints): {pc(e2e['in_domain']['correct_resolution_rate'])} resolved with a source from the right root cause, "
                   f"{pc(e2e['out_of_domain']['correct_abstention_rate'])} of out-of-domain complaints abstained, median latency {ms(g(e2e, 'latency_total', 'p50_ms'))}.")
    if rag_p:
        out.append(f"* **Grounding:** citation validity {f(rag_p.get('citation_validity'))}, step faithfulness {f(rag_p.get('step_faithfulness'), 3)} (lexical-containment proxy), citation precision against the right root cause {f(rag_p.get('citation_precision_vs_gold'))} (evidence-only baseline {f(g(res, 'latest', 'rag', 'extractive_no_llm', 'citation_precision_vs_gold'))}).")
    d = g(res, "latest", "adaptive", "paired_differences", "gold", "ticket", "adaptive_minus_dense", "mrr", "mean_diff")
    if d is not None:
        out.append(f"* **Adaptive retrieval matches dense retrieval, it does not beat it** (MRR difference {d:+.3f} on gold tickets, interval includes zero on every held-out split). It is shipped as a bounded, observable option.")
    if ctrl and ds.get("intent_shift_40") and ds.get("new_topic_10"):
        i40, n10, n4, n6 = ds["intent_shift_40"], ds["new_topic_10"], ds.get("new_topic_4", {}), ds.get("new_topic_6", {})
        st4, st6 = g(res, "drift_demo", "small_topics", "new_topic_4", default={}), g(res, "drift_demo", "small_topics", "new_topic_6", default={})
        small = (f"; for 4 and 6 complaints, below what a 1% alarm can certify, discovery puts a proposal for the topic in front of a reviewer in {st4['discovery_proposal']} and {st6['discovery_proposal']} of {st6['trials']}"
                 if st4 and st6 else "")
        out.append(f"* **Drift detection** (injection demo): {pc(ctrl[0]['false_alarm_rate'], 1)} false alarms with no change; \"{i40['description']}\" found in {i40['detected']} of {i40['trials']} trials; "
                   f"a new topic with 10 of 100 recent complaints in {n10['detected']} of {n10['trials']}, with 4 and 6 complaints in {n4.get('detected')} and {n6.get('detected')} of {n4.get('trials')}{small} (limits in docs/drift.md).")
    e = sc.get("search", {}).get("ef_search=100")
    if e:
        out.append(f"* **Database:** 100,000 synthetic 384-d vectors, HNSW: recall@10 {f(e['recall@10'], 3)}, p99 {e['p99_ms']} ms, {sc['storage']['total_mb']:.0f} MB; batched ingestion {g(res, 'ingest_benchmark', 'speedup')}x faster than per-row transactions.")
    c1, c25, l1 = ld.get(("no_llm", 1)), ld.get(("no_llm", 25)), ld.get(("llm", 1))
    if c1 and c25 and l1:
        out.append(f"* **Load (one API process, one laptop):** {c1['rps']} req/s without the LLM at 1 user, {c25['rps']} at 25 users (CPU-bound; errors {c25['errors']}); with the LLM one user gets full answers at {l1['rps']} req/s, and extra concurrent users are served from evidence only rather than queued.")
    return (["**At a glance**", ""] + out + [""]) if out else []


def render(kind: str, res: dict | None = None) -> str:
    res = res if res is not None else load_results()
    detail = kind == "evaluation"
    parts = [run_info(res)] + ([headline(res)] if not detail else []) + [retrieval(res, detail), adaptive(res, detail), classification(res, detail)]
    if detail:
        parts.append(robustness(res))
    parts += [rag(res), clustering(res, detail), evolving_and_discovery(res, detail), drift(res, detail), failures(res, detail), scale(res, detail), load(res, detail)]
    return "\n".join(line for part in parts for line in part).rstrip() + "\n"


def _replace(text: str, block: str) -> str:
    s = text.index(START)
    line_end = text.index("\n", s) + 1
    e = text.index(END, line_end)
    return text[:line_end] + "\n" + block + "\n" + text[e:]


def check_or_write(write: bool, root: Path | None = None, res: dict | None = None) -> list[str]:
    """Returns the files whose generated block differs from what the result files say (and rewrites them when write=True)."""
    root = root or REPO_ROOT
    stale = []
    for rel, kind in TARGETS.items():
        p = root / rel
        text = p.read_text(encoding="utf-8")
        if START not in text:
            continue
        new = _replace(text, render(kind, res))
        if new != text:
            stale.append(rel)
            if write:
                p.write_text(new, encoding="utf-8", newline="\n")
    return stale
