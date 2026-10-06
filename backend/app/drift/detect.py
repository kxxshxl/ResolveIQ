"""Drift detection: compare a recent window of logged requests with a baseline window and explain what moved.

Pure functions (no database, no model): the service feeds rows and embeddings in, the evaluation feeds synthetic windows in.
What is compared, and how (docs/drift.md has the reasoning and the measured false-alarm rate):

  category mix   intent / product / severity / sentiment      chi-square homogeneity test and a test per category (Bonferroni inside the
                                                              dimension); PSI as the effect size; per-category breakdown for explanation
  quality        evidence confidence                          Kolmogorov-Smirnov test (D is the effect size)
                 abstention rate                              two-proportion test plus a minimum absolute rise
  embeddings     where complaints sit in vector space         permutation test on the shift of the window centroid
  new topics     groups of similar complaints that recent       recent and baseline complaints are clustered TOGETHER without looking at
                 traffic over-represents                        which window they came from; each cluster is then tested with an exact
                                                                hypergeometric test for "more recent members than chance allows"

The p-values of the first three families are combined with the Holm-Bonferroni procedure at one family-wise level (`alpha`), and an alert also
needs the effect to exceed a minimum size, so a handful of requests cannot raise one and a million requests cannot raise one over a trivial
difference. New-topic clusters have their own budget of the same size (Bonferroni over the clusters tested, weighted towards clusters the corpus does not explain).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from app.discovery.clustering import DiscoveryParams, cluster_embeddings, keywords_per_cluster
from app.drift import stats

CATEGORICAL = ("intent", "product", "severity", "sentiment")
WATCH_P = 0.2   # clusters this close to significant are listed (not alerted on) so a reviewer can see near misses


@dataclass
class DriftConfig:
    min_requests: int = 30               # a window smaller than this is reported, never alerted on
    alpha: float = 0.01                  # family-wise false-alarm budget per report, for each of the two families (tests, new-topic clusters)
    psi_threshold: float = 0.10          # category mix: minimum Population Stability Index (conventional "moderate shift")
    ks_threshold: float = 0.15           # evidence confidence: minimum KS distance
    abstention_increase: float = 0.10    # minimum absolute rise in abstention rate
    embedding_shift: float = 0.02        # minimum cosine distance between window centroids
    cluster_enrichment: float = 2.0      # a cluster's share of recent complaints must be at least this multiple of the recent window's share overall
    new_topic_baseline_share: float = 0.15   # a cluster with at most this share of baseline members is a NEW topic; more than that, a surge of a known one
    unseen_quantile: float = 0.05        # descriptive only: "unseen" = farther from every baseline complaint than 95% of baseline complaints are from each other
    min_embedded: int = 30               # embedding tests need at least this many distinct complaints per window
    permutations: int = 500
    max_clusters: int = 10
    novelty_threshold: float = 0.80      # mean evidence below this: the corpus does not explain the cluster
    unexplained_weight: float = 0.75     # share of the cluster test budget given to clusters the corpus does not explain (0 = equal split; see _cluster_weights)
    cluster: DiscoveryParams = field(default_factory=lambda: DiscoveryParams(distance_threshold=0.65, min_cluster_size=4))
    seed: int = 17

    def public(self) -> dict:
        keys = ("min_requests", "alpha", "psi_threshold", "ks_threshold", "abstention_increase", "embedding_shift", "cluster_enrichment",
                "new_topic_baseline_share", "unseen_quantile", "min_embedded", "permutations", "novelty_threshold", "unexplained_weight")
        return {k: getattr(self, k) for k in keys} | {"cluster_distance_threshold": self.cluster.distance_threshold, "cluster_min_size": self.cluster.min_cluster_size}


@dataclass
class Window:
    rows: list[dict]                                      # every request in the window: status, intent, product, severity, sentiment, evidence
    emb_rows: list[dict] = field(default_factory=list)    # distinct complaints that were embedded: request_id, complaint, evidence, status, intent
    X: np.ndarray | None = None                           # unit-length embeddings, row-aligned with emb_rows


def normalise_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def sample_for_embedding(rows: list[dict], cap: int) -> list[dict]:
    """Distinct complaints (copy-paste and retries count once), thinned evenly to at most `cap` so the sample spans the whole window."""
    seen, out = set(), []
    for r in rows:
        key = hashlib.sha256(normalise_text(r.get("complaint") or "").encode()).hexdigest()
        if r.get("complaint") and key not in seen:
            seen.add(key)
            out.append(r)
    if len(out) <= cap:
        return out
    step = len(out) / cap
    return [out[int(i * step)] for i in range(cap)]


def _distinct(rows: list[dict]) -> list[dict]:
    """Each distinct complaint once (rows without text are kept as they are)."""
    if not any(r.get("complaint") for r in rows):
        return rows
    return sample_for_embedding(rows, 10**9)


def holm(pvalues: dict[str, float], alpha: float) -> set[str]:
    """Holm-Bonferroni step-down: the names whose p-value survives correction for testing all of `pvalues` at family-wise level alpha."""
    rejected: set[str] = set()
    m = len(pvalues)
    for i, (name, p) in enumerate(sorted(pvalues.items(), key=lambda kv: kv[1])):
        if p > alpha / (m - i):
            break
        rejected.add(name)
    return rejected


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _movers(changes: list[dict], k: int = 3) -> str:
    top = sorted((c for c in changes if c["delta"] != 0), key=lambda c: -abs(c["delta"]))[:k]
    return ", ".join(f"{c['label']} {_pct(c['baseline_share'])} to {_pct(c['recent_share'])}" for c in top)


def _p(p: float) -> str:
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ new-topic clusters
def _describe_cluster(rank: int, recent_idx: list[int], m: int, p_adj: float, enrichment: float, recent: Window, baseline: Window, cfg: DriftConfig, n_tested: int) -> dict:
    rows = [recent.emb_rows[i] for i in recent_idx]
    X = recent.X[recent_idx]
    centroid = X.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) or 1.0)
    order = np.argsort(-(X @ centroid))
    texts = [r["complaint"] for r in rows]
    background = [r["complaint"] for r in baseline.emb_rows[:400]]
    ev = [r["evidence"] for r in rows if r.get("evidence") is not None]
    abstained = sum(r.get("status") == "abstained" for r in rows) / len(rows)
    mean_ev = float(np.mean(ev)) if ev else None
    k = len(recent_idx)
    baseline_members = m - k
    return {
        "cluster_id": f"c{rank}", "size": k, "share_of_recent": round(k / max(1, len(recent.emb_rows)), 4),
        "baseline_members": baseline_members, "recent_share_of_cluster": round(k / m, 4), "enrichment": round(enrichment, 2),
        "kind": "new_topic" if baseline_members / m <= cfg.new_topic_baseline_share else "surge",
        "p_value": p_adj, "tests_in_family": n_tested, "significant": bool(p_adj <= cfg.alpha and enrichment >= cfg.cluster_enrichment),
        "keywords": keywords_per_cluster([texts], background, 6)[0], "examples": [texts[i][:200] for i in order[:3]],
        "mean_evidence": round(mean_ev, 4) if mean_ev is not None else None, "abstained_share": round(abstained, 4),
        "intent_mix": dict(sorted(stats.counts(r.get("intent") for r in rows).items(), key=lambda kv: -kv[1])[:4]),
        "covered_by_corpus": bool((mean_ev is not None and mean_ev >= cfg.novelty_threshold) and abstained < 0.5),
        "request_ids": [r["request_id"] for r in rows][:200], "cohesion": round(float(np.mean(X @ centroid)), 4),
    }


def _unexplained(members: list[int], rows: list[dict], cfg: DriftConfig) -> bool:
    """Whether the corpus explains a cluster poorly: mean evidence of ALL its members (both windows) below the novelty threshold. Evidence belongs to
    the complaint text, not to the window it arrived in, so this is decided blind to window membership and may steer the test budget."""
    ev = [rows[i]["evidence"] for i in members if rows[i].get("evidence") is not None]
    return bool(ev) and float(np.mean(ev)) < cfg.novelty_threshold


def _cluster_weights(unexplained: list[bool], share: float) -> list[float]:
    """Weighted Bonferroni (weights sum to 1; cluster i is tested at alpha * w_i). Clusters the corpus does not explain share `share` of the budget,
    the others the rest; with only one kind present it gets everything, and share = 0 gives the plain equal split. Known-topic surges keep a
    second detector (the category-mix tests); an unexplained new topic has only this one, which is why the budget leans its way."""
    n_u = sum(unexplained)
    n_e = len(unexplained) - n_u
    if not share or not n_u or not n_e:
        return [1.0 / len(unexplained)] * len(unexplained)
    return [share / n_u if u else (1.0 - share) / n_e for u in unexplained]


def _joint_clusters(recent: Window, baseline: Window, cfg: DriftConfig) -> list[dict]:
    """Cluster both windows together, blind to which window each complaint came from, then test each cluster for over-representation of recent ones.

    Because the clustering ignores window membership, under 'no drift' the number of recent complaints in any cluster is hypergeometric given the
    cluster sizes: an exact conditional test. Exact duplicates were removed per window beforehand. The correction over clusters is a weighted
    Bonferroni whose weights depend only on evidence (blind to window membership), so the family-wise level is unchanged.
    """
    from scipy.stats import hypergeom

    nr, nb = len(recent.emb_rows), len(baseline.emb_rows)
    X = np.vstack([recent.X, baseline.X])
    rows = recent.emb_rows + baseline.emb_rows
    cands = []
    for g in cluster_embeddings(X, cfg.cluster):
        rec = [i for i in g if i < nr]
        if len(rec) >= cfg.cluster.min_cluster_size:
            cands.append((g, rec))
    if not cands:
        return []
    weights = _cluster_weights([_unexplained(g, rows, cfg) for g, _ in cands], cfg.unexplained_weight)
    N = nr + nb
    scored = []
    for (g, rec), w in zip(cands, weights):
        m, k = len(g), len(rec)
        scored.append((min(1.0, float(hypergeom.sf(k - 1, N, nr, m)) / w), (k / m) / (nr / N), rec, m))
    scored.sort(key=lambda s: (s[0], -len(s[2])))
    return [_describe_cluster(rank, rec, m, p, enr, recent, baseline, cfg, len(cands))
            for rank, (p, enr, rec, m) in enumerate(scored, 1) if p <= WATCH_P][: cfg.max_clusters]


def _embedding_section(recent: Window, baseline: Window, cfg: DriftConfig, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    nb, nr = len(baseline.emb_rows), len(recent.emb_rows)
    if baseline.X is None or recent.X is None or nb < cfg.min_embedded or nr < cfg.min_embedded:
        return {"status": "skipped", "reason": f"needs at least {cfg.min_embedded} distinct complaints in each window (recent {nr}, baseline {nb})"}, []
    shift = stats.centroid_shift(baseline.X, recent.X, cfg.permutations, rng)
    base_nn = stats.nearest_similarity(baseline.X, baseline.X, exclude_self=True)
    thr = float(np.quantile(base_nn, cfg.unseen_quantile))
    recent_nn = stats.nearest_similarity(recent.X, baseline.X)
    unseen = int((recent_nn < thr).sum())
    sec = {"status": "ok", "n_recent": nr, "n_baseline": nb, "centroid_distance": round(shift["distance"], 5), "centroid_p_value": shift["p_value"],
           "unseen_similarity_threshold": round(thr, 4), "unseen_count": unseen, "unseen_rate": round(unseen / nr, 4), "unseen_expected_rate": cfg.unseen_quantile,
           "median_nn_similarity": {"baseline": round(float(np.median(base_nn)), 4), "recent": round(float(np.median(recent_nn)), 4)}}
    return sec, _joint_clusters(recent, baseline, cfg)


# ------------------------------------------------------------------ the report
def _dimension(dim: str, baseline: Window, recent: Window, cfg: DriftConfig) -> tuple[dict, float]:
    b, r = stats.counts(x.get(dim) for x in baseline.rows), stats.counts(x.get(dim) for x in recent.rows)
    omnibus = stats.chi2_homogeneity(b, r)
    score, _ = stats.psi(b, r)
    changes = stats.category_changes(b, r)
    k = max(1, len(changes))
    # two ways to see a shift: the whole mix moved a little (omnibus) or one category moved a lot (per-category); splitting the budget between them costs a factor 2
    p_dim = min(1.0, 2 * min(omnibus["p_value"], k * min((c["p_value"] for c in changes), default=1.0)))
    return {"psi": round(score, 4), "p_value": p_dim, "chi2_p_value": omnibus["p_value"], "dof": omnibus["dof"], "pooled_categories": omnibus["pooled"],
            "categories": changes}, p_dim


def analyze(recent: Window, baseline: Window, cfg: DriftConfig | None = None) -> dict:
    cfg = cfg or DriftConfig()
    rng = np.random.default_rng(cfg.seed)
    nr, nb = len(recent.rows), len(baseline.rows)
    enough = nr >= cfg.min_requests and nb >= cfg.min_requests
    report: dict = {"n_recent": nr, "n_baseline": nb, "thresholds": cfg.public(), "alerts": [], "distributions": {}, "quality": {}, "embedding": {}, "emerging_clusters": []}
    alerts: list[dict] = []
    pvals: dict[str, float] = {}

    for dim in CATEGORICAL:
        report["distributions"][dim], pvals[f"{dim}_distribution"] = _dimension(dim, baseline, recent, cfg)

    # Evidence and abstention belong to the complaint text: a complaint repeated 20 times has one evidence value, not 20 independent ones. These two tests
    # therefore see each distinct complaint once (counting copies as independent observations measurably inflates false alarms, docs/drift.md).
    # The category-mix tests keep every request: a surge of repeats IS a change in traffic.
    dist_b, dist_r = _distinct(baseline.rows), _distinct(recent.rows)
    eb = [x["evidence"] for x in dist_b if x.get("evidence") is not None]
    er = [x["evidence"] for x in dist_r if x.get("evidence") is not None]
    ks = stats.ks_test(eb, er)
    pvals["evidence_confidence"] = ks["p_value"]
    report["quality"]["evidence"] = {"baseline": stats.summary(eb), "recent": stats.summary(er), "ks_d": round(ks["d"], 4), "p_value": ks["p_value"]}
    xb, xr = sum(x["status"] == "abstained" for x in dist_b), sum(x["status"] == "abstained" for x in dist_r)
    rate_b, rate_r = (xb / len(dist_b) if dist_b else 0.0), (xr / len(dist_r) if dist_r else 0.0)
    p_abs = stats.proportion_test(xb, len(dist_b), xr, len(dist_r))
    pvals["abstention_rate"] = p_abs
    report["quality"]["abstention"] = {"baseline_rate": round(rate_b, 4), "recent_rate": round(rate_r, 4), "p_value": p_abs}

    embedding, clusters = _embedding_section(recent, baseline, cfg, rng)
    report["embedding"] = embedding
    if embedding.get("status") == "ok":
        pvals["embedding_centroid"] = embedding["centroid_p_value"]

    survivors = holm(pvals, cfg.alpha) if enough else set()
    report["tests"] = {"n": len(pvals), "family_alpha": cfg.alpha, "method": "Holm-Bonferroni", "p_values": pvals, "survive_correction": sorted(survivors)}

    for dim in CATEGORICAL:
        d = report["distributions"][dim]
        d["drifted"] = bool(f"{dim}_distribution" in survivors and d["psi"] >= cfg.psi_threshold)
        if d["drifted"]:
            alerts.append({"signal": f"{dim}_distribution", "kind": "distribution", "p_value": d["p_value"], "effect": {"psi": d["psi"]},
                           "message": f"{dim} mix shifted (p={_p(d['p_value'])}, PSI {d['psi']:.2f}): {_movers(d['categories'])}"})
    q = report["quality"]
    q["evidence"]["drifted"] = bool("evidence_confidence" in survivors and ks["d"] >= cfg.ks_threshold and er and eb and np.mean(er) < np.mean(eb))
    if q["evidence"]["drifted"]:
        alerts.append({"signal": "evidence_confidence", "kind": "quality", "p_value": ks["p_value"], "effect": {"ks_d": round(ks["d"], 4)},
                       "message": f"evidence confidence fell (KS D={ks['d']:.2f}, p={_p(ks['p_value'])}): mean {np.mean(eb):.2f} to {np.mean(er):.2f}, "
                                  f"10th percentile {np.percentile(eb, 10):.2f} to {np.percentile(er, 10):.2f}. Retrieval regression, stale knowledge base or vocabulary shift"})
    q["abstention"]["drifted"] = bool("abstention_rate" in survivors and rate_r - rate_b >= cfg.abstention_increase)
    if q["abstention"]["drifted"]:
        alerts.append({"signal": "abstention_rate", "kind": "quality", "p_value": p_abs, "effect": {"increase": round(rate_r - rate_b, 4)},
                       "message": f"abstention rose from {_pct(rate_b)} to {_pct(rate_r)} (p={_p(p_abs)}): requests the corpus cannot answer are becoming common"})
    if embedding.get("status") == "ok":
        embedding["centroid_drifted"] = bool("embedding_centroid" in survivors and embedding["centroid_distance"] >= cfg.embedding_shift)
        if embedding["centroid_drifted"]:
            alerts.append({"signal": "embedding_centroid", "kind": "embedding", "p_value": embedding["centroid_p_value"], "effect": {"cosine_distance": embedding["centroid_distance"]},
                           "message": f"complaints moved in embedding space (centroid cosine distance {embedding['centroid_distance']:.3f}, permutation p={_p(embedding['centroid_p_value'])})"})
    for c in clusters:
        c["significant"] = bool(enough and c["significant"])
        if c["significant"]:
            what = {"new_topic": "almost nothing like it appears in the baseline", "surge": f"the baseline has {c['baseline_members']} similar complaints, recent traffic far more"}[c["kind"]]
            expl = "no existing evidence explains it" if not c["covered_by_corpus"] else "existing tickets explain it"
            alerts.append({"signal": "emerging_cluster", "kind": "emerging_cluster", "p_value": c["p_value"], "effect": {"size": c["size"], "enrichment": c["enrichment"]},
                           "cluster_id": c["cluster_id"],
                           "message": f"{c['size']} recent complaints form a {'new topic' if c['kind'] == 'new_topic' else 'surging topic'} "
                                      f"({', '.join(c['keywords'][:4]) or 'no keywords'}): {what}; {expl} (p={_p(c['p_value'])})"})
    report["emerging_clusters"] = clusters
    report["alerts"] = alerts
    report["status"] = "insufficient_data" if not enough else ("alert" if alerts else "ok")
    if not enough:
        report["note"] = f"need at least {cfg.min_requests} requests in each window (recent {nr}, baseline {nb}); nothing is alerted on yet"
    return report
