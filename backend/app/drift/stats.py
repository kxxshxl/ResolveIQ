"""Statistics behind drift detection. Pure, deterministic (a seeded generator is passed in) and free of I/O.

Every detector pairs a *significance* test (is the difference bigger than sampling noise at these sample sizes?) with an
*effect size* (is it big enough to matter?). Alerting needs both: significance alone fires on tiny differences once a window
holds thousands of requests, effect size alone fires on noise when it holds thirty.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


def counts(values: Iterable[str | None], unknown: str = "unknown") -> dict[str, int]:
    return dict(Counter(v if v else unknown for v in values))


# ------------------------------------------------------------------ categorical distributions
def psi(base: dict[str, int], recent: dict[str, int], pseudo: float = 0.5) -> tuple[float, dict[str, float]]:
    """Population Stability Index, sum((r - b) * ln(r / b)), and each category's contribution to it.

    Counts get an additive pseudo-count so a category absent from one window gives a finite value. Rule of thumb from credit
    scoring, widely used for feature drift: < 0.10 stable, 0.10 to 0.25 moderate shift, > 0.25 major shift.
    """
    keys = sorted(set(base) | set(recent))
    if not keys:
        return 0.0, {}
    nb, nr = sum(base.values()) + pseudo * len(keys), sum(recent.values()) + pseudo * len(keys)
    contrib = {}
    for k in keys:
        b, r = (base.get(k, 0) + pseudo) / nb, (recent.get(k, 0) + pseudo) / nr
        contrib[k] = (r - b) * math.log(r / b)
    return float(sum(contrib.values())), contrib


def chi2_homogeneity(base: dict[str, int], recent: dict[str, int], min_expected: float = 5.0) -> dict:
    """Chi-square test that both windows are drawn from one category distribution.

    Categories whose expected count is below `min_expected` in either window are pooled into one "(other)" bucket first (the usual
    validity rule for the chi-square approximation). If fewer than two buckets remain the test does not apply and p = 1.
    """
    nb, nr = sum(base.values()), sum(recent.values())
    if nb == 0 or nr == 0:
        return {"chi2": 0.0, "dof": 0, "p_value": 1.0, "pooled": []}
    keep, pooled = [], []
    for k in sorted(set(base) | set(recent)):
        total = base.get(k, 0) + recent.get(k, 0)
        exp_min = total * min(nb, nr) / (nb + nr)
        (keep if exp_min >= min_expected else pooled).append(k)
    b = [base.get(k, 0) for k in keep]
    r = [recent.get(k, 0) for k in keep]
    if pooled:
        b.append(sum(base.get(k, 0) for k in pooled))
        r.append(sum(recent.get(k, 0) for k in pooled))
    keepcols = [i for i in range(len(b)) if b[i] + r[i] > 0]
    if len(keepcols) < 2:
        return {"chi2": 0.0, "dof": 0, "p_value": 1.0, "pooled": pooled}
    table = np.array([[b[i] for i in keepcols], [r[i] for i in keepcols]], dtype=float)
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "pooled": pooled}


def proportion_test(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided p for 'same underlying rate' in two windows: pooled z-test, or Fisher's exact test when counts are small."""
    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = (x1 + x2) / (n1 + n2)
    if pooled in (0.0, 1.0):
        return 1.0
    if min(pooled * min(n1, n2), (1 - pooled) * min(n1, n2)) < 10:
        return float(stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])[1])
    z = (x2 / n2 - x1 / n1) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return float(2 * stats.norm.sf(abs(z)))


def category_changes(base: dict[str, int], recent: dict[str, int], min_count: int = 3) -> list[dict]:
    """Per-category view: share in each window, change in percentage points, own p-value, share of the PSI, and a plain-language flag."""
    nb, nr = sum(base.values()), sum(recent.values())
    _, contrib = psi(base, recent)
    total = sum(contrib.values()) or 1.0
    out = []
    for k in sorted(set(base) | set(recent), key=lambda k: -(base.get(k, 0) + recent.get(k, 0))):
        b, r = base.get(k, 0), recent.get(k, 0)
        sb, sr = (b / nb if nb else 0.0), (r / nr if nr else 0.0)
        p = proportion_test(b, nb, r, nr)
        flag = None
        if b == 0 and r >= min_count:
            flag = "new"
        elif r == 0 and b >= min_count:
            flag = "vanished"
        elif p < 0.01 and abs(sr - sb) >= 0.03:
            flag = "up" if sr > sb else "down"
        out.append({"label": k, "baseline_n": b, "recent_n": r, "baseline_share": round(sb, 4), "recent_share": round(sr, 4),
                    "delta": round(sr - sb, 4), "p_value": p, "psi_share": round(contrib.get(k, 0.0) / total, 4), "flag": flag})
    return out


# ------------------------------------------------------------------ numeric distributions
def ks_test(base: Sequence[float], recent: Sequence[float]) -> dict:
    """Two-sample Kolmogorov-Smirnov: D is the largest gap between the two empirical CDFs (0 identical .. 1 disjoint)."""
    if len(base) < 2 or len(recent) < 2:
        return {"d": 0.0, "p_value": 1.0}
    r = stats.ks_2samp(base, recent)
    return {"d": float(r.statistic), "p_value": float(r.pvalue)}


def summary(values: Sequence[float]) -> dict | None:
    if not len(values):
        return None
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "median": round(float(np.median(a)), 4), "p90": round(float(np.percentile(a, 90)), 4)}


# ------------------------------------------------------------------ embeddings
def centroid_shift(Xb: np.ndarray, Xr: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """Cosine distance between the two windows' mean embeddings, with a permutation p-value.

    The p-value asks: if the window labels were shuffled, how often would a split this lopsided appear? No distributional
    assumption is made about embeddings, which are not Gaussian.
    """
    def dist(a_sum: np.ndarray, a_n: int, b_sum: np.ndarray, b_n: int) -> float:
        a, b = a_sum / a_n, b_sum / b_n
        den = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return 1.0 - float(a @ b) / den

    nb, nr = len(Xb), len(Xr)
    pool = np.vstack([Xb, Xr]).astype(np.float64)
    total = pool.sum(axis=0)
    obs = dist(pool[:nb].sum(axis=0), nb, pool[nb:].sum(axis=0), nr)
    ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(nb + nr)[:nr]
        r_sum = pool[idx].sum(axis=0)
        ge += dist(total - r_sum, nb, r_sum, nr) >= obs
    return {"distance": obs, "p_value": (1 + ge) / (n_perm + 1)}


def nearest_similarity(Xq: np.ndarray, Xref: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Highest cosine similarity of each query row to any reference row (embeddings are unit length)."""
    S = Xq @ Xref.T
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    return S.max(axis=1)

