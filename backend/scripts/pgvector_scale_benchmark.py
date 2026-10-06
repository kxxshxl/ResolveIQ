"""SYNTHETIC pgvector scale experiment: ~100k 384-dimensional vectors in a throw-away database.

    python scripts/pgvector_scale_benchmark.py [--n 100000] [--queries 500] [--keep]

This is a bounded experiment about the vector index, NOT a production benchmark: the vectors are synthetic (a Gaussian mixture on the unit sphere, shaped so that
within-topic cosine similarity looks like sentence embeddings), the corpus has no text, one Postgres container serves it, and the load is a handful of client threads.
It answers: how long does the HNSW build take, what do search latency percentiles and recall look like against an exact brute-force ground truth as ef_search changes,
what does a selective metadata filter do to recall, how much disk does it take, and does the index return sane results (self-queries, ordering, exact distances).

It creates database `resolveiq_scale_bench` with the admin URL from the environment, never touches the application database, and drops it afterwards unless --keep.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import REPO_ROOT, get_settings  # noqa: E402

DB = "resolveiq_scale_bench"


def swap(url: str, db: str) -> str:
    return urlunparse(urlparse(url)._replace(path=f"/{db}"))


def lit(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def make_vectors(n: int, dim: int, topics: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors in `topics` clusters; also returns the topic of each vector and the cluster centres."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(topics, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    topic = rng.integers(0, topics, size=n)
    # noise scaled so that two vectors from the same topic have cosine similarity around 0.55 to 0.7, like related support complaints
    noise = rng.normal(size=(n, dim)).astype(np.float32) * (0.9 / np.sqrt(dim))
    X = centres[topic] + noise
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, topic, centres


def pct(a, p):
    return round(float(np.percentile(a, p)), 3)


def lat_stats(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": pct(ms, 50), "p95_ms": pct(ms, 95), "p99_ms": pct(ms, 99), "mean_ms": round(float(np.mean(ms)), 3), "max_ms": round(max(ms), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--topics", type=int, default=400)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=64)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[10, 20, 40, 100, 200])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "eval" / "results" / "pgvector_scale.json"))
    a = ap.parse_args()
    admin = get_settings().database_admin_url
    root = swap(admin, "postgres")
    res: dict = {"experiment": "SYNTHETIC pgvector scale experiment (not a production benchmark)",
                 "setup": {"vectors": a.n, "dimensions": a.dim, "topics": a.topics, "queries": a.queries, "k": a.k, "seed": a.seed, "hnsw": {"m": a.m, "ef_construction": a.ef_construction}}}
    with psycopg.connect(root, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {DB}")
    url = swap(admin, DB)
    try:
        t = time.perf_counter()
        X, topic, centres = make_vectors(a.n, a.dim, a.topics, a.seed)
        rng = np.random.default_rng(a.seed + 1)
        grp = rng.integers(0, 1000, size=a.n)                      # an attribute unrelated to the content, e.g. a customer segment
        qt = rng.integers(0, a.topics, size=a.queries)
        Q = centres[qt] + rng.normal(size=(a.queries, a.dim)).astype(np.float32) * (0.9 / np.sqrt(a.dim))
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        res["setup"]["generate_s"] = round(time.perf_counter() - t, 1)
        sims = X @ X[:2000].T
        same = (topic[:, None] == topic[None, :2000])
        res["setup"]["mean_within_topic_cosine"] = round(float(sims[same & (sims < 0.999)].mean()), 3)
        res["setup"]["mean_across_topic_cosine"] = round(float(sims[~same].mean()), 3)
        del sims, same

        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION vector")
            res["server"] = {"postgres": conn.execute("SHOW server_version").fetchone()[0], "pgvector": conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0],
                             "shared_buffers": conn.execute("SHOW shared_buffers").fetchone()[0], "max_parallel_maintenance_workers": conn.execute("SHOW max_parallel_maintenance_workers").fetchone()[0]}
            conn.execute(f"CREATE TABLE bench (id integer PRIMARY KEY, topic smallint NOT NULL, grp smallint NOT NULL, embedding vector({a.dim}) NOT NULL)")
            t = time.perf_counter()
            with conn.cursor().copy("COPY bench (id, topic, grp, embedding) FROM STDIN") as cp:
                for i in range(a.n):
                    cp.write_row((i, int(topic[i]), int(grp[i]), lit(X[i])))
            res["load"] = {"seconds": round(time.perf_counter() - t, 1), "rows_per_second": round(a.n / (time.perf_counter() - t))}
            conn.execute("ANALYZE bench")

            conn.execute("SET maintenance_work_mem = '1GB'")
            conn.execute("SET max_parallel_maintenance_workers = 0")   # a parallel build needs shared memory the default 64 MB /dev/shm of the container cannot give
            t = time.perf_counter()
            conn.execute(f"CREATE INDEX bench_hnsw ON bench USING hnsw (embedding vector_cosine_ops) WITH (m = {a.m}, ef_construction = {a.ef_construction})")
            res["index_build"] = {"seconds": round(time.perf_counter() - t, 1), "maintenance_work_mem": "1GB", "parallel_workers": 0,
                                  "note": "serial build: the container's 64 MB /dev/shm cannot hold the shared memory a parallel build asks for, so this is a lower bound on build speed"}
            conn.execute("CREATE INDEX bench_topic ON bench (topic)")
            conn.execute("CREATE INDEX bench_grp ON bench (grp)")
            conn.execute("ANALYZE bench")
            sz = conn.execute("SELECT pg_relation_size('bench'), pg_relation_size('bench_hnsw'), pg_total_relation_size('bench'), pg_relation_size('bench_topic')").fetchone()
            res["storage"] = {"table_mb": round(sz[0] / 2**20, 1), "hnsw_index_mb": round(sz[1] / 2**20, 1), "btree_topic_index_mb": round(sz[3] / 2**20, 1), "total_mb": round(sz[2] / 2**20, 1),
                              "raw_vector_data_mb": round(a.n * a.dim * 4 / 2**20, 1), "index_bytes_per_vector": round(sz[1] / a.n)}
            plan = "\n".join(r[0] for r in conn.execute(f"EXPLAIN SELECT id FROM bench ORDER BY embedding <=> '{lit(Q[0])}'::vector LIMIT {a.k}").fetchall())
            res["plan_uses_hnsw"] = "bench_hnsw" in plan

        # exact ground truth (brute force in numpy: the index is judged against this, not against itself)
        t = time.perf_counter()
        S = Q @ X.T
        truth = np.argsort(-S, axis=1)[:, : a.k]
        res["setup"]["ground_truth_s"] = round(time.perf_counter() - t, 1)

        def one(conn, q, k=a.k, where=""):
            t0 = time.perf_counter()
            rows = conn.execute(f"SELECT id, 1 - (embedding <=> %s::vector) AS sim FROM bench {where} ORDER BY embedding <=> %s::vector LIMIT {k}", (lit(q), lit(q))).fetchall()
            return rows, (time.perf_counter() - t0) * 1000

        res["search"] = {}
        with psycopg.connect(url, autocommit=True) as conn:
            for _ in range(20):                                    # warm the buffer cache and the plan
                one(conn, Q[0])
            for ef in a.ef_search:
                conn.execute(f"SET hnsw.ef_search = {ef}")
                lat, rec, top1 = [], [], 0
                for qi, q in enumerate(Q):
                    rows, ms = one(conn, q)
                    lat.append(ms)
                    got = {r[0] for r in rows}
                    rec.append(len(got & set(truth[qi].tolist())) / a.k)
                    top1 += rows[0][0] == int(truth[qi][0])
                res["search"][f"ef_search={ef}"] = {**lat_stats(lat), "recall@10": round(float(np.mean(rec)), 4), "min_recall": round(float(min(rec)), 2), "top1_agreement": round(top1 / a.queries, 4)}
            conn.execute("SET hnsw.ef_search = 40")

            # sanity checks on the index itself
            sample = rng.choice(a.n, size=200, replace=False)
            selfq = sum(one(conn, X[i], k=1)[0][0][0] == int(i) for i in sample)
            ordered = distances_ok = 0
            for qi in range(min(100, a.queries)):
                rows, _ = one(conn, Q[qi])
                s = [r[1] for r in rows]
                ordered += all(s[i] >= s[i + 1] - 1e-9 for i in range(len(s) - 1))
                exact = X[[r[0] for r in rows]] @ Q[qi]
                distances_ok += bool(np.allclose(exact, s, atol=1e-4))
            res["sanity"] = {"self_query_returns_itself_at_rank_1": f"{selfq}/200", "results_sorted_by_similarity": f"{ordered}/{min(100, a.queries)}", "reported_similarity_matches_exact_dot_product": f"{distances_ok}/{min(100, a.queries)}"}

            # selective metadata filters, the classic ANN pitfall: the index returns ~ef_search candidates and the filter then removes most of them.
            # "independent" = the attribute says nothing about the content (worst case); "correlated" = it follows the topic, as an intent or product filter does.
            res["filtered_search"] = {}
            for kind, col, hi_share in (("independent attribute", "grp", 1000), ("correlated attribute", "topic", a.topics)):
                for share in (0.05, 0.005):
                    hi = max(1, round(hi_share * share))
                    where = f"WHERE {col} < {hi}"
                    mask = (grp if col == "grp" else topic) < hi
                    if col == "topic":
                        qt_f = rng.integers(0, hi, size=100)
                        Qf = centres[qt_f] + rng.normal(size=(100, a.dim)).astype(np.float32) * (0.9 / np.sqrt(a.dim))
                        Qf /= np.linalg.norm(Qf, axis=1, keepdims=True)
                    else:
                        Qf = Q[:100] if a.queries >= 100 else Q
                    Sf = Qf @ X.T
                    Sf[:, ~mask] = -9
                    truth_f = np.argsort(-Sf, axis=1)[:, : a.k]
                    filt: dict = {"filter": where.replace("WHERE ", ""), "selectivity": round(float(mask.mean()), 4), "matching_rows": int(mask.sum())}
                    conn.execute("SET hnsw.ef_search = 40")
                    for label, setting in (("iterative_scan=off", "off"), ("iterative_scan=relaxed_order", "relaxed_order")):
                        conn.execute(f"SET hnsw.iterative_scan = {setting}")
                        plan = " | ".join(r[0] for r in conn.execute(f"EXPLAIN SELECT id FROM bench {where} ORDER BY embedding <=> '{lit(Qf[0])}'::vector LIMIT {a.k}").fetchall())
                        lat, rec, short = [], [], 0
                        for qi, q in enumerate(Qf):
                            rows, ms = one(conn, q, where=where)
                            lat.append(ms)
                            rec.append(len({r[0] for r in rows} & set(truth_f[qi].tolist())) / a.k)
                            short += len(rows) < a.k
                        filt[label] = {**lat_stats(lat), "recall@10": round(float(np.mean(rec)), 4), "queries_returning_fewer_than_k": short,
                                       "plan": "hnsw index scan" if "bench_hnsw" in plan else ("btree index" if "bench_topic" in plan or "bench_grp" in plan else "sequential scan")}
                    res["filtered_search"][f"{kind}, {share:.1%} of rows"] = filt
            conn.execute("SET hnsw.iterative_scan = off")

        # concurrency: N client threads hammering the index for a fixed number of queries each
        lats: list[float] = []
        lock = threading.Lock()

        def worker(seed_off: int):
            with psycopg.connect(url, autocommit=True) as conn:
                conn.execute("SET hnsw.ef_search = 40")
                mine = []
                for i in range(a.queries // 2):
                    mine.append(one(conn, Q[(i * 7 + seed_off) % a.queries])[1])
            with lock:
                lats.extend(mine)

        t = time.perf_counter()
        th = [threading.Thread(target=worker, args=(i,)) for i in range(a.threads)]
        [x.start() for x in th]
        [x.join() for x in th]
        wall = time.perf_counter() - t
        res["concurrency"] = {"client_threads": a.threads, "queries": len(lats), "wall_s": round(wall, 2), "queries_per_second": round(len(lats) / wall, 1), **lat_stats(lats), "ef_search": 40}
        res["caveats"] = ["synthetic vectors (Gaussian mixture), no text", "one Postgres container on a laptop, client and server share the machine", "latencies include the client round trip",
                          "memory use was not measured; the HNSW index size is the lower bound on what must stay cached", "not a production benchmark"]
    finally:
        if not a.keep:
            with psycopg.connect(root, autocommit=True) as c:
                c.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
