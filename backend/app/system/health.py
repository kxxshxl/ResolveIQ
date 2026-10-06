"""Database health: safe operational metrics only (sizes, counts, index usage, coverage, integrity). No row contents, no credentials.

Every query is a catalog or aggregate read bounded by table size; the whole report runs in tens of milliseconds on the project's corpus. The same checks are what
the database review (docs/database.md) used to find the issues it fixed.
"""
from __future__ import annotations

import re
import time

from app.core.config import Settings
from app.db.repository import Repository
from app.observability import tracing

KEY_TABLES = ("tickets", "knowledge_articles", "ticket_embeddings", "kb_embeddings", "resolution_requests", "feedback", "jobs", "taxonomy_proposals", "drift_snapshots")


def _hnsw_params(indexdef: str) -> dict:
    m = re.search(r"WITH \((.*?)\)", indexdef)
    params = dict(p.strip().split("=") for p in m.group(1).split(",")) if m else {}
    return {"m": int(params.get("m", 16)), "ef_construction": int(params.get("ef_construction", 64)), "defaults": not params}


@tracing.traced("db.health", result=lambda r: {"resolveiq.db.findings": len(r["findings"]), "resolveiq.db.status": r["status"]})
async def database_health(repo: Repository, s: Settings) -> dict:
    t0 = time.perf_counter()
    q = repo._fetch
    findings: list[dict] = []

    def finding(level: str, what: str, detail: str):
        findings.append({"level": level, "what": what, "detail": detail})

    ver = (await q("SELECT extversion FROM pg_extension WHERE extname='vector'"))
    server = (await q("SHOW server_version"))[0]["server_version"]
    tables = await q(
        """SELECT c.relname AS name, c.reltuples::bigint AS est_rows, pg_total_relation_size(c.oid) AS total_bytes, pg_relation_size(c.oid) AS table_bytes,
                  pg_indexes_size(c.oid) AS index_bytes, coalesce(st.n_dead_tup, 0) AS dead_rows, st.last_autovacuum, st.last_autoanalyze
           FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace LEFT JOIN pg_stat_user_tables st ON st.relid = c.oid
           WHERE n.nspname = 'public' AND c.relkind = 'r' AND c.relname = ANY(%s) ORDER BY c.relname""", (list(KEY_TABLES),))
    for t in tables:
        t["exact_rows"] = (await q(f"SELECT count(*) AS n FROM {t['name']}"))[0]["n"] if t["name"] not in ("ticket_embeddings", "kb_embeddings", "resolution_requests") else None
    indexes = await q(
        """SELECT i.relname AS name, t.relname AS table, pg_relation_size(i.oid) AS bytes, coalesce(s.idx_scan, 0) AS scans, x.indisunique AS is_unique, x.indisprimary AS is_primary,
                  pg_get_indexdef(i.oid) AS definition
           FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_class t ON t.oid = x.indrelid JOIN pg_namespace n ON n.oid = t.relnamespace
           LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = i.oid WHERE n.nspname = 'public' AND t.relname = ANY(%s) ORDER BY pg_relation_size(i.oid) DESC""", (list(KEY_TABLES),))
    vector_indexes = []
    unused = []
    for ix in indexes:
        d = ix.pop("definition")
        ix["method"] = re.search(r"USING (\w+)", d).group(1) if re.search(r"USING (\w+)", d) else None
        if ix["method"] == "hnsw":
            vector_indexes.append({"name": ix["name"], "table": ix["table"], "bytes": ix["bytes"], **_hnsw_params(d)})
        if ix["scans"] == 0 and not ix["is_unique"] and not ix["is_primary"] and ix["bytes"] > 64 * 1024:
            unused.append(ix["name"])
    if unused:
        finding("info", "indexes never scanned", f"{', '.join(unused[:5])} have not been used since statistics were last reset; fine on a fresh database, a candidate for removal on a busy one")

    model = s.embedding_model
    cov = (await q(
        """SELECT (SELECT count(*) FROM tickets d WHERE d.status='active' AND NOT EXISTS (SELECT 1 FROM ticket_embeddings e WHERE e.ticket_pk = d.id AND e.model = %(m)s)) AS tickets_missing,
                  (SELECT count(*) FROM knowledge_articles d WHERE d.status='active' AND NOT EXISTS (SELECT 1 FROM kb_embeddings e WHERE e.article_pk = d.id AND e.model = %(m)s)) AS articles_missing,
                  (SELECT count(*) FROM ticket_embeddings WHERE model <> %(m)s) AS tickets_other_model, (SELECT count(*) FROM kb_embeddings WHERE model <> %(m)s) AS articles_other_model""", {"m": model}))[0]
    if cov["tickets_missing"] or cov["articles_missing"]:
        finding("warn", "documents without an embedding", f"{cov['tickets_missing']} tickets and {cov['articles_missing']} articles have no {model} embedding and can never be retrieved (run reindex)")
    integrity = (await q(
        """SELECT (SELECT count(*) FROM ticket_embeddings e WHERE NOT EXISTS (SELECT 1 FROM tickets d WHERE d.id = e.ticket_pk)) AS orphan_ticket_embeddings,
                  (SELECT count(*) FROM kb_embeddings e WHERE NOT EXISTS (SELECT 1 FROM knowledge_articles d WHERE d.id = e.article_pk)) AS orphan_article_embeddings,
                  (SELECT count(*) FROM feedback WHERE request_id IS NULL) AS feedback_without_request,
                  (SELECT count(*) FROM taxonomy_proposals p WHERE p.status = 'pending' AND cardinality(p.member_request_ids) > 0
                      AND NOT EXISTS (SELECT 1 FROM resolution_requests r WHERE r.request_id = ANY(p.member_request_ids))) AS pending_proposals_without_requests,
                  (SELECT count(*) FROM tickets WHERE status='active' AND intent NOT IN (SELECT label_id FROM taxonomy_labels WHERE dimension='intent')) AS tickets_with_unknown_intent,
                  (SELECT count(*) FROM knowledge_articles WHERE status='active' AND category NOT IN (SELECT label_id FROM taxonomy_labels WHERE dimension='intent')) AS articles_with_unknown_category"""))[0]
    for k, v in integrity.items():
        if v:
            finding("warn", k.replace("_", " "), f"{v} row(s)")
    dangling = (await q(
        """WITH cited AS (SELECT DISTINCT c->>'source_id' AS sid FROM (SELECT result->'citations' AS cites FROM resolution_requests ORDER BY created_at DESC LIMIT 500) r,
                                  jsonb_array_elements(coalesce(r.cites, '[]'::jsonb)) c)
           SELECT count(*) AS n FROM cited WHERE sid NOT IN (SELECT ticket_id FROM tickets) AND sid NOT IN (SELECT article_id FROM knowledge_articles)"""))[0]["n"]
    if dangling:
        finding("info", "cited sources no longer in the corpus", f"{dangling} source id(s) cited by the last 500 resolutions were deleted since; old cases will show them as missing")
    dup = (await q(
        """SELECT (SELECT count(*) FROM (SELECT 1 FROM tickets GROUP BY content_hash HAVING count(*) > 1) x) AS ticket_duplicate_groups,
                  (SELECT count(*) FROM (SELECT 1 FROM knowledge_articles GROUP BY content_hash HAVING count(*) > 1) x) AS article_duplicate_groups"""))[0]
    if dup["ticket_duplicate_groups"] or dup["article_duplicate_groups"]:
        finding("info", "duplicate content", f"{dup['ticket_duplicate_groups']} ticket and {dup['article_duplicate_groups']} article groups share identical text (different ids)")

    runtime = (await q("SELECT (SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()) AS connections, current_setting('max_connections')::int AS max_connections"))[0]
    cache = (await q("SELECT blks_hit, blks_read, xact_commit, xact_rollback, deadlocks FROM pg_stat_database WHERE datname = current_database()"))[0]
    total_blocks = cache["blks_hit"] + cache["blks_read"]
    async with repo.pool.connection() as conn:
        ef = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())["hnsw.ef_search"]
    pool = repo.pool.get_stats()
    dead = [t for t in tables if t["est_rows"] > 1000 and t["dead_rows"] > 0.2 * max(t["est_rows"], 1)]
    for t in dead:
        finding("warn", "table bloat", f"{t['name']}: {t['dead_rows']} dead rows ({t['dead_rows'] / max(t['est_rows'], 1):.0%}); check autovacuum")
    status = "warn" if any(f["level"] == "warn" for f in findings) else "ok"
    return {
        "status": status, "findings": findings, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "server": {"postgres": server, "pgvector": ver[0]["extversion"] if ver else None, "connections": runtime["connections"], "max_connections": runtime["max_connections"],
                   "buffer_cache_hit_ratio": round(cache["blks_hit"] / total_blocks, 4) if total_blocks else None, "deadlocks": cache["deadlocks"], "rollback_ratio": round(
                       cache["xact_rollback"] / max(1, cache["xact_commit"] + cache["xact_rollback"]), 4)},
        "pool": {"min": s.db_pool_min, "max": s.db_pool_max, "size": pool.get("pool_size"), "available": pool.get("pool_available"), "requests_waiting": pool.get("requests_waiting"),
                 "statement_timeout_ms": s.db_statement_timeout_ms},
        "vector_search": {"embedding_model": model, "dimensions": s.embedding_dim, "hnsw_ef_search": int(ef), "indexes": vector_indexes},
        "tables": [{"name": t["name"], "rows": t["exact_rows"] if t["exact_rows"] is not None else max(0, int(t["est_rows"])), "total_bytes": t["total_bytes"], "table_bytes": t["table_bytes"],
                    "index_bytes": t["index_bytes"], "dead_rows": t["dead_rows"], "last_autovacuum": t["last_autovacuum"], "last_autoanalyze": t["last_autoanalyze"]} for t in tables],
        "indexes": [{k: ix[k] for k in ("name", "table", "method", "bytes", "scans", "is_unique")} for ix in indexes[:30]],
        "embedding_coverage": cov, "integrity": integrity, "duplicates": dup}
