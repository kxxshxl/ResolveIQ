-- Case replay, feedback intelligence and database hardening.

-- A case (resolution_requests row) now keeps the stage-by-stage trace, the provenance (model, prompt, taxonomy, corpus, retrieval configuration) and the
-- query embedding. The embedding is the one the request already computed, so keeping it costs 1.5 KB and lets clustering and drift analysis skip re-embedding.
ALTER TABLE resolution_requests ADD COLUMN trace           jsonb;
ALTER TABLE resolution_requests ADD COLUMN provenance      jsonb;
ALTER TABLE resolution_requests ADD COLUMN embedding       vector({{EMBEDDING_DIM}});
ALTER TABLE resolution_requests ADD COLUMN embedding_model text;
-- aggregations by intent over a time range (quality view, timeline); the older (status, created_at) and created_at indexes stay
CREATE INDEX resolution_requests_intent_idx ON resolution_requests ((classification->>'intent'), created_at DESC);
-- backfill of embeddings for rows logged before this migration (or served from the response cache) only ever looks at rows that lack one
CREATE INDEX resolution_requests_unembedded_idx ON resolution_requests (created_at DESC) WHERE embedding IS NULL;

-- Feedback keeps what the agent said and which pipeline version they were rating.
ALTER TABLE feedback ADD COLUMN reasons          text[] NOT NULL DEFAULT '{}';
ALTER TABLE feedback ADD COLUMN rejected_sources text[] NOT NULL DEFAULT '{}';
ALTER TABLE feedback ADD COLUMN edited_steps     jsonb;     -- the resolution as the agent would send it (redacted)
ALTER TABLE feedback ADD COLUMN provenance       jsonb;     -- copy of the rated resolution's provenance, so later pipeline changes cannot rewrite history
-- feedback.request_id is a foreign key with ON DELETE SET NULL: without an index every delete of a request scans the whole feedback table
CREATE INDEX feedback_request_idx ON feedback (request_id);
CREATE INDEX feedback_created_idx ON feedback (created_at DESC);

-- Duplicate detection: the same complaint and resolution ingested under two ids. Not unique (a deliberate re-ingest must stay possible); reported by the health view.
ALTER TABLE tickets            ADD COLUMN content_hash text GENERATED ALWAYS AS (md5(lower(complaint_text) || '|' || lower(resolution_summary))) STORED;
ALTER TABLE knowledge_articles ADD COLUMN content_hash text GENERATED ALWAYS AS (md5(lower(title) || '|' || lower(content))) STORED;
CREATE INDEX tickets_content_hash_idx ON tickets (content_hash);
CREATE INDEX articles_content_hash_idx ON knowledge_articles (content_hash);

-- Scheduling and de-duplication of background work look up jobs by kind
CREATE INDEX jobs_kind_created_idx ON jobs (kind, created_at DESC);
