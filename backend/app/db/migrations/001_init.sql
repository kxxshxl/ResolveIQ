-- ResolveIQ schema v1. {{EMBEDDING_DIM}} is substituted by the migration runner.
CREATE EXTENSION IF NOT EXISTS vector;

-- Extensible label space for every classification dimension (intent/product/severity/sentiment).
-- New classes are plain INSERTs; no code or schema change. Each change bumps taxonomy_versions.
CREATE TABLE taxonomy_versions (
    version     integer PRIMARY KEY,
    note        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE taxonomy_labels (
    dimension   text NOT NULL CHECK (dimension IN ('intent','product','severity','sentiment')),
    label_id    text NOT NULL,
    description text NOT NULL DEFAULT '',
    keywords    text[] NOT NULL DEFAULT '{}',
    examples    text[] NOT NULL DEFAULT '{}',
    rank        integer,                       -- ordering for ordinal dimensions (severity)
    team        text,                          -- escalation target when evidence is weak (intent only)
    status      text NOT NULL DEFAULT 'active' CHECK (status IN ('active','deprecated')),
    introduced_in integer NOT NULL REFERENCES taxonomy_versions(version),
    PRIMARY KEY (dimension, label_id)
);

CREATE TABLE tickets (
    id                 bigserial PRIMARY KEY,
    ticket_id          text NOT NULL UNIQUE,
    complaint_text     text NOT NULL,           -- PII-redacted at ingestion; raw text is never stored
    intent             text NOT NULL,
    product            text NOT NULL,
    severity           text NOT NULL,
    sentiment          text NOT NULL,
    resolution_steps   jsonb NOT NULL DEFAULT '[]',
    resolution_summary text NOT NULL,
    resolved_at        timestamptz NOT NULL,
    metadata           jsonb NOT NULL DEFAULT '{}',
    taxonomy_version   integer,
    source             text NOT NULL DEFAULT 'historical',
    status             text NOT NULL DEFAULT 'active' CHECK (status IN ('active','archived')),
    created_at         timestamptz NOT NULL DEFAULT now(),
    search_text        tsvector GENERATED ALWAYS AS (to_tsvector('english', complaint_text)) STORED
);
CREATE INDEX tickets_search_idx ON tickets USING gin (search_text);
CREATE INDEX tickets_intent_idx ON tickets (intent);
CREATE INDEX tickets_product_idx ON tickets (product);
CREATE INDEX tickets_resolved_idx ON tickets (resolved_at DESC);

CREATE TABLE knowledge_articles (
    id          bigserial PRIMARY KEY,
    article_id  text NOT NULL UNIQUE,
    title       text NOT NULL,
    content     text NOT NULL,
    steps       jsonb NOT NULL DEFAULT '[]',
    category    text NOT NULL,                  -- intent label
    product     text NOT NULL,
    tags        text[] NOT NULL DEFAULT '{}',
    metadata    jsonb NOT NULL DEFAULT '{}',
    status      text NOT NULL DEFAULT 'active' CHECK (status IN ('active','deprecated')),
    deprecated_reason text,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    search_text tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED
);
CREATE INDEX articles_search_idx ON knowledge_articles USING gin (search_text);
CREATE INDEX articles_category_idx ON knowledge_articles (category);
CREATE INDEX articles_product_idx ON knowledge_articles (product);

-- Embeddings live in their own tables keyed by (row, model) so a new embedding model can be
-- backfilled side-by-side and switched with a config change (see docs/production.md).
CREATE TABLE ticket_embeddings (
    ticket_pk  bigint NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    model      text NOT NULL,
    embedding  vector({{EMBEDDING_DIM}}) NOT NULL,
    PRIMARY KEY (ticket_pk, model)
);
CREATE INDEX ticket_emb_hnsw ON ticket_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE kb_embeddings (
    article_pk bigint NOT NULL REFERENCES knowledge_articles(id) ON DELETE CASCADE,
    model      text NOT NULL,
    embedding  vector({{EMBEDDING_DIM}}) NOT NULL,
    PRIMARY KEY (article_pk, model)
);
CREATE INDEX kb_emb_hnsw ON kb_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE resolution_requests (
    request_id     uuid PRIMARY KEY,
    trace_id       text,
    complaint      text NOT NULL,               -- redacted
    classification jsonb,
    retrieved      jsonb,
    result         jsonb,
    status         text NOT NULL,
    confidence     real,
    latency_ms     integer,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX resolution_requests_created_idx ON resolution_requests (created_at DESC);

CREATE TABLE feedback (
    feedback_id      bigserial PRIMARY KEY,
    request_id       uuid REFERENCES resolution_requests(request_id) ON DELETE SET NULL,
    rating           text NOT NULL CHECK (rating IN ('helpful','not_helpful')),
    comment          text,
    corrected_intent text,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE jobs (
    job_id      uuid PRIMARY KEY,
    kind        text NOT NULL,
    status      text NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
    payload     jsonb,
    result      jsonb,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE TABLE evaluation_runs (
    run_id     uuid PRIMARY KEY,
    suites     text[] NOT NULL,
    results    jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Key/value metadata; corpus_version is bumped on every ingestion and keys the response cache.
CREATE TABLE meta (key text PRIMARY KEY, value text NOT NULL);
INSERT INTO meta VALUES ('corpus_version', '1');
