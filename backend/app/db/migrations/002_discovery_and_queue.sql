-- Emerging-class discovery proposals + Postgres-backed job queue columns (worker service).

-- A proposal is a cluster of recent complaints that existing classes explain poorly. It is advice for a human
-- reviewer, never applied automatically: accepting it creates a taxonomy label (or extends an existing one).
CREATE TABLE taxonomy_proposals (
    proposal_id         uuid PRIMARY KEY,
    status              text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','accepted','rejected','superseded')),
    recommendation      text NOT NULL CHECK (recommendation IN ('new_class','extend_existing')),
    label_id            text NOT NULL,
    description         text NOT NULL,
    keywords            text[] NOT NULL DEFAULT '{}',
    examples            text[] NOT NULL DEFAULT '{}',
    product             text,
    team                text,
    size                integer NOT NULL,
    cohesion            real NOT NULL,
    mean_evidence       real,
    nearest_intent      text,
    neighbor_agreement  real,
    member_request_ids  uuid[] NOT NULL DEFAULT '{}',
    job_id              uuid,
    decided_note        text,
    created_at          timestamptz NOT NULL DEFAULT now(),
    decided_at          timestamptz
);
CREATE INDEX taxonomy_proposals_status_idx ON taxonomy_proposals (status, created_at DESC);

-- Queue semantics for the worker: claim with FOR UPDATE SKIP LOCKED, retry with backoff, recover stuck jobs.
ALTER TABLE jobs ADD COLUMN attempts     integer     NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN max_attempts integer     NOT NULL DEFAULT 3;
ALTER TABLE jobs ADD COLUMN run_after    timestamptz NOT NULL DEFAULT now();
ALTER TABLE jobs ADD COLUMN locked_by    text;
ALTER TABLE jobs ADD COLUMN locked_at    timestamptz;
ALTER TABLE jobs ADD COLUMN started_at   timestamptz;
CREATE INDEX jobs_claim_idx ON jobs (run_after) WHERE status = 'queued';

CREATE INDEX resolution_requests_status_idx ON resolution_requests (status, created_at DESC);
