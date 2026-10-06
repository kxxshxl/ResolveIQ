-- Drift analyses: one row per run of the drift monitor (worker job `drift_analysis`), kept as the history the UI and alerts read.
-- `report` is the full explainable result (distributions, tests, effect sizes, emerging clusters); the scalar columns are for listing.
CREATE TABLE drift_snapshots (
    snapshot_id       uuid PRIMARY KEY,
    created_at        timestamptz NOT NULL DEFAULT now(),
    window_hours      integer NOT NULL,
    baseline_days     integer NOT NULL,
    status            text NOT NULL CHECK (status IN ('ok','alert','insufficient_data')),
    n_recent          integer NOT NULL,
    n_baseline        integer NOT NULL,
    alert_count       integer NOT NULL DEFAULT 0,
    emerging_clusters integer NOT NULL DEFAULT 0,
    job_id            uuid,
    report            jsonb NOT NULL
);
CREATE INDEX drift_snapshots_created_idx ON drift_snapshots (created_at DESC);
