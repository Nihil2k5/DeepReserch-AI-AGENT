-- Durable research runs, node checkpoints, span tree and event log.
-- Applied in filename order by deep_research.db.Database.migrate().

CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    question          TEXT        NOT NULL,
    focus             TEXT,
    status            TEXT        NOT NULL DEFAULT 'queued',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ,
    worker_id         TEXT,
    attempt           INTEGER     NOT NULL DEFAULT 0,
    error             TEXT,
    cancel_requested  BOOLEAN     NOT NULL DEFAULT FALSE,
    report            JSONB,
    -- usage / budget accounting
    llm_calls         INTEGER     NOT NULL DEFAULT 0,
    searches          INTEGER     NOT NULL DEFAULT 0,
    tokens_in         BIGINT      NOT NULL DEFAULT 0,
    tokens_out        BIGINT      NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(12,6) NOT NULL DEFAULT 0,
    -- execution stats
    duration_ms       DOUBLE PRECISION,
    nodes_executed    INTEGER     NOT NULL DEFAULT 0,
    nodes_resumed     INTEGER     NOT NULL DEFAULT 0
);

-- Job-queue claim path: oldest queued run first.
CREATE INDEX IF NOT EXISTS runs_queue_idx ON runs (status, created_at);

-- Resumable node outputs. A run can be picked up by another worker after a
-- crash: completed nodes are skipped and their outputs replayed from here.
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id      TEXT        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    node_key    TEXT        NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'done',
    output      JSONB,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, node_key)
);

-- Span tree per run: node -> llm/tool children, with timings and attributes.
CREATE TABLE IF NOT EXISTS spans (
    id           BIGSERIAL PRIMARY KEY,
    span_id      TEXT        NOT NULL,
    run_id       TEXT        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    parent_id    TEXT,
    name         TEXT        NOT NULL,
    kind         TEXT        NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL,
    ended_at     TIMESTAMPTZ,
    duration_ms  DOUBLE PRECISION,
    status       TEXT        NOT NULL DEFAULT 'ok',
    attributes   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    error        TEXT
);

CREATE INDEX IF NOT EXISTS spans_run_idx ON spans (run_id, started_at);
CREATE INDEX IF NOT EXISTS spans_parent_idx ON spans (run_id, parent_id);

-- Append-only progress log; the SSE endpoint tails this by id.
CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    level       TEXT        NOT NULL DEFAULT 'info',
    type        TEXT        NOT NULL,
    message     TEXT        NOT NULL DEFAULT '',
    data        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS events_run_idx ON events (run_id, id);
