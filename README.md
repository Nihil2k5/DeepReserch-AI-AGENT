# Deep Research Agent

A durable, multi-agent web research pipeline. Give it a question; it decomposes
it, fans out across the web in parallel, **fact-checks its own evidence**, and
produces a cited report with an explicit list of what it could *not* establish.

The parts that make it a system rather than a demo:

| Concern | How this repo handles it |
| --- | --- |
| Long runs die half-way | Every node checkpoints to Postgres; a crashed run resumes instead of restarting |
| Agents hallucinate | A verifier node audits each claim against its own quoted source and drops failures |
| Runaway cost | Hard budgets on LLM calls, searches, dollars and wall clock, enforced *before* each call |
| No idea what happened | A span tree per run (node → llm/tool), plus an append-only event log |
| "Is it any good?" | An evaluation harness with deterministic metrics **and** an LLM judge |
| One worker can't keep up | Queue runs are claimed with `FOR UPDATE SKIP LOCKED`; scale with more workers |

> **This service requires real infrastructure.** Postgres and Redis are always
> needed; the LLM and search are pluggable. The cheapest fully-local setup is
> **Ollama (`llama3.2:3b`) + headless-Chrome scraping** — no API keys at all. Or
> use an OpenAI-compatible endpoint plus an [Exa](https://exa.ai) API key for
> production-quality research.

---

## Architecture

```
                     ┌────────────────────────────────────────────────────┐
   POST /runs  ──────►  runs (queued)  ──► worker claims with            │
                     │                      FOR UPDATE SKIP LOCKED       │
                     └────────────────────────────────────────────────────┘
                                        │
   ┌────────────────────────────────────▼─────────────────────────────────┐
   │  Durable graph (every node writes a checkpoint key)                  │
   │                                                                      │
   │   plan ──► research.round{N} ──► verify ──┬─► (gaps? next round)     │
   │                 │  fan-out, thread pool   │                          │
   │                 │  per sub-question       │                          │
   │                 ├─ research.queries (LLM) │                          │
   │                 ├─ search (Exa)           │                          │
   │                 └─ research.extract (LLM) │                          │
   │                                           ▼                          │
   │                          synthesize ──► critique                     │
   └──────────────────────────────────────┬───────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼──────────────────────────────┐
        ▼                                 ▼                              ▼
   Postgres: runs,              span tree (node→llm→tool)      Langfuse (optional
   checkpoints, spans,          + append-only event log        hosted trace UI)
   events, usage
```

### The graph

| Node | Responsibility | Model call |
| --- | --- | --- |
| `plan` | Decompose the question into 3-6 independent sub-questions | reasoning model |
| `research:{sq}` | Generate queries, search, extract claims **with verbatim quotes** | cheap model ×2 |
| `verify:{sq}` | Audit every claim against its own quote; report gaps + follow-up queries | cheap model |
| `synthesize` | Write the cited report from **verified evidence only** | reasoning model |
| `critique` | Score coverage, flag unsupported claims | cheap model |

Research fans out across a thread pool so independent sub-questions overlap, and
a follow-up round is scheduled automatically when the verifier reports gaps —
using the follow-up queries the verifier itself proposed.

---

## Durability: why checkpoints

Every node writes its output to `checkpoints` under a deterministic key
(`plan`, `research:q2`, `verify:q2:r1`, `synthesize`, `critique`). Executing a run
means walking the plan and **skipping any key that already exists**.

That single rule buys three things:

1. **Crash recovery.** Kill a worker mid-run; the stale-run sweep flips the run
   back to `queued`, another worker claims it and resumes from the last
   checkpoint. Verified in the test suite: a full resume re-issues **zero** model
   calls and replays every node.
2. **Inspectability.** `SELECT node_key FROM checkpoints WHERE run_id = ...` is a
   progress bar. A resumed run's remaining work is visible.
3. **Replayable debugging.** Re-running a finished run is free and deterministic,
   which is what makes the eval harness affordable to iterate on.

Node outputs are the *only* state — there is no in-memory graph object to drift
out of sync with the database.

---

## Setup

### 1. Infrastructure

```bash
docker compose up -d db cache      # Postgres 16 + Redis 7
```

Or point `DR_DATABASE_URL` / `DR_REDIS_URL` at your own instances.

### 2. Credentials

| Variable | Where to get it |
| --- | --- |
| `DR_LLM_API_KEY` | Any OpenAI-compatible endpoint (OpenAI, Azure, Groq, Together, Ollama, vLLM) |
| `DR_EXA_API_KEY` | [dashboard.exa.ai](https://dashboard.exa.ai) — free credits on signup |
| `DR_LANGFUSE_*` | Optional. [cloud.langfuse.com](https://cloud.langfuse.com) or self-hosted |

```bash
cp .env.example .env      # then fill in the two keys
```

Missing configuration fails fast with one actionable message naming every
unset variable, rather than a stack trace from deep inside a run.

### 3. Install and run

```bash
pip install -e ".[dev,observability]"

python -m deep_research            # API on :8100
research-worker                     # separate process, scale with --concurrency

research "How do post-quantum signature schemes differ?" --out report.md
```

Try the same question through the queue so it streams progress:

```bash
research "..." --queue             # prints the run id
curl -N localhost:8100/runs/<id>/events/stream
```

---

## Running fully local (zero API keys)

The whole pipeline — LLM included — runs on your laptop with no accounts and
no cost: **Ollama supplies the model, headless Chrome replaces the search API.**
Postgres + Redis still come from `docker compose up -d db cache`.

```bash
# 1. Model + browser driver
ollama pull llama3.2:3b
pip install -e ".[dev,observability,browser]"

# 2. Point the two providers at local infrastructure (.env)
DR_LLM_PROVIDER=ollama
DR_LLM_BASE_URL=http://localhost:11434/v1
DR_LLM_MODEL=llama3.2:3b
DR_LLM_REASONING_MODEL=llama3.2:3b
DR_SEARCH_PROVIDER=browser

# 3. Start the stack and open the dashboard
python -m deep_research &            # API on :8100
research-worker &                    # executes queued runs
open http://localhost:8100/ui/       # question in, live pipeline, report out
```

Nothing else changes: same graph, same checkpoints, same budget enforcement,
same span tree. `cost_usd` stays at 0.00 because local tokens are priced at
zero.

How the browser provider works (`tools/browser_search.py`): Playwright drives
your **installed Chrome** (`channel="chrome"`, so there is no browser download),
scrapes the DuckDuckGo HTML results page, then visits the top hits and extracts
readable text (`main`/`article` preferred, scripts/images/fonts blocked). It is
deliberately the "hard way" — the honest version of what Exa does for you — and
swapping back to Exa is one env var.

Two engineering notes you'd otherwise rediscover the hard way:

- **Playwright's sync API is thread-affine.** The research fan-out runs on a
  `ThreadPoolExecutor`, and Playwright objects refuse calls from any thread but
  their own ("Cannot switch to a different thread"). All scraping is funneled
  through one dedicated thread.
- **A dead driver must be abandoned, not closed.** When Chrome's transport dies
  mid-run, `close()` on the dead connection can hang forever. The tool detects
  `connection closed`, leaks the old thread on purpose, rebuilds a fresh pool +
  browser, and retries once. Combined with checkpointing, a crashed browser
  costs one query, not the run.

Honest limits of local mode: a 3B model writes simpler synthesis and makes more
formatting slips than a hosted frontier model (the synthesizer's one-shot
citation-repair pass covers the most common one), DuckDuckGo scraping is slower
and flakier than a search API, and long runs run at CPU speed on machines
without a discrete GPU. Quality, not correctness, is what you trade away — the
durability, verification and budget machinery are identical.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/runs` | Queue a run (returns `202` + run id) |
| `GET` | `/runs` | List runs, filter by `status` |
| `GET` | `/runs/{id}` | Run detail + completed checkpoint keys + span count |
| `GET` | `/runs/{id}/report` | Report as JSON, or `?format=markdown` |
| `GET` | `/runs/{id}/spans` | Full span tree |
| `GET` | `/runs/{id}/events` | Event log, cursor-paginated via `after_id` |
| `GET` | `/runs/{id}/events/stream` | SSE progress tail (works across workers) |
| `POST` | `/runs/{id}/cancel` | Request cancellation |
| `POST` | `/runs/{id}/execute` | Run synchronously (demos and tests) |
| `GET` | `/health`, `/metrics` | Dependency health; Prometheus text |

Set `DR_API_KEY` to require an `X-API-Key` header on mutating routes.

### Dashboard UI

`http://localhost:8100/ui/` ( `/` redirects there) serves a single-page dashboard:
submit a question, watch the pipeline light up stage-by-stage over SSE
(plan → research ×N → verify → synthesize → critique) with a live event feed,
then read the rendered report with clickable source links. Static assets live in
`src/deep_research/api/static/` and are served same-origin by FastAPI — no build
step, no CORS.

---

## Evaluation

```bash
research-eval --dataset eval/dataset.jsonl --limit 2
```

Two layers, deliberately separate:

**Deterministic** (`evaluation/metrics.py`) — no model needed, so it can gate CI:

| Metric | Question it answers |
| --- | --- |
| `citation_validity` | Does every `[n]` marker resolve to a real source? (should be 1.0) |
| `citation_density` | What fraction of sentences are sourced? |
| `source_diversity` | How many distinct domains were used? |
| `quality_score` | Single 0-1 roll-up for regression tracking |

**Judged** (`evaluation/judge.py`) — `fact_coverage`: for each required fact,
does the report actually establish it? Graded semantically by a *different,
cheaper* model than the synthesizer, because grading your own homework with the
same prompt is how eval numbers stop meaning anything. Unmentioned facts always
count against coverage, so a malformed judge response can only understate.

Every case also records cost, tokens and duration, so a quality gain can be
weighed against what it cost.

---

## Observability

The span tree is stored in Postgres **and** mirrored to Langfuse when keys are
set. Here is a real trace from a 3-sub-question run:

```
  plan                                  0.68ms  [node]
    plan.decompose                        0.04ms  [llm]
  research.round0                       5.98ms  [node]
    research                              0.96ms  [node]
      research.queries                      0.06ms  [llm]
      search:alpha storage round-trip efficiency  0.02ms  [tool]
      research.extract                      0.04ms  [llm]
    research                              0.73ms  [node]
      ...
  verify                                0.32ms  [node]
    verify.check                          0.03ms  [llm]
  synthesize                            0.69ms  [node]
    synthesize.report                     0.03ms  [llm]
  critique                              0.27ms  [node]
    critique.review                       0.04ms  [llm]
```

### How the Langfuse mirroring works

Langfuse 3+ is built on OpenTelemetry, which propagates parent context through
**contextvars**. Those do not cross a thread boundary, so the fan-out spans —
opened on worker threads — would each land as a new root and the tree would
flatten. The exporter therefore:

- assigns one trace per run via `create_trace_id(seed=run_id)`;
- opens a **single root observation per run**, named after the question, so a run
  renders as one tree instead of six;
- passes each child the `parent_span_id` of its parent observation **explicitly**,
  which works across threads;
- maps span kinds onto observation types (`node`→span, `llm`→generation,
  `tool`→tool) and attaches token/cost usage to the generation spans;
- closes the root in a `finally` on both success and failure paths.

Two practical notes:

- **Credentials are checked once at startup.** A `401` from Langfuse is logged as
  an error and surfaced in `/health` as `langfuse.authenticated = false`, because
  a silently disabled exporter is the worst failure mode: you believe you have
  traces and you don't.
- **The OTel exporter is process-global.** The first `Langfuse` client in a
  process fixes the export target, so you cannot point two exporters at different
  hosts within one process. One host per process, which is the normal case.

Export failures never break a run: the first error disables the exporter with one
warning, and the Postgres span table remains the authoritative record.

### Troubleshooting a 401

`Invalid credentials. Confirm that you've configured the correct host.` almost
always means the keys belong to a different region than the host:

| Region | Host |
| --- | --- |
| EU | `https://cloud.langfuse.com` |
| US | `https://us.cloud.langfuse.com` |
| JP | `https://jp.cloud.langfuse.com` |

Verify a key pair directly:

```bash
python -c "import langfuse,os; print(langfuse.Langfuse().auth_check())"
# needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY exported
```

---

## Design decisions

- **Verify, don't just summarize.** The extractor is forced to emit a verbatim
  `quote` alongside every claim, and the verifier's only job is to attack those
  pairs. Rejected evidence never reaches the synthesizer. This is the difference
  between a research system and a fluent summarizer.
- **We assign citation numbers, not the model.** Sources are numbered before
  synthesis and markers referencing numbers we never issued are stripped
  afterwards. A report cannot cite a source that doesn't exist.
- **Budgets are enforced before the call.** Checking after means you've already
  paid. And when the research budget is exhausted mid-run, synthesis is granted
  headroom: a partial report plus a stated limitation beats no report.
- **Postgres as the queue.** `FOR UPDATE SKIP LOCKED` gives safe multi-worker
  claiming with no extra service to run. Redis is used only for per-run locks and
  outbound rate limiting, and the coordinator fails *open* — a Redis outage
  degrades throughput, not correctness.
- **Exa's REST API over httpx, not the SDK.** One fewer pinned dependency, an
  explicit retry policy, and clean extracted page text (which the extractor can
  quote directly, unlike a SERP API returning links).
- **A `browser` search provider that does it the hard way.** Headless Chrome
  (Playwright, your installed Chrome — no download) scrapes DuckDuckGo and pulls
  readable page text, so the pipeline runs with zero API keys. Playwright's sync
  driver is thread-affine, so all scraping is funneled through one dedicated
  thread, and a crashed driver is detected and rebuilt rather than hanging the
  worker. Swapping in Exa is one env var — which is the point.
- **Offline-capable tests, not offline production.** Fake providers exist so all
  91 unit tests run in ~2s with no infrastructure — but they are never selected
  by default, and CI additionally runs real-Postgres integration tests covering
  the schema and the concurrent claim.

---

## Testing

```bash
pytest -m "not integration"     # 91 tests, no infrastructure, ~2s
ruff check src tests

# needs DR_TEST_DATABASE_URL
pytest -m integration           # schema, SQL, SKIP LOCKED under 4 threads
```

Coverage: budget enforcement (pre-call, cost, wall clock, headroom), JSON
extraction from messy model output, citation sanitation and markdown rendering,
state derivation from checkpoints, the full graph with stubbed providers,
**resume semantics** (a second execution re-issues zero model calls), partial
resume, cancellation mid-run, budget exhaustion producing a partial report,
evidence rejection propagating into the report, span-tree nesting, every API
route including SSE and auth, and the in-memory repository contract.

### Verification status

Honest accounting of what has actually been run:

- ✅ **96 unit tests pass**; all 47 modules import; `ruff` clean
- ✅ Full graph executed end-to-end (9 nodes, 12 LLM calls, 3 searches, 9
  checkpoints, full span tree) using **stubbed providers**
- ✅ **Langfuse export verified for real** against a local OTLP capture server:
  one trace, one root named after the question, all six stages parented to it,
  the two fan-out research spans nested under their round's stage span with six
  grandchildren, LLM spans typed as `generation` carrying `usage_details` and
  model name, search spans typed as `tool`
- ✅ Export failure paths verified: a dead host and a mid-run crash both leave the
  run successful and no root span dangling
- ✅ Fail-fast config validation produces a single actionable error
- ⚠️ **The Langfuse credentials supplied for this build were rejected (HTTP 401)
  on the EU, US and JP hosts**, so no trace has been delivered to a live Langfuse
  project. The export path is verified against the captured wire format instead.
  See *Troubleshooting a 401* above.
- ⚠️ **Not yet executed against the live Exa/LLM APIs or Postgres** — that needs
  real credentials and a running database (`docker compose up -d db cache`). The
  six integration tests exist and are wired into CI, but have not been run here.
- ✅ **Fully-local mode executed for real** (Ollama `llama3.2:3b` + headless-Chrome
  scraping against live websites, real Postgres + Redis): multiple end-to-end
  runs produced multi-section cited reports at $0.00, checkpoint resume was
  exercised by restarting the worker mid-run (`nodes_resumed > 0`, no repeated
  LLM calls), and a crashed Chrome driver was recovered in-run. Two production
  bugs were found and fixed this way: Playwright sync thread-affinity breaking
  the research fan-out, and a dead driver transport hanging `close()` forever.
- ✅ **`llama3.2:3b` benchmarked on the pipeline's own prompts**
  (`benchmarks/llama32_bench.py`): plan / query-writing / verbatim-quote
  extraction / strict verification all pass; the synthesizer's one-shot citation
  repair covers the model's one observed formatting failure mode.

---

## Project layout

```
src/deep_research/
├── graph/
│   ├── engine.py           Durable runner: resume, fan-out, budgets, cancellation
│   ├── state.py            Read view over checkpoints (the only source of state)
│   ├── context.py          Budgets + tracing + usage accounting for every call
│   └── nodes/              planner · researcher · verifier · synthesizer · critic
├── api/                    FastAPI app, routes, dependencies
├── evaluation/             Metrics, LLM judge, dataset, harness, CLI
├── llm/                    OpenAI-compatible client (OpenAI, Ollama, …) · fake (tests)
├── tools/                  Exa search · headless-Chrome scraper · fake (tests)
├── repository.py           Postgres + in-memory persistence
├── db.py                   Connection pool + migrations
├── worker.py               Queue consumer with graceful shutdown
├── budget.py               Pre-call budget enforcement
├── tracing.py              Span tracer + Langfuse exporter
├── reporting.py            Citation sanitation + markdown rendering
├── locking.py              Redis locks and rate limiting
└── config.py               Fail-fast, env-driven settings
```

---

## Roadmap

- Rerank search results with a cross-encoder before extraction
- Persistent evidence cache keyed by URL, to avoid re-reading the same page
- Human-in-the-loop approval between research and synthesis
- Export reports to PDF/DOCX with footnotes
- Per-tenant quotas and cost attribution
