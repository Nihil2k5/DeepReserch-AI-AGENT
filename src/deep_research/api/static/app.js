"use strict";

// --- tiny DOM helpers -------------------------------------------------------
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

// --- state ------------------------------------------------------------------
let currentRunId = null;
let eventSource = null;
let currentReport = null;
let startedAt = null;
let tickerId = null;

// --- health -----------------------------------------------------------------
async function checkHealth() {
  const badge = $("health");
  try {
    const res = await fetch("/health");
    const body = await res.json();
    badge.textContent = body.database ? "db ok" : "db down";
    badge.className = "badge " + (body.database ? "ok" : "bad");
  } catch {
    badge.textContent = "api unreachable";
    badge.className = "badge bad";
  }
}

// --- run form ---------------------------------------------------------------
$("ask-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const error = $("error");
  error.hidden = true;

  const question = $("question").value.trim();
  const focus = $("focus").value.trim();
  if (question.length < 8) {
    error.textContent = "Question must be at least 8 characters.";
    error.hidden = false;
    return;
  }

  const btn = $("ask-btn");
  btn.disabled = true;
  btn.textContent = "Queuing…";

  try {
    const res = await fetch("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, focus: focus || null }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || body.error || `HTTP ${res.status}`);
    }
    const body = await res.json();
    startRun(body.run_id, question);
  } catch (err) {
    error.textContent = "Failed to queue run: " + err.message;
    error.hidden = false;
    btn.disabled = false;
    btn.textContent = "Start research run";
  }
});

// Example chips fill the textarea.
document.querySelectorAll("button.example").forEach((b) =>
  b.addEventListener("click", () => {
    $("question").value = b.textContent;
    $("question").focus();
  })
);

// --- run lifecycle ----------------------------------------------------------
function startRun(runId, question) {
  currentRunId = runId;
  currentReport = null;
  startedAt = Date.now();
  startTicker();

  $("feed").replaceChildren();
  $("run-error").hidden = true;
  $("pipeline-card").hidden = false;
  $("metrics").hidden = false;
  $("report").hidden = true;
  $("report-empty").hidden = false;
  $("report-empty").textContent = "Researching… report appears when the run finishes.";
  $("copy-btn").hidden = true;
  $("cancel-btn").hidden = false;

  resetStages();
  $("run-meta").textContent = `run ${runId}`;
  $("m-status").textContent = "queued";
  $("m-status").className = "pill queued";
  setMetrics({ llm_calls: 0, searches: 0, cost_usd: 0 });
  feedRow({ type: "run.queued", message: "Run queued — waiting for a worker", data: {} });

  loadRunList();
  openStream(runId);
}

function resetStages() {
  document.querySelectorAll(".stage").forEach((s) => (s.className = "stage"));
}

function openStream(runId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/runs/${runId}/events/stream`);

  eventSource.addEventListener("progress", (ev) => {
    const event = JSON.parse(ev.data);
    feedRow(event);
    onEvent(event);
  });

  eventSource.addEventListener("done", async (ev) => {
    const { status, report_ready } = JSON.parse(ev.data);
    eventSource.close();
    eventSource = null;
    stopTicker();
    $("cancel-btn").hidden = true;
    $("m-status").textContent = status;
    $("m-status").className = `pill ${status}`;

    // Settle final metrics from the run record.
    try {
      const detail = await (await fetch(`/runs/${runId}`)).json();
      setMetrics(detail.run.metrics);
      $("m-status").textContent = detail.run.status;
      $("m-status").className = `pill ${detail.run.status}`;
      if (detail.run.status === "failed") {
        $("run-error").textContent = detail.run.error || "The run failed.";
        $("run-error").hidden = false;
      }
    } catch { /* metrics stay as-is */ }

    if (report_ready) {
      loadReport(runId);
    } else {
      $("report-empty").textContent = `Run finished with status “${status}” — no report.`;
      feedRow({ type: "run.finished", message: `Run ${status}, no report`, data: {} });
    }
    $("ask-btn").disabled = false;
    $("ask-btn").textContent = "Start research run";
    loadRunList();
  });

  eventSource.onerror = () => {
    // EventSource auto-reconnects; the done handler cleans up.
  };
}

// --- metrics row ------------------------------------------------------------
function setMetrics(m) {
  if (!m) return;
  $("m-llm").textContent = m.llm_calls ?? 0;
  $("m-searches").textContent = m.searches ?? 0;
  $("m-cost").textContent = `$${Number(m.cost_usd ?? 0).toFixed(2)}`;
}

function startTicker() {
  stopTicker();
  tickerId = setInterval(() => {
    if (!startedAt) return;
    const s = Math.floor((Date.now() - startedAt) / 1000);
    $("m-elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 1000);
}

function stopTicker() {
  if (tickerId) clearInterval(tickerId);
  tickerId = null;
}

// --- feed + pipeline --------------------------------------------------------
// node_key is "plan", "research:q1", "verify:q1:r2", "synthesize", "critique".
function stageForEvent(event) {
  const key = (event.data && event.data.node_key) || "";
  const base = key.split(":")[0];
  return { plan: 1, research: 1, verify: 1, synthesize: 1, critique: 1 }[base] ? base : null;
}

function onEvent(event) {
  const stage = stageForEvent(event);
  if (!stage) return;
  const node = document.querySelector(`.stage[data-stage="${stage}"]`);
  if (!node) return;
  if (event.type === "node.started") {
    node.classList.remove("done");
    node.classList.add("active");
  }
  if (event.type === "node.completed") {
    node.classList.remove("active");
    node.classList.add("done");
  }
}

function feedRow(event) {
  const feed = $("feed");
  if (feed.querySelector(".muted")) feed.replaceChildren();
  const row = el("div", `row lvl-${event.level || "info"}`);
  const time = event.created_at ? new Date(event.created_at) : new Date();
  row.append(
    el("span", "t", time.toLocaleTimeString()),
    el("span", "type", event.type),
    el("span", "msg", event.message || ""),
  );
  const interesting = event.data && Object.keys(event.data).length &&
    !["node_key", "round_index"].every((k) => Object.keys(event.data).length === 1 && k in event.data);
  if (interesting) row.append(el("span", "data", JSON.stringify(event.data)));
  feed.append(row);
  feed.scrollTop = feed.scrollHeight;
}

// --- report -----------------------------------------------------------------
async function loadReport(runId) {
  try {
    const res = await fetch(`/runs/${runId}/report`);
    const body = await res.json();
    if (!body.report) {
      $("report-empty").textContent = `Run ${runId} finished with status “${body.status}” — no report.`;
      return;
    }
    renderReport(body.report);
  } catch (err) {
    $("report-empty").textContent = "Failed to load report: " + err.message;
  }
}

/** Render body text, turning [3]-style markers into clickable citation chips. */
function renderBody(text, citations) {
  const container = el("span");
  const byIndex = new Map((citations || []).map((c) => [c.index, c]));
  const parts = String(text).split(/(\[\d+\])/g);
  for (const part of parts) {
    const match = part.match(/^\[(\d+)\]$/);
    if (match && byIndex.has(Number(match[1]))) {
      const c = byIndex.get(Number(match[1]));
      const a = el("a", "citation-chip", match[1]);
      a.href = c.url;
      a.target = "_blank";
      a.rel = "noopener";
      a.title = c.title || c.url;
      container.append(a);
    } else {
      container.append(document.createTextNode(part));
    }
  }
  return container;
}

function renderReport(report) {
  currentReport = report;
  $("report-empty").hidden = true;
  $("report").hidden = false;
  $("copy-btn").hidden = false;

  $("report-question").textContent = report.question;

  const summary = $("report-summary");
  summary.replaceChildren();
  if (report.executive_summary) {
    summary.append(renderBody(report.executive_summary, report.citations));
  }

  const sections = $("report-sections");
  sections.replaceChildren();
  for (const s of report.sections || []) {
    sections.append(el("h4", "", s.heading));
    const p = el("p");
    p.append(renderBody(s.body, report.citations));
    sections.append(p);
  }

  const cites = $("report-citations");
  cites.replaceChildren();
  for (const c of report.citations || []) {
    const li = el("li");
    const a = el("a", "", `${c.title || c.url}`);
    a.href = c.url;
    a.target = "_blank";
    a.rel = "noopener";
    li.append(a);
    cites.append(li);
  }

  const limWrap = $("report-limitations-wrap");
  const lims = $("report-limitations");
  lims.replaceChildren();
  for (const l of report.limitations || []) lims.append(el("li", "", l));
  limWrap.hidden = !(report.limitations || []).length;

  const critique = $("report-critique");
  critique.replaceChildren();
  if (report.critique) {
    const c = report.critique;
    critique.append(
      el("span", "score", `Coverage ${(c.coverage_score * 100).toFixed(0)}%`),
      el("span", "", c.verdict ? ` — ${c.verdict}` : ""),
    );
    for (const i of c.improvements || []) critique.append(el("div", "", `• ${i}`));
    critique.hidden = false;
  } else {
    critique.hidden = true;
  }
}

// --- copy as markdown --------------------------------------------------------
$("copy-btn").addEventListener("click", async () => {
  if (!currentReport) return;
  const r = currentReport;
  const md = [
    `# ${r.question}`,
    "",
    r.executive_summary,
    "",
    ...r.sections.map((s) => `## ${s.heading}\n\n${s.body}`),
    "",
    "## Sources",
    ...(r.citations || []).map((c) => `- [${c.index}] ${c.title || ""} ${c.url}`),
  ].join("\n");
  try {
    await navigator.clipboard.writeText(md);
    $("copy-btn").textContent = "Copied!";
    setTimeout(() => ($("copy-btn").textContent = "Copy markdown"), 1500);
  } catch {
    $("copy-btn").textContent = "Copy failed";
  }
});

// --- cancel -------------------------------------------------------------------
$("cancel-btn").addEventListener("click", async () => {
  if (!currentRunId) return;
  $("cancel-btn").disabled = true;
  try {
    await fetch(`/runs/${currentRunId}/cancel`, { method: "POST" });
    feedRow({ type: "cancel.requested", message: "Cancellation requested…", data: {} });
  } finally {
    setTimeout(() => ($("cancel-btn").disabled = false), 1000);
  }
});

// --- run history ---------------------------------------------------------------
async function loadRunList() {
  try {
    const res = await fetch("/runs?limit=25");
    const runs = await res.json();
    if (!Array.isArray(runs)) return;
    const list = $("run-list");
    list.replaceChildren();
    if (!runs.length) {
      list.append(el("div", "muted", "No runs yet."));
      return;
    }
    for (const r of runs) {
      const item = el("div", "run-item" + (r.id === currentRunId ? " current" : ""));
      item.addEventListener("click", () => attach(r.id));
      item.append(
        el("div", "q", r.question),
        el(
          "div",
          "meta",
          el("span", `dot ${r.status}`),
          el("span", "", r.status),
          el("span", "", r.created_at ? new Date(r.created_at).toLocaleString() : ""),
          r.report ? el("span", "", "· report ✓") : "",
        ),
      );
      list.append(item);
    }
  } catch { /* leave whatever is there */ }
}

/** Attach the dashboard to any run (history click or deep link). */
async function attach(runId) {
  try {
    const res = await fetch(`/runs/${runId}`);
    if (!res.ok) return;
    const detail = await res.json();
    currentRunId = runId;
    currentReport = null;
    $("question").value = detail.run.question;
    $("pipeline-card").hidden = false;
    $("metrics").hidden = false;
    $("run-meta").textContent = `run ${runId}`;
    $("run-error").hidden = true;
    $("feed").replaceChildren();

    resetStages();
    for (const key of detail.checkpoints || []) {
      const chip = document.querySelector(`.stage[data-stage="${key.split(":")[0]}"]`);
      if (chip) chip.classList.add("done");
    }

    setMetrics(detail.run.metrics);
    $("m-status").textContent = detail.run.status;
    $("m-status").className = `pill ${detail.run.status}`;
    startedAt = detail.run.started_at ? new Date(detail.run.started_at).getTime() : null;
    if (detail.run.status === "running") {
      startTicker();
      $("cancel-btn").hidden = false;
      $("ask-btn").disabled = true;
      $("ask-btn").textContent = "Watching run…";
    } else {
      stopTicker();
      $("cancel-btn").hidden = true;
      $("ask-btn").disabled = false;
      $("ask-btn").textContent = "Start research run";
      if (detail.run.status === "failed" && detail.run.error) {
        $("run-error").textContent = detail.run.error;
        $("run-error").hidden = false;
      }
    }

    if (detail.run.report) renderReport(detail.run.report);
    else if (detail.run.status !== "running") {
      $("report-empty").hidden = false;
      $("report-empty").textContent = `Run ${detail.run.status} — no report.`;
    }

    loadRunList();
    if (detail.run.status === "running" || detail.run.status === "queued") openStream(runId);
  } catch { /* bad id: ignore */ }
}

// --- boot ----------------------------------------------------------------------
async function boot() {
  checkHealth();
  await loadRunList();
  const runId = new URLSearchParams(location.search).get("run");
  if (runId) {
    attach(runId);
  } else {
    // Show the most recent finished report so the page never looks dead.
    try {
      const runs = await (await fetch("/runs?limit=25")).json();
      const finished = Array.isArray(runs) && runs.find((r) => r.status === "succeeded" && r.report);
      if (finished) {
        currentRunId = finished.id;
        attach(finished.id);
      }
    } catch { /* empty dashboard is fine */ }
  }
  setInterval(checkHealth, 15000);
}

boot();
