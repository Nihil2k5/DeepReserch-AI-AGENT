"""Benchmark llama3.2:3b against the pipeline's actual node prompts.

Grades outputs exactly the way the nodes do:
- planner: extract_json -> sub_questions list of {id, question}
- queries: extract_json -> queries list
- extract: extract_json -> evidence items with claim/quote/source, quote must be
  verbatim in the given source text, source index valid, confidence present
- verify:  extract_json -> supported/unsupported id lists

Run:  python benchmarks/llama32_bench.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_research.structured import extract_json  # noqa: E402

MODEL = "llama3.2:3b"
URL = "http://localhost:11434/v1/chat/completions"

# --- real scraped content (from a live run of the pipeline) -----------------
SOURCES = [
    (
        "What is RAG?",
        "Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources. "
        "In other words, it fills a model's knowledge gaps with up-to-date, verifiable facts rather than relying solely on its training data. "
        "A 2023 study by researchers at Stanford found RAG reduced hallucinations by up to 60 percent in knowledge-intensive tasks. "
        "The global RAG market was valued at $1,042.7 million in 2024 and is expected to grow at a compound annual growth rate (CAGR) of 44.7% through 2030.",
    ),
    (
        "Agentic vs Traditional RAG",
        "Agentic RAG integrates AI agents into the RAG pipeline, enabling autonomous reasoning, tool usage, and multi-step decision making. "
        "Traditional RAG retrieves once and answers from whatever it got, even if that's noise. Agentic RAG can look at what it retrieved, decide it's not good enough, and go again. "
        "Agentic RAG systems routinely route queries to a different source, rewrite the query, or escalate to a human. "
        "Consider deploying agentic RAG when queries are complex and multi-hop reasoning is required.",
    ),
]


def call(system: str, user: str, max_tokens: int = 400) -> tuple[str, float]:
    """One chat completion; returns (text, elapsed_seconds)."""
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.load(resp)
    elapsed = time.perf_counter() - start
    return payload["choices"][0]["message"]["content"], elapsed


# --- tasks -------------------------------------------------------------------
def bench_plan() -> dict:
    system = (
        "You are a research planner. Decompose a research question into 2-4 "
        "independent, searchable sub-questions. Respond with JSON only, matching "
        'exactly this shape: {"sub_questions": [{"id": "q1", "question": "...", "rationale": "..."}]}'
    )
    user = "Research question:\nHow are vector databases evolving in 2026?\n\nProduce at most 4 sub-questions."
    text, elapsed = call(system, user)
    payload = extract_json(text, default={}) or {}
    items = payload.get("sub_questions") if isinstance(payload, dict) else None
    ok = isinstance(items, list) and len(items) >= 2 and all(
        isinstance(i, dict) and str(i.get("question") or "").strip() for i in items
    )
    return {"task": "plan", "ok": ok, "detail": f"{len(items) if isinstance(items, list) else 0} sub-questions", "elapsed": elapsed, "raw": text[:120]}


def bench_queries() -> dict:
    system = (
        "You write web search queries for a research agent.\n\nRules:\n"
        "- Produce 2-3 queries, each phrased as a real search someone would type.\n"
        "- Vary them: one broad, one specific to data/numbers, one targeting authoritative sources.\n"
        "- Do not use boolean operators, quotes or site: filters.\n"
        '- Each query must stand alone (no pronouns referring to the original question).\n\n'
        'Respond with JSON only: {"queries": ["...", "..."]}'
    )
    text, elapsed = call(system, "Sub-question: What is the market size of vector databases?")
    payload = extract_json(text, default={}) or {}
    queries = payload.get("queries") if isinstance(payload, dict) else None
    ok = isinstance(queries, list) and 2 <= len(queries) <= 3 and all(
        isinstance(q, str) and 3 <= len(q.split()) <= 12 and '"' not in q for q in queries
    )
    return {"task": "queries", "ok": ok, "detail": "; ".join(queries) if isinstance(queries, list) else "bad", "elapsed": elapsed, "raw": text[:120]}


def bench_extract() -> dict:
    system = (
        "You extract verifiable evidence from web sources.\n\nFor each source, extract claims that help answer the sub-question.\n\nRules:\n"
        "- Every claim MUST come with an exact quote copied verbatim from that source. Never paraphrase inside the quote field.\n"
        "- One claim per distinct fact. Prefer numbers, dates, names and concrete statements.\n"
        '- Set "source" to the bracketed number of the source the quote came from.\n'
        '- "confidence" is 0-1 for how directly the quote supports the claim.\n'
        "- Extract at most 6 claims. If a source has nothing relevant, skip it.\n"
        "- Never invent facts that are not present in the sources.\n\n"
        'Respond with JSON only:\n{"evidence": [{"claim": "...", "quote": "...", "source": 1, "confidence": 0.9}]}'
    )
    sources_block = "\n\n".join(
        f"[{i}] {t}\nURL: http://example.com/{i}\n{txt[:900]}" for i, (t, txt) in enumerate(SOURCES, start=1)
    )
    user = f"Sub-question: What is the market size of RAG?\n\nSources:\n{sources_block}"
    text, elapsed = call(system, user, max_tokens=900)

    payload = extract_json(text, default={}) or {}
    items = payload.get("evidence") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        return {"task": "extract", "ok": False, "detail": "no evidence list", "elapsed": elapsed, "raw": text[:200]}

    valid = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "")
        src = item.get("source")
        if not isinstance(src, int) or not (1 <= src <= len(SOURCES)):
            continue
        # The node's rule: the quote must appear verbatim in that source text.
        if quote.strip() and quote.strip() in SOURCES[src - 1][1]:
            valid += 1
    return {
        "task": "extract",
        "ok": valid >= 2,
        "detail": f"{valid}/{len(items)} quotes verbatim+valid source",
        "elapsed": elapsed,
        "raw": text[:200],
    }


def bench_verify() -> dict:
    system = (
        "You are a strict fact-checker. Audit research evidence against the sub-question.\n\nRules:\n"
        "- supported: the quote actually supports the claim\n"
        "- unsupported: the quote does not support the claim (paraphrased quote, unrelated, or invented)\n"
        "- If every item is fine, list them all as supported.\n\n"
        'Respond with JSON only: {"supported": ["e1"], "unsupported": [], "gaps": [], "follow_up_queries": ["..."]}'
    )
    items = (
        "id: e1\nclaim: The RAG market is expected to reach $5 billion by 2030.\nquote: valued at $1,042.7 million in 2024 and is expected to grow at a CAGR of 44.7% through 2030\nsource: http://example.com/1\n\n"
        "id: e2\nclaim: RAG eliminates hallucinations completely.\nquote: it fills a model's knowledge gaps with up-to-date, verifiable facts\nsource: http://example.com/1"
    )
    user = f"Sub-question: What is the market size of RAG?\n\nEvidence to audit:\n{items}"
    text, elapsed = call(system, user)
    payload = extract_json(text, default={}) or {}
    sup = payload.get("supported") if isinstance(payload, dict) else None
    unsup = payload.get("unsupported") if isinstance(payload, dict) else None
    ok = (
        isinstance(sup, list)
        and isinstance(unsup, list)
        and "e2" in unsup  # the overclaiming item must be flagged
        and "e1" in sup
    )
    return {"task": "verify", "ok": ok, "detail": f"supported={sup} unsupported={unsup}", "elapsed": elapsed, "raw": text[:120]}


def main() -> None:
    tasks = [bench_plan, bench_queries, bench_extract, bench_verify]
    results = [fn() for fn in tasks]
    print(f"\n=== {MODEL} on real pipeline prompts ===")
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"[{mark}] {r['task']:8s} {r['elapsed']:6.1f}s  {r['detail']}")
        if not r["ok"]:
            print(f"       raw: {r['raw']!r}")
    print(f"\ntotal: {sum(r['elapsed'] for r in results):.1f}s")


if __name__ == "__main__":
    main()
