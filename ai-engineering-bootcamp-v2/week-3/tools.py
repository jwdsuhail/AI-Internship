"""
The agent's one real tool: search_docs.

It is a thin HTTP client over the Session 2 capstone's GET /debug/retrieve, so the
Pinecone index, embeddings, and chunking all stay in week-1/main.py. Nothing about
retrieval is reimplemented here on purpose — the agent is a new brain on the existing body.

Two decisions matter for the agent loop:

1. The raw cosine scores are returned to the model, not just the text. The model cannot
   decide "that search failed, try different wording" unless it can see the search failed.
2. Failures come back as ordinary dicts, never exceptions. An uncaught exception kills the
   run; a returned error is an observation the model can read and route around.
"""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# Mirrors MIN_RELEVANCE_SCORE in week-1/main.py. Chunks below this are noise on this corpus
# (measured there: on-topic 0.38-0.70, noise 0.00-0.25). Reported to the model so it can judge
# its own search quality instead of guessing.
RELEVANCE_FLOOR = 0.30

# Render's free tier cold-starts slowly; a short timeout would look like a tool bug.
TIMEOUT = 60.0

# Hard budget on searches per run. The instruction asks the model for at most 3 wordings, but
# an instruction is a suggestion — in practice it ran 6, which costs ~7 LLM calls and trips
# Gemini's 5-per-minute free-tier limit before it can write an answer. Enforcing the budget in
# the tool makes it a fact the model observes rather than a rule it may ignore. It is a budget,
# not a kill switch: once spent, the tool tells the model to answer or refuse with what it has,
# so the run still ends in a real final answer instead of an aborted loop.
# A run costs one LLM call per search plus one to write the answer, so 3 searches = 4 calls
# against a free-tier limit of 5 per minute. That fits, but only just: drop this to 2 via env
# if runs keep dying at the rate limit before they can answer.
MAX_SEARCHES = int(os.environ.get("MAX_SEARCHES", "3"))

_searches_used = {"n": 0}


def reset_search_budget() -> None:
    """Start a fresh budget. Called once per run, before the agent's first turn."""

    _searches_used["n"] = 0


def search_docs(query: str) -> dict:
    """Search the company documentation and return matching passages with relevance scores.

    Use this for any question about company policies, onboarding, billing, security,
    shipping, support, or engineering practices. You cannot answer such questions from
    memory — the answer must come from the passages this tool returns.

    Read the scores in the result. If usable_chunks is 0, the search wording did not match
    anything; call this tool again with different wording (synonyms, the phrasing a policy
    document would actually use) rather than giving up on the first try.

    You get 3 searches per question. Every result includes searches_remaining — watch it. When
    it reaches 0 the tool stops searching and you must answer with what you already have, or
    say you do not have enough information.

    Args:
        query: The search phrase. Prefer the wording a policy document would use.

    Returns:
        A dict with usable_chunks (how many passages cleared the relevance floor),
        top_score, and passages (each with chunk_id, score, and text). On failure,
        a dict with an "error" key describing what went wrong.
    """

    if not query.strip():
        return {"error": "query was empty; supply a search phrase"}

    if _searches_used["n"] >= MAX_SEARCHES:
        return {
            "searches_used": _searches_used["n"],
            "searches_remaining": 0,
            "budget_exhausted": True,
            "instruction": (
                f"You have used all {MAX_SEARCHES} allowed searches. Do not call this tool "
                "again. Answer now using passages you already retrieved, or if none of them "
                'answer the question, reply exactly: "I don\'t have enough information to '
                'answer that."'
            ),
        }

    _searches_used["n"] += 1

    url = f"{API_BASE_URL.rstrip('/')}/debug/retrieve"

    try:
        response = httpx.get(url, params={"q": query, "k": 5}, timeout=TIMEOUT)
    except httpx.RequestError as exc:
        # Returned, not raised: the model should see this and report it, not crash the run.
        return {
            "error": f"Cannot reach the docs service at {API_BASE_URL} ({exc.__class__.__name__}). "
            "The FastAPI service is probably not running. Do not retry this tool; "
            "tell the user the documentation service is unavailable."
        }

    if response.status_code != 200:
        return {"error": f"Docs service returned HTTP {response.status_code}: {response.text[:200]}"}

    matches = response.json().get("matches", [])
    usable = [m for m in matches if m.get("score", 0.0) >= RELEVANCE_FLOOR]

    return {
        "query": query,
        "relevance_floor": RELEVANCE_FLOOR,
        "searches_used": _searches_used["n"],
        "searches_remaining": MAX_SEARCHES - _searches_used["n"],
        "usable_chunks": len(usable),
        "top_score": max((m.get("score", 0.0) for m in matches), default=0.0),
        "passages": [
            {
                "chunk_id": m.get("chunk_id"),
                "source": m.get("source"),
                "score": m.get("score"),
                "text": m.get("text", ""),
            }
            for m in usable
        ],
    }
