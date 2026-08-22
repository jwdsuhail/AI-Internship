"""
Session 3 — Self-correcting docs agent (Google ADK).

JOB: When a user asks a question about the company docs, return a cited answer or an
honest refusal, using search_docs (the Session 2 Pinecone retrieval) — deciding for
itself how many times and with what wording to search.

Why this is an agent and not a workflow: the number of retrieval calls and the query
wording are not fixed. The tool hands back cosine scores, the model reads them, and it
chooses to search again with different wording, refuse, or answer. Session 2's /ask is
the workflow version of this same job — always retrieve once, always answer once.

Run:  python agent.py "What is the remote work policy?"
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ADK logs a full multi-frame traceback to stderr for any model error before re-raising it.
# We already catch those and surface them as a readable step, so the duplicate stack trace is
# pure noise that would swamp a live demo. Silence ADK's own handler, keep our own reporting.
logging.getLogger("google_adk").setLevel(logging.CRITICAL)

load_dotenv(Path(__file__).resolve().parent / ".env")

# Google issues this key under two different names depending on where you got it: AI Studio
# calls it GEMINI_API_KEY, the Cloud/genai docs call it GOOGLE_API_KEY. The ADK client only
# looks for GOOGLE_API_KEY, so mirror the other name across before any client is built —
# otherwise a perfectly valid key in .env looks like no key at all.
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tools import API_BASE_URL, MAX_SEARCHES, reset_search_budget, search_docs

# The course sample pins gemini-2.5-flash, which now 404s for keys issued after its retirement
# ("no longer available to new users"), so this picks from CANDIDATE_MODELS below instead.
#
# Free tier allows 5 requests per minute and 20 per day, both PER MODEL. One run costs ~4
# requests, so a single model covers roughly five runs a day. Override per invocation:
#   GEMINI_MODEL=gemini-3.1-flash-lite python agent.py "..."
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
APP_NAME = "docs_agent"

# The free-tier daily allowance is per-model, so an exhausted model is worked around by moving
# to another one rather than waiting for the reset. Ordered by preference: full flash models
# follow instructions and call tools more reliably than the lite variants, which matters for a
# loop that depends on the model judging its own search quality.
# 3-flash-preview leads because 3.7-flash returned 503 "experiencing high demand" on two
# separate attempts here; a demo that dies mid-run is worse than a marginally older model.
CANDIDATE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

# Backwards-compatible alias — the error path reads this to suggest where to go next.
FALLBACK_MODELS = CANDIDATE_MODELS

# Hard ceiling on the loop. Each LLM call is one "Think", so 8 allows roughly three
# search-and-reassess cycles plus a final answer. Without a cap a model that keeps
# rephrasing a doomed query bills you forever; ADK fails closed when this is hit.
# Overridable by env so the cap can be demonstrated (MAX_LLM_CALLS=2) without editing code.
MAX_LLM_CALLS = int(os.environ.get("MAX_LLM_CALLS", "8"))

INSTRUCTION = """You answer questions about the company documentation.

You have one tool: search_docs. You have no knowledge of this company's policies
yourself, so you must search before answering — never answer from memory.

You get {max_searches} searches per question. This is enforced by the tool, not a
suggestion — every result carries searches_remaining, and once it reaches 0 the
tool will refuse to search again.

How to work:
1. Call search_docs with the wording you think appears in the documents.
2. Read the result. Look at usable_chunks and top_score, not just the text.
3. If usable_chunks is 0, your wording missed. Rephrase and search again — use
   synonyms or the formal phrasing a policy document would use.
4. If a search returns usable passages that actually answer the question, answer
   using ONLY those passages. Do not keep searching once you can answer.
5. When searches_remaining reaches 0, stop. Answer with the best passages you
   have, or if none of them answer the question, reply exactly:
   "I don't have enough information to answer that." Do not guess.

When you answer, cite the chunk_id in square brackets after each fact you state,
for example: Employees may work remotely two days a week [employee_handbook#3].
Only cite chunk_ids that appeared in a search result you actually used.

If a search comes back with an "error" key, tell the user the documentation service
is unavailable and stop. Do not invent an answer.""".format(max_searches=MAX_SEARCHES)

def build_agent(model: str = MODEL) -> Agent:
    """Construct the agent against a specific Gemini model.

    A factory rather than a single module-level agent because free-tier quota is per-model:
    when one model's daily allowance is spent, the UI needs to rebuild the same agent on a
    different model without restarting the process.
    """

    return Agent(
        name="docs_agent",
        model=model,
        description=(
            "Answers company documentation questions with citations, searching repeatedly "
            "until it finds relevant passages or honestly refuses."
        ),
        instruction=INSTRUCTION,
        tools=[search_docs],
    )


# Module-level instance for `adk web` and the CLI, which do not need to switch models.
root_agent = build_agent()


def probe_models(models: list[str] | None = None) -> dict[str, str]:
    """Report each model's free-tier state by asking it for one token.

    There is no read-only way to query remaining quota, so this spends one request per model
    that still has allowance — which is why the UI puts it behind an explicit button rather
    than running it on page load.
    """

    from google import genai

    client = genai.Client()
    statuses: dict[str, str] = {}

    for model in models or CANDIDATE_MODELS:
        try:
            client.models.generate_content(model=model, contents="hi")
            statuses[model] = "available"
        except Exception as exc:
            detail = str(exc)
            if "PerDay" in detail:
                statuses[model] = "daily quota spent"
            elif "RESOURCE_EXHAUSTED" in detail or "429" in detail:
                statuses[model] = "rate limited — wait 60s"
            elif "503" in detail or "UNAVAILABLE" in detail:
                statuses[model] = "overloaded — retry shortly"
            elif "404" in detail or "NOT_FOUND" in detail:
                statuses[model] = "not available to this key"
            else:
                statuses[model] = f"error: {exc.__class__.__name__}"

    return statuses


async def run_agent(question: str, user_id: str = "user1", model: str | None = None):
    """Run one task and yield labelled Think / Act / Observe / Final steps as they happen.

    Yielding rather than returning a finished list is what lets the Streamlit UI show the
    loop unfolding — the same reason the CLI can print it live.
    """

    # The search budget lives in the tool module, which outlives a single run in a
    # long-lived process like Streamlit. Reset it here so every run starts with a full
    # budget — without this the second question in a session could never search at all.
    reset_search_budget()

    # Reuse the module-level agent unless a different model was picked, so the common path
    # does not rebuild an identical agent on every run.
    agent = root_agent if model in (None, MODEL) else build_agent(model)
    active_model = model or MODEL

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session = await session_service.create_session(app_name=APP_NAME, user_id=user_id)

    message = types.Content(role="user", parts=[types.Part(text=question)])
    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS)

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=message,
            run_config=run_config,
        ):
            if not (event.content and event.content.parts):
                continue

            for part in event.content.parts:
                # A tool call the model decided to make: the Act, and the visible edge of the Think.
                if part.function_call:
                    yield {
                        "kind": "act",
                        "tool": part.function_call.name,
                        "args": dict(part.function_call.args or {}),
                    }
                # What the tool actually returned: the Observe the model reasons over next.
                elif part.function_response:
                    yield {
                        "kind": "observe",
                        "tool": part.function_response.name,
                        "result": part.function_response.response,
                    }
                elif part.text and part.text.strip():
                    # Final text vs interim reasoning text — both worth showing.
                    yield {
                        "kind": "final" if event.is_final_response() else "think",
                        "text": part.text.strip(),
                    }
    except Exception as exc:
        # Fail closed, and say something a human can act on. The two realistic causes are the
        # free-tier rate limit and hitting our own step cap; everything else is reported raw.
        detail = str(exc)
        if "RESOURCE_EXHAUSTED" in detail or "429" in detail:
            # Two different quotas produce this: 5/minute and 20/day, both per model.
            per_day = "PerDay" in detail
            if per_day:
                # Never suggest the model that just failed — name the next one in the list.
                alternatives = [m for m in CANDIDATE_MODELS if m != active_model]
                suggestion = alternatives[0] if alternatives else "another Gemini model"
                message = (
                    f"Gemini free-tier daily quota exhausted for {active_model} (20 requests "
                    f"per day). The allowance is per-model — pick a different model in the "
                    f"sidebar (try {suggestion}) and run again."
                )
            else:
                message = (
                    f"Gemini free-tier rate limit hit for {active_model} (5 requests per "
                    "minute). Wait ~60 seconds and run again."
                )
        elif "max_llm_calls" in detail.lower() or "limit" in exc.__class__.__name__.lower():
            message = (
                f"Step cap reached: the agent used its {MAX_LLM_CALLS} allowed LLM calls without "
                "finishing. Stopping rather than looping — this is the bound working."
            )
        else:
            message = f"{exc.__class__.__name__}: {detail[:300]}"

        yield {"kind": "error", "text": message}


def _summarise(result) -> str:
    """Condense a tool result for console display — full passages drown the log."""

    if not isinstance(result, dict):
        return str(result)[:300]
    if "error" in result:
        return f"ERROR: {result['error']}"
    if result.get("budget_exhausted"):
        return f"SEARCH BUDGET SPENT ({MAX_SEARCHES}/{MAX_SEARCHES}) — must answer or refuse now"
    ids = ", ".join(p.get("chunk_id", "?") for p in result.get("passages", [])) or "none"
    return (
        f"usable_chunks={result.get('usable_chunks')} "
        f"top_score={result.get('top_score')} "
        f"searches_left={result.get('searches_remaining')} chunks=[{ids}]"
    )


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is the remote work policy?"

    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print(f"DOCS SERVICE: {API_BASE_URL}")
    print(f"{'=' * 70}\n")

    async for step in run_agent(question):
        if step["kind"] == "act":
            print(f"THINK   -> I should search for: {step['args'].get('query')!r}")
            print(f"ACT     -> {step['tool']}({step['args']})")
        elif step["kind"] == "observe":
            print(f"OBSERVE -> {_summarise(step['result'])}\n")
        elif step["kind"] == "think":
            print(f"THINK   -> {step['text']}\n")
        elif step["kind"] == "final":
            print(f"{'-' * 70}\nFINAL ANSWER:\n{step['text']}\n{'-' * 70}")
        elif step["kind"] == "error":
            print(f"!! {step['text']}")


if __name__ == "__main__":
    asyncio.run(main())
