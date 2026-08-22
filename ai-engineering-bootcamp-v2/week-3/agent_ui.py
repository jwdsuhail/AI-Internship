"""
Session 3 Streamlit UI — runs the ADK docs agent and shows the loop as it happens.

The point of this page is the middle column of the run, not the answer: every Think,
Act, and Observe is rendered so the agent's self-correction is visible to a viewer who
has never seen the code. Secrets are read from .env by tools.py / the ADK client and
are never entered or displayed here.

Run:  streamlit run agent_ui.py
"""

import asyncio
import json

import streamlit as st

from agent import CANDIDATE_MODELS, MAX_LLM_CALLS, MODEL, probe_models, run_agent
from tools import API_BASE_URL, MAX_SEARCHES, RELEVANCE_FLOOR

st.set_page_config(page_title="Docs Agent — Session 3", page_icon="🔍", layout="centered")

st.title("🔍 Self-correcting docs agent")
st.caption(
    "Google ADK agent over the Session 2 RAG service. It searches, reads the relevance "
    "scores, and decides whether to search again, answer, or refuse."
)

# Emoji per probe status, so the quota table is scannable at a glance rather than read.
STATUS_ICONS = {
    "available": "✅",
    "daily quota spent": "❌",
    "rate limited — wait 60s": "⏳",
    "overloaded — retry shortly": "⚠️",
    "not available to this key": "🚫",
}

with st.sidebar:
    st.subheader("Model")

    # Free-tier quota is per-model and one model's daily allowance runs out well before a
    # demo session does, so the model is a runtime choice rather than a constant.
    chosen_model = st.selectbox(
        "Gemini model",
        options=CANDIDATE_MODELS,
        index=CANDIDATE_MODELS.index(MODEL) if MODEL in CANDIDATE_MODELS else 0,
        help="Free-tier allowance is 20 requests per day PER MODEL. Switching models buys a "
        "fresh allowance without waiting for the daily reset.",
    )

    # Probing spends a real request against every model that still has quota, so it only runs
    # when asked. Results are cached in session state and survive Streamlit's reruns.
    if st.button("Check quota", use_container_width=True):
        with st.spinner("Probing models…"):
            st.session_state.quota = probe_models()

    st.caption(f"Costs 1 request per model with quota left ({len(CANDIDATE_MODELS)} models).")

    quota = st.session_state.get("quota")
    if quota:
        for name, status in quota.items():
            icon = STATUS_ICONS.get(status, "•")
            marker = " ← selected" if name == chosen_model else ""
            st.markdown(f"{icon} `{name}` — {status}{marker}")

        if quota.get(chosen_model) != "available":
            usable = [m for m, s in quota.items() if s == "available"]
            if usable:
                st.warning(f"Selected model is not usable. Try **{usable[0]}**.")
            else:
                st.error("No model has quota right now. Wait for the daily reset.")

    st.divider()
    st.subheader("Configuration")
    st.text(f"Docs service: {API_BASE_URL}")
    st.text(f"Step cap:     {MAX_LLM_CALLS} LLM calls")
    st.text(f"Search budget: {MAX_SEARCHES} per question")
    st.text(f"Score floor:  {RELEVANCE_FLOOR}")
    st.divider()
    st.markdown(
        "**Agent, not workflow:** the number of searches and the query wording are chosen "
        "by the model from the scores it observes — they are not hard-coded."
    )
    st.divider()
    st.caption(
        "Gemini free tier allows 5 requests/minute and 20 per day, both per model. One run "
        "uses ~4 requests — leave ~60s between runs."
    )

# Ordered easiest-first on purpose. The refusal is the strongest evidence of agency, but a
# refusal as the first thing a viewer clicks reads as a broken app, so it goes last and the
# caption says what to expect before they get there.
#
# Note that sloppy wording does NOT force a retry — the model silently expands "WFH" before
# searching. What forces the loop is a topic the corpus is genuinely missing: retrieval returns
# leave-adjacent passages above the score floor, the model reads them, sees they answer a
# different question, and rephrases.
st.markdown(
    "The first two are answered from the docs in a single search. **The last one is not "
    "covered by these documents** — the agent searches, rephrases, and then refuses rather "
    "than inventing an answer. That refusal is the correct outcome, not a failure."
)
examples = [
    "What is the refund window?",
    "how many days can i WFH",
    "What is the parental leave policy?",
]
cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_input(
    "Ask the company docs",
    value=st.session_state.get("question", ""),
    placeholder="e.g. how do I get a laptop on my first day?",
)

run = st.button("Run agent", type="primary", disabled=not question.strip())


def summarise(result) -> str:
    """One-line headline for a tool result, so the score is readable at a glance."""

    if not isinstance(result, dict):
        return str(result)[:200]
    if "error" in result:
        # Labelled because a red error line mid-demo reads as the app breaking. The tool
        # returning an error dict instead of raising is the designed path — the model gets to
        # see the failure and report it rather than the exception killing the run.
        return (
            f"❌ {result['error']}  \n"
            "*(expected behaviour — the tool returns errors as observations the model can "
            "read, rather than raising and killing the run)*"
        )
    if result.get("budget_exhausted"):
        return (
            f"🛑 search budget spent ({MAX_SEARCHES}/{MAX_SEARCHES}) → "
            "tool refuses further searches; the agent must answer or refuse now"
        )
    verdict = "nothing usable — the agent should rephrase" if result.get("usable_chunks") == 0 else "usable passages found"
    return (
        f"usable_chunks={result.get('usable_chunks')} · top_score={result.get('top_score')} · "
        f"searches left: {result.get('searches_remaining')} → {verdict}"
    )


async def collect(q: str, sink, model: str) -> None:
    """Drain the agent's step stream, rendering each step the moment it arrives."""

    async for step in run_agent(q, model=model):
        sink(step)


if run and question.strip():
    st.divider()
    st.subheader("Think → Act → Observe")

    steps_area = st.container()
    final_answer = {"text": None}
    step_log = []

    def render(step: dict) -> None:
        step_log.append(step)
        with steps_area:
            if step["kind"] == "act":
                query = step["args"].get("query", "")
                st.markdown(f"🧠 **THINK** — I need the docs; search for `{query}`")
                st.markdown(f"⚙️ **ACT** — `{step['tool']}(query={query!r})`")
            elif step["kind"] == "observe":
                st.markdown(f"👀 **OBSERVE** — {summarise(step['result'])}")
                with st.expander("raw tool result"):
                    st.code(json.dumps(step["result"], indent=2, default=str)[:4000], language="json")
                st.markdown("---")
            elif step["kind"] == "think":
                st.markdown(f"🧠 **THINK** — {step['text']}")
            elif step["kind"] == "final":
                final_answer["text"] = step["text"]
            elif step["kind"] == "error":
                st.error(step["text"])

    with st.spinner("Agent working…"):
        asyncio.run(collect(question, render, chosen_model))

    searches = sum(1 for s in step_log if s["kind"] == "act")
    tool_failed = any(
        s["kind"] == "observe" and isinstance(s["result"], dict) and "error" in s["result"]
        for s in step_log
    )

    st.divider()
    st.subheader("Final answer")
    if final_answer["text"]:
        # The agent's own wording is kept verbatim; the bracketed note is the UI telling the
        # viewer this ending was designed rather than a crash, since the two look identical.
        answer = final_answer["text"]
        if tool_failed:
            answer += (
                "\n\n*(expected — the docs service at "
                f"{API_BASE_URL} is unreachable. The tool returned that as an observation "
                "rather than raising, so the agent reported it instead of inventing an "
                "answer. Start the service and ask again for a grounded answer.)*"
            )
        st.success(answer)
    elif any(s["kind"] == "error" for s in step_log):
        # An interrupted run and a deliberate refusal both end without an answer, but they mean
        # opposite things. Saying "no answer" for a crash makes a broken run look like a decision.
        st.warning(
            "⚠️ The run was interrupted before the agent could answer — see the error above. "
            "This is not the agent declining to answer; it never got the chance."
        )
    else:
        st.warning("No final answer was produced — see the steps above.")
    st.caption(
        f"{searches} tool call(s) this run · cap is {MAX_LLM_CALLS} LLM calls · "
        f"model: {chosen_model}"
    )
