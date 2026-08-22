# Session 3 — Self-correcting docs agent (Google ADK)

Turns the Session 1/2 capstone (`../week-1`) into an agent. The RAG service stays exactly
as it was; this adds a Gemini brain that decides *how* to use it.

## The job

> When a user asks a question about the company docs, the agent returns a cited answer or an
> honest refusal, using `search_docs` (the Session 2 Pinecone retrieval) — deciding for itself
> how many times and with what wording to search.

## Agent or workflow?

**Agent.** The number of retrieval calls and the query wording are not fixed. `search_docs`
returns cosine relevance scores, the model reads them, and it chooses to search again with
different wording, answer, or refuse.

The proof is that the same code takes a different number of steps depending on what comes back:

| Question | Tool calls | Why |
|---|---|---|
| `how many days can i WFH` | 1 | First search hit at 0.62 — answer immediately |
| `What is the parental leave policy?` | 3 | Retrieval kept returning *other* leave policies; the model rephrased twice, then refused honestly |

Session 2's `POST /ask` is the workflow version of this same job: always retrieve once, always
answer once. That is the right shape when the steps are known in advance. Here they are not.

## The agent in 5 bullets

The assignment asks for an explanation a beginner can repeat. These five:

1. **One agent, one tool.** A single ADK `Agent` running a Gemini flash model, with exactly one
   tool called `search_docs`. No router, no sub-agents.
2. **The tool is real.** `search_docs` makes an HTTP call to the Session 2 FastAPI service,
   which embeds the query with OpenAI and searches a live Pinecone index. Nothing is faked.
3. **The tool returns scores, not just text.** That is the whole trick — the model can see
   *how good* its search was, so it can judge its own work instead of accepting the first result.
4. **The model decides what happens next.** Good scores → answer with citations. Bad scores →
   rephrase and search again. Three misses → refuse honestly rather than invent an answer.
5. **The loop is bounded.** `RunConfig(max_llm_calls=8)`. On hitting the cap the run stops and
   says so, rather than looping until the quota is gone.

Patterns for `Agent`, `Runner`, `InMemorySessionService`, and `types.Content` were taken from
`ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py`. The tool, the job, the
self-correction instruction, the step labelling, and the UI are original to this project.

## Architecture

```
Streamlit (agent_ui.py)
      |
      v
ADK Agent  (agent.py)  Gemini flash, max_llm_calls=8, max 3 searches
      |
      | tool: search_docs
      v
tools.py  --HTTP-->  week-1 FastAPI  /debug/retrieve
                            |
                            v
                     Pinecone + OpenAI embeddings
```

Deliberately **one agent with one tool**, not a multi-agent router. There is only one role
here, and the assignment is explicit that a router with nothing to route is worse than an
agent plus tools.

## Running it

Needs **two** processes. The agent is a new brain on the existing body.

```bash
# Terminal 1 — the Session 2 RAG service
cd ../week-1
./.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — the agent
cd week-3
source .venv/bin/activate          # Python 3.12 + google-adk

python agent.py "What is the parental leave policy?"   # CLI, prints Think/Act/Observe
streamlit run agent_ui.py                              # UI at localhost:8501
```

## Setup notes

- `cp .env.example .env`, then set your Gemini key. **Either** `GEMINI_API_KEY` or
  `GOOGLE_API_KEY` works — `agent.py` mirrors one to the other, because AI Studio issues the
  key under one name and the ADK client reads the other.
- Python **3.12+** is required by ADK. Use this folder's `.venv`; a bare `python3` may resolve
  to an older interpreter.
- The course sample's `gemini-2.5-flash` now 404s for newly issued keys ("no longer available
  to new users"). This defaults to `gemini-3-flash-preview` and offers a model picker in the
  Streamlit sidebar, because free-tier quota is per-model and one model runs out mid-session.

## Gotcha: free-tier rate limit

Gemini's free tier allows **5 requests per minute per model**. One full run costs ~4 requests,
so back-to-back runs will 429. Wait ~60 seconds between demo runs. The agent catches this and
reports it as a readable step instead of crashing.

## Two bounds, doing two different jobs

**`RunConfig(max_llm_calls=8)`** — the safety bound. On hitting it ADK stops and the run
reports it. Fails closed rather than looping until the quota is gone. Demonstrated in
`run_log_capped.txt` by running with `MAX_LLM_CALLS=2`.

**`MAX_SEARCHES=3` in `tools.py`** — the behavioural bound. The instruction originally *asked*
for at most 3 wordings and the model ran 6, costing ~7 LLM calls and tripping the free tier's
5-per-minute limit before it could answer. Moving the budget into the tool makes it a fact the
model observes (every result carries `searches_remaining`) rather than a rule it can drift
past. It does not kill the run: once spent, the tool tells the model to answer or refuse with
what it has, so the run still ends in a real final answer.

The lesson: if a constraint matters, it belongs in the tool, not the prompt.

## Tool error handling

`search_docs` never raises. A dead docs service, a non-200, or an empty query all come back as
`{"error": ...}`, which the model reads as an observation and reports to the user. An uncaught
exception inside a tool kills the whole run; a returned error does not.

## Files

| File | What it is |
|---|---|
| `agent.py` | The ADK agent, the step-labelling loop, and the CLI |
| `tools.py` | `search_docs` — the one real tool, over week-1's `/debug/retrieve` |
| `agent_ui.py` | Streamlit UI showing Think → Act → Observe |
| `run_log.txt` | Captured proof run |

Patterns (`Agent`, `Runner`, `InMemorySessionService`, `types.Content`) come from
`ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py`.

## Proof runs (captured logs)

| File | What it proves | Assignment requirement |
|---|---|---|
| `run_log.txt` | 3 searches, self-correction, honest refusal | multi-step task; Think → Act → Observe |
| `run_log_success.txt` | tool result used in a final cited answer | "the model uses that result in a final answer" |
| `run_log_capped.txt` | run stopped at `MAX_LLM_CALLS=2` | "loop is bounded / fail closed at the cap" |

Regenerate any of them:

```bash
python agent.py "What is the parental leave policy?"          > run_log.txt
python agent.py "What is the refund window?"                  > run_log_success.txt
MAX_LLM_CALLS=2 python agent.py "What is the parental leave policy?" > run_log_capped.txt
```

Wait ~60s between runs — Gemini's free tier allows 5 requests per minute and each run uses ~4.

## Submission checklist (Path A)

- [x] Stack named in one word: **ADK**
- [x] Multi-step task completes with at least one real tool call
- [x] Think → Act → Observe visible in logs
- [x] Loop is bounded (`max_llm_calls`), fails closed, demonstrated in `run_log_capped.txt`
- [x] Streamlit UI runs the agent (`streamlit run agent_ui.py`)
- [x] One-liner on agent vs workflow (top of this file)
- [ ] Screenshot / Loom captured and posted to Maven
- [x] No secrets in code, logs, or screenshots — `.env` is gitignored

## Not done, on purpose

- **`POST /agent` on the Render service** — Step 5 of the guide, explicitly optional. It would
  mean adding `google-adk` to week-1's requirements and redeploying. The localhost Streamlit
  demo satisfies Path A.
- **Multi-agent / A2A** — there is one role here. The guide warns that a router with nothing to
  route is worse than one agent plus tools.
- **MCP tool** — the existing FastAPI service is already the clean integration point.
