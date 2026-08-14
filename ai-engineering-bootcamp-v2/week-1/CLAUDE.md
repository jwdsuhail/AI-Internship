# Week 1 — Ship Your First AI Endpoint (context for Claude Code)

This is the Maven "AI Engineering Bootcamp" Session 1 homework. Goal: ship a
reliable `POST /ask` endpoint (typed I/O, structured output, cost tracking)
plus a Streamlit UI, deployed publicly on Render.

## Which folder to work in

This repo (forked from `akshika47/AI-Internship`) contains **two versions**
of Week 1 — use `week-1/`, not `week-1v2/`.

- `week-1/` — the original 5-stage version. `serve_stage1.py` through
  `serve_stage5.py`, `main.py` (full system), `test_all_stages.py`. **This
  matches the official assignment guide's exact commands and file names —
  use this one.**
- `week-1v2/` — a simplified "class-ready" rewrite with different file names
  (`smoke_test.py` instead of `test_all_stages.py`, only 3 stage files under
  `stages/`). Do not use this for the assignment — the official guide's
  instructions (`python test_all_stages.py`, "stages 1 to 5") don't match
  this folder's structure and will error if run here.

## Known gotcha: nested `answer` field in `main.py`

In `week-1/main.py` (the full stage-5 system), the response's `answer`
field is a **nested object**, not a plain string:

```json
{
  "answer": { "answer": "...", "confidence": 0.9, "sources_needed": false },
  "tokens_used": 123,
  "model": "gpt-4o",
  "latency_ms": 842,
  "cost_usd": 0.0031
}
```

The actual answer text is at `response.answer.answer`, not
`response.answer`. This matters for the Streamlit UI and any curl proof —
don't treat `answer` as a plain string when displaying it.

(`serve_stage1.py` is the simple version if you want to see a flat
`answer: str` shape for comparison — that's stage 1 only, not what
`main.py` / stage 5 does.)

## Assignment pass bar ("Ready when")

Enough to pass:
- Live `/ask` endpoint (public HTTPS URL, not localhost-only)
- Structured JSON response with `answer`, `tokens_used`, `cost_usd`
- Proof via curl against the **live** deployed URL (not local)
- A Streamlit UI that calls `/ask` and shows the result (screenshot required)

Optional add-ons (not required to pass): guardrail proof (`force_bad`
screenshot), README polish, a short model-cost writeup, LinkedIn post.

## Build order (do not skip ahead)

1. **Local first**: `cp .env.example .env`, add `OPENAI_API_KEY`, create a
   venv in this folder, `pip install -r requirements.txt`. Run
   `uvicorn main:app --host 127.0.0.1 --port 8000 --reload`, confirm with
   `python test_all_stages.py` and a local curl.
2. **Deploy to Render**: push this fork to GitHub, create a Render Web
   Service pointing at it. Since this is a monorepo, set Render's
   **Root Directory** to `ai-engineering-bootcamp-v2/week-1`. Build command
   `pip install -r requirements.txt`, start command
   `uvicorn main:app --host 0.0.0.0 --port $PORT`. Set `OPENAI_API_KEY` in
   Render's environment variables (never commit it).
3. **Prove it live**: curl the actual Render URL (not localhost) and confirm
   the JSON shape.
4. **Streamlit UI**: run `streamlit run demo_page.py`, point it at the live
   URL, screenshot a real question/response.
5. **Submit**: live URL + curl command + Streamlit screenshot go in the
   Maven submission channel **only**. Never post the live URL publicly
   (LinkedIn, GitHub README, etc.) — anyone with it can hit `/ask` and burn
   OpenAI credits on your key.

## Environment notes already confirmed

- Python 3.13 is fine for this course's stack (FastAPI/uvicorn/OpenAI SDK
  all support it).
- `.env` must never be committed — confirm `.gitignore` covers it before
  the first `git push`, not after.
