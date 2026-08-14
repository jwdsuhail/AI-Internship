"""Week 1 live demo — five stages in one file, built up live in class."""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field, ValidationError

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


# Session 2 — one embedding model for both ingest and query. Changing it means re-indexing,
# so it is locked here rather than passed per request.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # Must match the Pinecone index dimension exactly.

# Below this cosine score nothing retrieved is really about the question, so we refuse without
# calling the model. Measured on this corpus: on-topic hits land ~0.55-0.70, noise lands ~0.00.
MIN_RELEVANCE_SCORE = 0.25

REFUSAL_TEXT = "I don't have enough information to answer that."

# The grounding contract: context-only, cite chunk IDs, refuse rather than guess.
GROUNDING_PROMPT = """Answer using ONLY the context below.
If the context does not contain the answer, reply with exactly: "{refusal}"
Do not use outside knowledge. Cite the chunk IDs (shown in square brackets) that you used.

Context:
{context}

Question: {question}"""

_pinecone_index = None  # Cached so we do not rebuild the client on every request.


def get_index():
    """
    Connect to Pinecone on first use, not at import time.

    Lazy on purpose: if the vector store is misconfigured the service still boots and the
    Session 1 paths keep working, instead of the whole deploy crash-looping on Render.
    """

    global _pinecone_index
    if _pinecone_index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not api_key or not index_name:
            raise RuntimeError(
                "PINECONE_API_KEY and PINECONE_INDEX_NAME must be set (locally in .env, "
                "on Render in the Environment tab)"
            )
        _pinecone_index = Pinecone(api_key=api_key).Index(index_name)
    return _pinecone_index


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch in a single call — batching keeps ingest fast and cheap."""

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks, breaking on the largest natural boundary that fits.

    Overlap matters: a fact that straddles a chunk edge would otherwise be split in half and
    retrieved by neither query. Separators are tried widest-first so we cut between paragraphs
    before we resort to cutting mid-sentence.
    """

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Back off to the last clean boundary in this window; fall through to a hard cut.
        cut = -1
        for separator in separators:
            found = text.rfind(separator, start, end)
            if found > start:
                cut = found + len(separator)
                break
        if cut == -1:
            cut = end

        chunks.append(text[start:cut].strip())
        start = max(cut - chunk_overlap, start + 1)  # max() guards against a zero-width step.

    return [chunk for chunk in chunks if chunk]


class IngestRequest(BaseModel):
    """Plain text in, chunked vectors out — this is the pipeline the assignment asks for."""

    text: str
    document_id: str
    source: str | None = None  # Optional filename/label carried through to citations.
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)


class IngestResponse(BaseModel):
    """Typed receipt so the caller can confirm what actually landed in the index."""

    document_id: str
    chunks_indexed: int
    status: str


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """
    Chunk, embed, and upsert one document.

    curl -s -X POST http://127.0.0.1:8000/ingest \
      -H "Content-Type: application/json" \
      -d '{"text": "Remote work: up to 3 days per week.", "document_id": "handbook"}'
    """

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not body.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id must not be empty")
    if body.chunk_overlap >= body.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")

    chunks = chunk_text(body.text, body.chunk_size, body.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="text produced no chunks after cleaning")

    index = get_index()

    # Re-ingesting a document_id should replace it, not stack duplicates alongside it.
    # IDs are deterministic ("handbook#0"), so we clear the old ones by prefix first.
    try:
        stale = [vector_id for page in index.list(prefix=f"{body.document_id}#") for vector_id in page]
        if stale:
            index.delete(ids=stale)
    except Exception:
        pass  # Prefix listing is best-effort; deterministic IDs still overwrite in place.

    # Embed and upsert in batches — one call per chunk would be slow and needlessly expensive.
    batch_size = 100
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        vectors = [
            {
                "id": f"{body.document_id}#{offset + i}",
                "values": embedding,
                "metadata": {
                    "document_id": body.document_id,
                    "chunk_index": offset + i,
                    "source": body.source or body.document_id,
                    "text": chunk,  # Stored so retrieval can return text without a second lookup.
                },
            }
            for i, (chunk, embedding) in enumerate(zip(batch, embed_texts(batch)))
        ]
        index.upsert(vectors=vectors)

    return IngestResponse(
        document_id=body.document_id,
        chunks_indexed=len(chunks),
        status="indexed",
    )


def retrieve_chunks(question: str, k: int = 5, document_id: str | None = None) -> list[dict]:
    """
    Embed a question and return the top-k matching chunks with scores.

    Deliberately LLM-free: retrieval is tested and debugged on its own, so when an answer is
    wrong we already know whether the problem is the passages or the prompt.
    """

    query_vector = embed_texts([question])[0]
    query_filter = {"document_id": {"$eq": document_id}} if document_id else None

    response = get_index().query(
        vector=query_vector,
        top_k=k,
        include_metadata=True,
        filter=query_filter,
    )

    matches = []
    for match in response.get("matches", []):
        metadata = match.get("metadata") or {}
        matches.append(
            {
                "chunk_id": match.get("id"),
                "score": round(float(match.get("score", 0.0)), 4),
                "document_id": metadata.get("document_id"),
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "source": metadata.get("source"),
                "text": metadata.get("text", ""),
            }
        )
    return matches


@app.get("/debug/retrieve")
def debug_retrieve(q: str, k: int = 5, document_id: str | None = None) -> dict:
    """
    Retrieval-only view: top-k chunks and scores for a question, no generation.

    curl -s "http://127.0.0.1:8000/debug/retrieve?q=What+is+the+remote+work+policy&k=5"
    """

    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")

    try:
        matches = retrieve_chunks(q, k=k, document_id=document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}")

    return {"question": q, "k": k, "matches_found": len(matches), "matches": matches}


@app.get("/debug/vectorstore")
def debug_vectorstore() -> dict:
    """Confirm Pinecone is reachable and its dimension matches our embedding model."""

    try:
        stats = get_index().describe_index_stats()
    except Exception as exc:  # Surface config/network errors as a clear 503, not a 500.
        raise HTTPException(status_code=503, detail=f"Pinecone unreachable: {exc}")

    dimension = getattr(stats, "dimension", None)
    return {
        "status": "ok",
        "index": os.environ.get("PINECONE_INDEX_NAME"),
        "embedding_model": EMBEDDING_MODEL,
        "dimension": dimension,
        # A mismatch here is the classic Session 2 bug — catch it before ingesting anything.
        "dimension_matches_model": dimension == EMBEDDING_DIM,
        "total_vectors": getattr(stats, "total_vector_count", 0),
    }


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    k: int = Field(default=5, ge=1, le=20)  # Session 2 — how many chunks to ground on.
    document_id: str | None = None  # Session 2 — optionally scope the search to one document.


class Citation(BaseModel):
    """One retrieved chunk, surfaced so the caller can audit where the answer came from."""

    chunk_id: str
    document_id: str
    score: float


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    attempts: int
    citations: list[Citation] = []  # Session 2 — chunks the answer actually used.
    retrieved: list[str] = []  # Every chunk that cleared the relevance floor, for auditing.
    refused: bool = False  # True when the documents did not contain the answer.


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """
    Answer one question from the ingested documents only, with citations and a refusal path.

    Session 2 wraps the Session 1 generation machinery rather than replacing it: retrieval and
    grounding happen first, then the same structured-output call, retry, and cost accounting run.
    """

    model = body.model or DEFAULT_MODEL
    last_error: str | None = None
    start = time.perf_counter()

    # Retrieve first — the LLM never sees a question we have no documents for.
    try:
        matches = retrieve_chunks(body.question, k=body.k, document_id=body.document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}")

    # Only chunks that clear the floor are grounded on or cited — passing near-miss chunks to
    # the model wastes tokens and produces citations the answer never actually used.
    relevant = [m for m in matches if m["score"] >= MIN_RELEVANCE_SCORE]

    citations = [
        Citation(chunk_id=m["chunk_id"], document_id=m["document_id"], score=m["score"])
        for m in relevant
    ]

    # Cheap refusal: if nothing clears the relevance floor, say so without spending tokens.
    if not relevant:
        return AskResponse(
            answer=Answer(answer=REFUSAL_TEXT, confidence=0.0, sources_needed=True),
            tokens_used=0,
            model=model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cost_usd=0.0,
            attempts=0,
            citations=[],
            refused=True,
        )

    context = "\n\n".join(
        f"[{m['chunk_id']}] (document_id: {m['document_id']})\n{m['text']}" for m in relevant
    )
    grounded_prompt = GROUNDING_PROMPT.format(
        context=context, question=body.question, refusal=REFUSAL_TEXT
    )

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    grounded_prompt, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounded_prompt, model
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

            # The model can still refuse even when retrieval cleared the floor — report that.
            refused = REFUSAL_TEXT.lower().rstrip(".") in answer.answer.lower()

            # Cite what the answer actually used, not everything we retrieved. The prompt asks
            # the model to inline chunk IDs, so parse those back out; if it cited nothing,
            # fall back to the single best chunk rather than claiming all of them.
            cited_ids = set(re.findall(r"\[([^\[\]]+#\d+)\]", answer.answer))
            used = [c for c in citations if c.chunk_id in cited_ids] or citations[:1]

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                attempts=attempt + 1,
                citations=[] if refused else used,
                retrieved=[c.chunk_id for c in citations],
                refused=refused,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
