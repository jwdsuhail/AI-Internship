"""
Week 2 RAG demo UI — a thin front end over the live FastAPI service.

The API is the source of truth: this page only calls /ingest, /ask, and the debug routes.
No chunking, embedding, or retrieval logic lives here on purpose.

Run:  streamlit run rag_demo.py
"""

import os

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 120.0  # Render free tier cold-starts slowly; be patient rather than fail fast.

st.set_page_config(page_title="RAG Demo — Session 2", page_icon="📚", layout="centered")


def api_post(base_url: str, path: str, payload: dict) -> tuple[int, dict | str]:
    """POST helper that turns connection failures into a readable message, not a traceback."""

    try:
        response = httpx.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=TIMEOUT)
    except httpx.RequestError:
        return 0, {"error": f"Cannot reach {base_url} — is the service running?"}
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


def api_get(base_url: str, path: str, params: dict | None = None) -> tuple[int, dict | str]:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}{path}", params=params, timeout=TIMEOUT)
    except httpx.RequestError:
        return 0, {"error": f"Cannot reach {base_url} — is the service running?"}
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


st.title("Document Q&A")
st.caption("Ingest your documents, then ask questions answered only from those documents.")

# Sidebar: point at localhost while developing, at the Render URL for the submission screenshot.
base_url = st.sidebar.text_input("API base URL", DEFAULT_API)

st.sidebar.divider()
if st.sidebar.button("Check index status"):
    status, data = api_get(base_url, "/debug/vectorstore")
    if status == 200 and isinstance(data, dict):
        st.sidebar.success(f"{data['total_vectors']} vectors in `{data['index']}`")
        if not data.get("dimension_matches_model"):
            st.sidebar.error(f"Dimension mismatch: index is {data['dimension']}, model needs 1536")
    else:
        st.sidebar.error(data)

ingest_tab, ask_tab, retrieval_tab = st.tabs(["Ingest", "Ask", "Retrieval debug"])

with ingest_tab:
    st.subheader("Add a document")
    document_id = st.text_input("document_id", "handbook", help="Re-using an ID replaces that document.")
    source = st.text_input("source (optional)", "", placeholder="employee_handbook.txt")
    text = st.text_area("Document text", height=240, placeholder="Paste plain text here...")

    col_size, col_overlap = st.columns(2)
    chunk_size = col_size.number_input("chunk_size", 100, 4000, 800, step=100)
    chunk_overlap = col_overlap.number_input("chunk_overlap", 0, 1000, 100, step=50)

    if st.button("Ingest", type="primary"):
        if not text.strip():
            st.warning("Paste some text first.")
        else:
            payload = {
                "text": text,
                "document_id": document_id,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
            }
            if source.strip():
                payload["source"] = source.strip()

            with st.spinner("Chunking, embedding, upserting..."):
                status, data = api_post(base_url, "/ingest", payload)

            if status == 200 and isinstance(data, dict):
                st.success(f"Indexed **{data['chunks_indexed']}** chunks as `{data['document_id']}`")
            else:
                st.error(data)

with ask_tab:
    st.subheader("Ask a question")
    question = st.text_input("Question", "What is the remote work policy?")

    col_k, col_filter = st.columns(2)
    k = col_k.number_input("top-k chunks", 1, 20, 5)
    filter_doc = col_filter.text_input("limit to document_id (optional)", "")

    if st.button("Ask", type="primary"):
        payload = {"question": question, "k": int(k)}
        if filter_doc.strip():
            payload["document_id"] = filter_doc.strip()

        with st.spinner("Retrieving and generating..."):
            status, data = api_post(base_url, "/ask", payload)

        if status != 200 or not isinstance(data, dict):
            st.error(data)
        else:
            # The answer is nested: response.answer.answer, not response.answer.
            answer = data["answer"]

            if data.get("refused"):
                st.warning(f"**Refused** — {answer['answer']}")
                st.caption("The documents did not contain this answer, so the service declined to guess.")
            else:
                st.success(answer["answer"])

            st.write("**Citations**")
            citations = data.get("citations", [])
            if citations:
                st.table(
                    [
                        {"chunk_id": c["chunk_id"], "document_id": c["document_id"], "score": c["score"]}
                        for c in citations
                    ]
                )
            else:
                st.caption("None — nothing in the index supported an answer.")

            # Showing what was retrieved (even on a refusal) proves the search actually ran.
            retrieved = data.get("retrieved", [])
            if retrieved:
                st.caption(f"Retrieved and considered: {', '.join(f'`{r}`' for r in retrieved)}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tokens", data["tokens_used"])
            m2.metric("Cost (USD)", f"${data['cost_usd']:.5f}")
            m3.metric("Latency", f"{data['latency_ms']} ms")
            m4.metric("Confidence", f"{answer['confidence']:.2f}")

            with st.expander("Raw JSON"):
                st.json(data)

with retrieval_tab:
    st.subheader("Retrieval only — no LLM")
    st.caption("Use this to check which chunks come back before blaming the prompt.")
    debug_q = st.text_input("Question", "What is the remote work policy?", key="debug_q")
    debug_k = st.number_input("top-k", 1, 20, 5, key="debug_k")

    if st.button("Retrieve"):
        with st.spinner("Embedding and searching..."):
            status, data = api_get(base_url, "/debug/retrieve", {"q": debug_q, "k": int(debug_k)})

        if status != 200 or not isinstance(data, dict):
            st.error(data)
        else:
            for match in data["matches"]:
                st.markdown(f"**`{match['chunk_id']}`** — score `{match['score']}`")
                st.caption(match["text"][:400] + ("..." if len(match["text"]) > 400 else ""))
                st.divider()
