"""Session 2 — Streamlit UI for ingest + ask against the RAG API.

Run:
  streamlit run rag_ui.py
  ASK_API_URL=https://your-service.onrender.com streamlit run rag_ui.py

The API is the source of truth: this page only sends HTTP requests to
POST /ingest and POST /ask and displays what comes back. No chunking,
embedding, retrieval, or prompting happens here.
"""

import json
import os

import httpx
import streamlit as st

# Render's free tier can take ~a minute to wake from sleep.
TIMEOUT_SECONDS = 120

st.set_page_config(page_title="Northwind RAG", page_icon="📄", layout="wide")


def call_api(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    """Send one request to the API; return (status code, JSON body or error detail)."""

    url = f"{st.session_state.api_url.rstrip('/')}{path}"
    try:
        response = httpx.request(method, url, json=payload, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return 0, {"detail": f"Could not reach {url}: {type(exc).__name__}"}
    try:
        return response.status_code, response.json()
    except json.JSONDecodeError:
        return response.status_code, {"detail": response.text[:300]}


def error_text(body: dict) -> str:
    """FastAPI errors put the message in 'detail' (a string, or a list for 422s)."""

    detail = body.get("detail", body)
    if isinstance(detail, list):
        return "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
    return str(detail)


# --- Sidebar: which API to talk to -------------------------------------------
with st.sidebar:
    st.header("API")
    st.text_input(
        "Base URL",
        value=os.getenv("ASK_API_URL", "http://127.0.0.1:8000"),
        key="api_url",
        help="Set ASK_API_URL to change the default. No keys live here — the API holds them.",
    )
    if st.button("Check connection"):
        with st.spinner("Contacting the API…"):
            status, body = call_api("GET", "/debug/pinecone")
        if status == 200 and body.get("ok"):
            st.success(f"Connected — {body['vectors']} chunks in index `{body['index']}`")
        else:
            st.error(f"Not ready ({status}): {error_text(body)}")

st.title("Northwind RAG")
st.caption("Ingest a document, then ask questions answered only from what has been ingested.")

ingest_col, ask_col = st.columns(2, gap="large")

# --- Ingest ------------------------------------------------------------------
with ingest_col:
    st.subheader("1. Ingest a document")
    with st.form("ingest"):
        document_id = st.text_input(
            "document_id", placeholder="POL-101", help="Letters, digits, - _ . only"
        )
        source = st.text_input("source (optional)", placeholder="doc1_handbook.txt")
        text = st.text_area("Text", height=260, placeholder="Paste the document text here…")
        submitted = st.form_submit_button("Ingest", type="primary")

    if submitted:
        payload = {"document_id": document_id.strip(), "text": text}
        if source.strip():
            payload["source"] = source.strip()
        with st.spinner("Chunking, embedding, storing…"):
            st.session_state.ingest_result = call_api("POST", "/ingest", payload)

    if "ingest_result" in st.session_state:
        status, body = st.session_state.ingest_result
        if status == 200:
            st.success(
                f"Indexed **{body['document_id']}** — {body['chunks_indexed']} chunks "
                f"(embedding cost ${body['cost_usd']:.6f})"
            )
            st.caption("Re-ingesting the same document_id replaces its chunks.")
        else:
            st.error(f"Ingest failed ({status}): {error_text(body)}")

# --- Ask ---------------------------------------------------------------------
with ask_col:
    st.subheader("2. Ask a question")
    with st.form("ask"):
        question = st.text_input("Question", placeholder="How many remote days are allowed?")
        asked = st.form_submit_button("Ask", type="primary")

    if asked:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Retrieving and answering…"):
                st.session_state.ask_result = call_api("POST", "/ask", {"question": question})
            st.session_state.ask_question = question

    if "ask_result" in st.session_state:
        status, body = st.session_state.ask_result
        if status != 200:
            st.error(f"Ask failed ({status}): {error_text(body)}")
        else:
            answer = body["answer"]
            st.markdown(f"**Q:** {st.session_state.ask_question}")

            if answer["refused"]:
                st.warning(f"**Refused — not in the documents**\n\n{answer['answer']}")
            else:
                st.success(answer["answer"])
                st.markdown(
                    "**Cited:** " + "  ".join(f"`{doc_id}`" for doc_id in answer["citations"])
                    if answer["citations"]
                    else "**Cited:** _no citation returned_"
                )

            st.markdown(
                "**Retrieved chunks:** "
                + "  ".join(f"`{chunk_id}`" for chunk_id in body["retrieved_chunk_ids"])
            )

            # One line rather than st.metric columns, which truncate in a half-width column.
            st.caption(
                f"{body['tokens_used']:,} tokens · ${body['cost_usd']:.6f} · "
                f"{body['latency_ms']:,} ms · {body['model']}"
            )

            with st.expander("Full JSON response"):
                st.json(body)
