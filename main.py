"""Week 1 live demo — five stages in one file, built up live in class."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

import rag

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Browsers refuse cross-origin calls unless the API opts in, so a React frontend
# needs this. Add the deployed frontend's URL to ALLOWED_ORIGINS in Render
# (comma-separated) - keeping it to a known list rather than "*" means a random
# page cannot spend our credit from a visitor's browser.
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Stage 4 default — gpt-4o-mini keeps a public endpoint cheap (~17x less than
# gpt-4o per call); pass "model" per request to compare costs in the demo.
DEFAULT_MODEL = "gpt-4o-mini"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    # Session 2 — RAG grounding.
    refused: bool = Field(description="True when the context does not contain the answer")
    citations: list[str] = Field(description="document_id of every passage used; empty if refused")


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int  # Generation + question-embedding tokens.
    model: str
    latency_ms: int  # Retrieval + generation.
    cost_usd: float  # Generation + question-embedding cost.
    retrieved_chunk_ids: list[str]  # Everything the model was shown, best match first.


class IngestRequest(BaseModel):
    """One plain-text document. The ID becomes the prefix of every chunk ID, so
    it is limited to characters Pinecone IDs accept and cannot contain '#'.
    Emptiness is checked in the route so it returns a clear 400."""

    document_id: str = Field(max_length=100, pattern=r"^[A-Za-z0-9_.-]*$")
    text: str
    source: str | None = Field(default=None, max_length=200)  # e.g. the original filename


class IngestResponse(BaseModel):
    """What was stored and what it cost, mirroring /ask's cost readout."""

    document_id: str
    chunks_indexed: int
    status: str
    embedding_tokens: int
    cost_usd: float


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(messages: list[dict], model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=messages,
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


def call_model_unsafe(messages: list[dict], model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            *messages[:-1],
            {
                "role": "user",
                "content": (
                    f"{messages[-1]['content']}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, "
                    "sources_needed, refused, citations. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            },
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


@app.get("/")
def root():
    """Landing route so the base URL is useful instead of a bare 404."""

    return {
        "service": "ask-api",
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["POST /ask", "POST /ingest"],
        "example": {"question": "What is RAG in one sentence?"},
    }


# Example:
#   curl -s -X POST http://127.0.0.1:8000/ingest \
#     -H "Content-Type: application/json" \
#     -d '{"document_id": "POL-114", "source": "doc2_expenses.txt",
#          "text": "Mileage is reimbursed at 45p per mile over 50 miles..."}'
#
# Re-sending the same document_id replaces that document's chunks.
@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk, embed, and store a document so /ask can answer from it."""

    if not body.document_id:
        raise HTTPException(status_code=400, detail="document_id must not be empty")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        chunks, tokens = rag.ingest_document(
            client, body.document_id, body.text, source=body.source or body.document_id
        )
    except RuntimeError as exc:
        # Missing Pinecone config is a server setup problem, not the caller's fault.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestResponse(
        document_id=body.document_id,
        chunks_indexed=chunks,
        status="indexed",
        embedding_tokens=tokens,
        cost_usd=round(rag.embedding_cost_usd(tokens), 6),
    )


@app.get("/debug/pinecone")
def debug_pinecone():
    """Health check: is Pinecone reachable, and does the index fit our embeddings?"""

    try:
        return rag.pinecone_health()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Bad key, network down, Pinecone outage — report it rather than a bare 500.
        raise HTTPException(
            status_code=503, detail=f"Pinecone unreachable: {type(exc).__name__}"
        ) from exc


# Example: curl -s "http://127.0.0.1:8000/debug/retrieve?q=What+is+the+mileage+rate"
@app.get("/debug/retrieve")
def debug_retrieve(
    q: str = Query(description="The question to search for"),
    k: int = Query(default=5, ge=1, le=20, description="How many chunks to return"),
):
    """
    Show what retrieval finds for a question — top-k chunks with scores — WITHOUT
    calling the LLM. If the right passage is not here, /ask cannot answer well,
    so check this first.
    """

    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")

    try:
        chunks, tokens = rag.retrieve(client, q, top_k=k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "question": q,
        "top_k": k,
        "results": chunks,
        "embedding_tokens": tokens,
        "cost_usd": round(rag.embedding_cost_usd(tokens), 8),
    }


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question from the ingested documents, with citations, refusal,
    structured output, guardrails, and cost visibility."""

    model = body.model or DEFAULT_MODEL
    last_error: str | None = None
    start = time.perf_counter()

    # Session 2: retrieve first. The model only ever sees these passages.
    try:
        chunks, embedding_tokens = rag.retrieve(client, body.question, top_k=rag.RAG_TOP_K)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    embedding_cost = rag.embedding_cost_usd(embedding_tokens)
    chunk_ids = [chunk["id"] for chunk in chunks]

    if not chunks:
        # Nothing ingested yet — refuse without paying for a model call.
        return AskResponse(
            answer=Answer(
                answer=rag.REFUSAL_ANSWER,
                confidence=0.0,
                sources_needed=True,
                refused=True,
                citations=[],
            ),
            tokens_used=embedding_tokens,
            model=model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cost_usd=round(embedding_cost, 6),
            retrieved_chunk_ids=[],
        )

    messages = rag.build_grounded_messages(body.question, chunks)
    retrieved_document_ids = {chunk["document_id"] for chunk in chunks}

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    messages, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    messages, model
                )

            # Guardrail: a citation must point at a document the model was actually
            # shown — drop anything invented, and a refusal cites nothing. The model
            # sometimes cites a chunk ID ("POL-114#3"); reduce it to its document ID.
            cited = (c.strip().split("#")[0] for c in answer.citations)
            answer.citations = (
                []
                if answer.refused
                else list(dict.fromkeys(c for c in cited if c in retrieved_document_ids))
            )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens) + embedding_cost

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used + embedding_tokens,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=chunk_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
