"""Week 2 — RAG building blocks: chunk, embed, and store documents in Pinecone."""

import os
import re
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# text-embedding-3-small: 1536 dimensions, $0.02 per 1M tokens — cheap enough to
# re-ingest freely while experimenting.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_PRICE_PER_1K = 0.00002

# ~800 characters per chunk with ~100 overlap, so a sentence cut at a boundary
# still appears whole in one chunk. Override with CHUNK_SIZE / CHUNK_OVERLAP env
# vars to experiment; re-ingest afterwards so stored chunks match the new size.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


# How many chunks /ask hands the model. 8 rather than 5: with 5, the handbook's
# £350 home-office rule only just made the cut (5th) behind four expense-policy
# chunks that share the word "claim". A few extra chunks cost a fraction of a
# cent per question and keep borderline-but-correct passages in.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 8))


def index_name() -> str:
    """Read at call time, not import time, so values from .env are already loaded."""

    return os.getenv("PINECONE_INDEX", "ask-api")


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    """
    One Pinecone client for the whole process. Lazy so /ask still starts and
    works when PINECONE_API_KEY is not set yet.
    """

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set — add it to .env")
    return Pinecone(api_key=api_key)


@lru_cache(maxsize=1)
def get_index():
    """Connect to the Pinecone index, creating it on first use."""

    pc = get_pinecone()
    if not pc.has_index(index_name()):
        # aws/us-east-1 is the region Pinecone's free Starter plan allows.
        pc.create_index(
            name=index_name(),
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name())


def pinecone_health() -> dict:
    """
    Confirm Pinecone is reachable and the index fits our embedding model.
    Read-only: never creates the index, so a typo in PINECONE_INDEX shows up
    here instead of silently creating a second, empty index.
    """

    pc = get_pinecone()
    name = index_name()
    if not pc.has_index(name):
        return {"reachable": True, "index": name, "exists": False, "ok": False}

    description = pc.describe_index(name)
    stats = pc.Index(name).describe_index_stats()
    # A dimension mismatch makes every upsert fail, so check it up front.
    dimension_ok = description.dimension == EMBEDDING_DIMENSIONS
    return {
        "reachable": True,
        "index": name,
        "exists": True,
        "ready": description.status.ready,
        "dimension": description.dimension,
        "metric": description.metric,
        "embedding_model": EMBEDDING_MODEL,
        "dimension_matches_model": dimension_ok,
        "vectors": stats.total_vector_count,
        "ok": bool(description.status.ready and dimension_ok),
    }


# A section heading is a title line between two rules of '=' signs:
#   ====================
#   2. OUR COMPANY AND VALUES
#   ====================
SECTION_HEADING = re.compile(r"^=+[ \t]*\n([^\n=][^\n]*)\n=+[ \t]*$", re.MULTILINE)


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Split a document at its section headings into (title, body) pairs.
    Text before the first heading comes back with title None; a document with
    no headings comes back whole.
    """

    headings = list(SECTION_HEADING.finditer(text))
    if not headings:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    preamble = text[: headings[0].start()].strip()
    if preamble:
        sections.append((None, preamble))
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[heading.end() : end].strip()
        if body:
            sections.append((heading.group(1).strip(), body))
    return sections


def chunk_text(text: str) -> list[str]:
    """
    Split a document into retrieval-sized pieces, one section at a time.

    Splitting purely by length cut headings off from their content (a chunk
    ending in a bare "2. OUR COMPANY AND VALUES" outranked the values
    themselves) and split lists across chunks. So: split at section headings
    first, then by length inside each section, and start every chunk with
    "<document title> — <section title>" so each piece says where it came from.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
    )
    sections = split_sections(text)
    if len(sections) == 1 and sections[0][0] is None:
        return splitter.split_text(text)

    # The document's first line is its title, e.g. "Northwind Robotics Employee Handbook".
    doc_title = text.strip().splitlines()[0].strip()
    chunks: list[str] = []
    for title, body in sections:
        prefix = f"{doc_title} — {title}\n\n" if title else ""
        chunks.extend(prefix + piece for piece in splitter.split_text(body))
    return chunks


def embed(client: OpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed a batch of texts in one API call; returns vectors and tokens billed."""

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return vectors, response.usage.total_tokens


def chunk_id(document_id: str, chunk_index: int) -> str:
    """Stable ID — re-ingesting the same document overwrites instead of duplicating."""

    return f"{document_id}#{chunk_index}"


def delete_document(document_id: str) -> int:
    """
    Remove every stored chunk of a document. Needed on re-ingest: if the new
    version has fewer chunks, the old tail chunks would otherwise linger and
    keep being retrieved.
    """

    index = get_index()
    stale_ids = [
        item.id
        for page in index.list(prefix=f"{document_id}#")
        for item in page.vectors
    ]
    if stale_ids:
        index.delete(ids=stale_ids)
    return len(stale_ids)


def ingest_document(
    client: OpenAI, document_id: str, text: str, source: str
) -> tuple[int, int]:
    """Chunk, embed, and upsert one document. Returns (chunks stored, embedding tokens)."""

    chunks = chunk_text(text)
    vectors, tokens = embed(client, chunks)

    delete_document(document_id)
    get_index().upsert(
        vectors=[
            {
                "id": chunk_id(document_id, i),
                "values": vector,
                # Keep the text in metadata so retrieval can hand it straight to the model.
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": i,
                    "source": source,
                    "text": chunk,
                },
            }
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
    )
    return len(chunks), tokens


def retrieve(client: OpenAI, question: str, top_k: int = 5) -> tuple[list[dict], int]:
    """
    Find the chunks most similar to a question. Embeds with the same model as
    ingest — vectors from different models are not comparable.
    Returns (chunks best-first, embedding tokens).
    """

    [vector], tokens = embed(client, [question])
    response = get_index().query(vector=vector, top_k=top_k, include_metadata=True)
    chunks = [
        {
            "id": match.id,
            # Cosine similarity: closer to 1 means more similar.
            "score": round(match.score, 4),
            "document_id": match.metadata.get("document_id"),
            "chunk_index": int(match.metadata.get("chunk_index", 0)),
            "source": match.metadata.get("source"),
            "text": match.metadata.get("text", ""),
        }
        for match in response.matches
    ]
    return chunks, tokens


REFUSAL_ANSWER = "I don't have enough information to answer that."

# The course's grounding template, word for word, plus one paragraph telling the
# model how to fill the structured Answer fields (refused, citations, ...) —
# without it the model has no instructions for those fields.
GROUNDING_TEMPLATE = """Answer using ONLY the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."
Cite the document_id of each chunk you used.

Also fill these fields: refused is true only when you gave that sentence;
citations lists the document_ids you cited (empty when refused); confidence is
0 to 1 (0 when refused); sources_needed is true when refused.

Context:
{retrieved_chunks}

Question: {question}"""


def build_grounded_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Fill the grounding template with the retrieved passages and the question."""

    retrieved_chunks = "\n\n".join(
        f"[document_id: {chunk['document_id']} | chunk_id: {chunk['id']}]\n{chunk['text']}"
        for chunk in chunks
    )
    prompt = GROUNDING_TEMPLATE.format(retrieved_chunks=retrieved_chunks, question=question)
    return [{"role": "user", "content": prompt}]


def embedding_cost_usd(tokens: int) -> float:
    """Embedding calls cost money too — surface it like /ask does."""

    return tokens / 1000 * EMBEDDING_PRICE_PER_1K
