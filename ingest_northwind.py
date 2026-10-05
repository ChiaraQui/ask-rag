"""
Ingest the Northwind sample docs through POST /ingest.

Usage (server must be running):
    python ingest_northwind.py                                   # local
    python ingest_northwind.py https://your-service.onrender.com  # deployed

Goes through the API rather than calling rag.py directly, so it exercises the
same path a real client uses. Safe to re-run: each document_id replaces its
previous chunks instead of duplicating them.
"""

import sys
import time
from pathlib import Path

import httpx

DOCS_DIR = Path(__file__).resolve().parent / "docs" / "northwind"

# Stable IDs from docs/northwind/README.md — citations show these, so they must
# never change between runs.
DOCUMENTS = {
    "POL-101": "doc1_handbook.txt",
    "POL-114": "doc2_expenses.txt",
    "POL-207": "doc3_security.txt",
    "SPEC-WB9": "doc4_product.txt",
    "POL-220": "doc5_it_acceptable_use.txt",
    "POL-118": "doc6_facilities.txt",
}


def main() -> None:
    base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    print(f"Ingesting {len(DOCUMENTS)} documents into {base_url}\n")

    # Render's free tier can take ~a minute to wake up, hence the long timeout.
    with httpx.Client(base_url=base_url, timeout=120) as http:
        ingested = 0
        total_cost = 0.0
        for document_id, filename in DOCUMENTS.items():
            text = (DOCS_DIR / filename).read_text(encoding="utf-8")
            response = http.post(
                "/ingest",
                json={"document_id": document_id, "source": filename, "text": text},
            )
            response.raise_for_status()
            result = response.json()
            ingested += result["chunks_indexed"]
            total_cost += result["cost_usd"]
            print(f"  {document_id:<9} {filename:<28} {result['chunks_indexed']:>3} chunks")

        print(f"\nChunks ingested this run: {ingested} (embedding cost ${total_cost:.6f})")

        # Pinecone updates its counts a few seconds after an upsert, so poll
        # until the total catches up instead of printing a stale number.
        for _ in range(15):
            vectors = http.get("/debug/pinecone").json()["vectors"]
            if vectors >= ingested:
                break
            time.sleep(2)
        print(f"Total chunks in the vector store: {vectors}")


if __name__ == "__main__":
    main()
