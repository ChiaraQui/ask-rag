"""
Golden-set eval for the RAG service: known-answer questions plus refusal cases.

Usage (server must be running; OPENAI_API_KEY in .env for the faithfulness judge):
    python eval_golden.py                                   # local
    python eval_golden.py https://your-service.onrender.com  # deployed

Writes a markdown table to eval_results.md and prints it.

Per question it records three things, each checked separately so a failure
points at the stage that broke:
  - retrieval hit: a chunk from the expected document that contains the
    evidence sentence is in the top 5 (GET /debug/retrieve, no LLM)
  - faithfulness:  every claim in the answer is supported by the passages the
    model was shown (an LLM judge reads passages + answer)
  - correctness:   the answer contains the expected facts and cites the expected
    document (plain string checks, no LLM)
For refusal questions, all that matters is that /ask refused and cited nothing.
"""

import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

RETRIEVAL_K = 5  # "right chunk in top-5"
ASK_K = 8  # /ask shows the model RAG_TOP_K = 8 chunks; the judge must see the same ones
JUDGE_MODEL = "gpt-4o-mini"

# expected_facts: each inner list is one fact; any spelling in it counts.
# evidence: text that must appear in a retrieved chunk for a retrieval hit.
GOLDEN_SET = [
    {
        "question": "How many remote days are allowed?",
        "expected": "Up to 3 days per week",
        "document_id": "POL-101",
        "evidence": ["up to three days per week"],
        "expected_facts": [["three", "3"]],
    },
    {
        "question": "What is the mileage rate?",
        "expected": "45p per mile over 50 miles",
        "document_id": "POL-114",
        "evidence": ["45 pence per mile"],
        "expected_facts": [["45"]],
    },
    {
        "question": "How quickly must a lost laptop be reported?",
        "expected": "Within 1 hour",
        "document_id": "POL-207",
        "evidence": ["within one hour"],
        "expected_facts": [["one hour", "1 hour"]],
    },
    {
        "question": "What is the WB-9 payload limit?",
        "expected": "25 kg",
        "document_id": "SPEC-WB9",
        "evidence": ["payload capacity of 25 kg"],
        "expected_facts": [["25 kg", "25kg"]],
    },
    {
        "question": "What are the company's values?",
        "expected": "Safety first, Customer truth, Own the outcome, Teach what you learn",
        "document_id": "POL-101",
        # All four in one chunk — the case that section-aware chunking fixed.
        "evidence": ["Safety first", "Customer truth", "Own the outcome", "Teach what you learn"],
        "expected_facts": [
            ["safety first"],
            ["customer truth"],
            ["own the outcome"],
            ["teach what you learn"],
        ],
    },
    {
        "question": "How much can I claim for home office equipment?",
        "expected": "Up to £350 every 36 months",
        "document_id": "POL-101",
        "evidence": ["£350"],
        "expected_facts": [["350"]],
    },
    {
        "question": "What is the parental leave policy?",
        "expected": "Refuse (not in docs)",
        "document_id": None,
    },
    {
        "question": "Do employees get a company car?",
        "expected": "Refuse (not in docs)",
        "document_id": None,
    },
]


class Verdict(BaseModel):
    grounded: bool
    reason: str


def judge_faithfulness(judge: OpenAI, answer: str, passages: list[dict]) -> Verdict:
    """Ask an LLM whether every claim in the answer is supported by the passages."""

    context = "\n\n".join(f"[{p['id']}]\n{p['text']}" for p in passages)
    completion = judge.chat.completions.parse(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are grading a RAG system. Decide whether EVERY factual claim in the "
                    "answer is directly supported by the passages. Paraphrase is fine; any claim "
                    "not found in the passages makes it ungrounded. Ignore citation labels like "
                    "(POL-101).\n\n"
                    f"Passages:\n{context}\n\nAnswer:\n{answer}"
                ),
            }
        ],
        response_format=Verdict,
    )
    return completion.choices[0].message.parsed


def contains_all(text: str, facts: list[list[str]]) -> bool:
    text = text.lower()
    return all(any(option.lower() in text for option in fact) for fact in facts)


def mark(value: bool | None) -> str:
    return "n/a" if value is None else ("✅" if value else "❌")


def main() -> None:
    base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    judge = OpenAI()
    rows = []

    with httpx.Client(base_url=base_url, timeout=120) as http:
        for case in GOLDEN_SET:
            question = case["question"]
            print(f"… {question}")
            ask = http.post("/ask", json={"question": question}).json()
            answer = ask["answer"]
            # Same question → same embedding → the same chunks /ask was given.
            shown = http.get("/debug/retrieve", params={"q": question, "k": ASK_K}).json()["results"]

            if case["document_id"] is None:
                refused_cleanly = answer["refused"] and not answer["citations"]
                hit, faithful, correct = None, refused_cleanly, refused_cleanly
                note = "refused" if refused_cleanly else f"answered: {answer['answer'][:60]}"
            else:
                top = shown[:RETRIEVAL_K]
                hit_chunk = next(
                    (
                        c
                        for c in top
                        if c["document_id"] == case["document_id"]
                        and all(e.lower() in c["text"].lower() for e in case["evidence"])
                    ),
                    None,
                )
                hit = hit_chunk is not None
                if answer["refused"]:
                    faithful, correct = True, False  # Refusing invents nothing, but it is wrong here.
                    note = "refused a question the docs answer"
                else:
                    verdict = judge_faithfulness(judge, answer["answer"], shown)
                    faithful = verdict.grounded
                    correct = contains_all(answer["answer"], case["expected_facts"]) and (
                        case["document_id"] in answer["citations"]
                    )
                    rank = top.index(hit_chunk) + 1 if hit_chunk else None
                    note = f"evidence at rank {rank}" if rank else "evidence not in top 5"
                    if not faithful:
                        note += f"; judge: {verdict.reason}"

            rows.append(
                {
                    "question": question,
                    "expected": case["expected"],
                    "got": answer["answer"].replace("\n", " ").replace("|", "/"),
                    "cited": ", ".join(answer["citations"]) or "—",
                    "hit": hit,
                    "faithful": faithful,
                    "correct": correct,
                    "note": note,
                }
            )

    answerable = [r for r in rows if r["hit"] is not None]
    lines = [
        "| # | Question | Expected | Got | Cited | Retrieval hit (top-5) | Faithful | Correct | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, r in enumerate(rows, start=1):
        lines.append(
            f"| {n} | {r['question']} | {r['expected']} | {r['got']} | {r['cited']} | "
            f"{mark(r['hit'])} | {mark(r['faithful'])} | {mark(r['correct'])} | {r['note']} |"
        )
    lines += [
        "",
        f"Retrieval hit: {sum(r['hit'] for r in answerable)}/{len(answerable)} · "
        f"Faithful: {sum(r['faithful'] for r in rows)}/{len(rows)} · "
        f"Correct (incl. refusals): {sum(r['correct'] for r in rows)}/{len(rows)}",
    ]
    table = "\n".join(lines)

    out = Path(__file__).resolve().parent / "eval_results.md"
    out.write_text(f"# Golden-set eval\n\nService: `/ask` (k={ASK_K}), judge: {JUDGE_MODEL}\n\n{table}\n")
    print("\n" + table + f"\n\nSaved to {out.name}")


if __name__ == "__main__":
    main()
