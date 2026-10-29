"""Answer questions about the NCA Essential Cybersecurity Controls (ECC) with retrieval + Claude.

Flow: detect the question's language -> embed it with bge-m3 -> take the top-5 most similar
controls in that language -> send them to Claude with strict instructions to answer only from
them and cite control IDs, or refuse.

Usage:
    python rag.py "Who must approve the cybersecurity strategy?"
    python rag.py --retrieve-only "..."      # show retrieved controls, no API call
"""
import argparse
import json
import os
import re
import sys
from functools import lru_cache

import numpy as np

from ingest import INDEX_DIR, get_model

TOP_K = 5
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

# USD per million tokens (input, output). Update if you change CLAUDE_MODEL.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

CONTROL_ID = re.compile(r"\b\d{1,2}-\d{1,2}-\d{1,2}(?:-\d{1,2})?\b")
ARABIC = re.compile("[؀-ۿ]")

REFUSAL = {"en": "NOT COVERED", "ar": "غير مغطى في الضوابط المتاحة"}

SYSTEM_PROMPT = """You answer questions about the Saudi National Cybersecurity Authority's Essential Cybersecurity Controls (ECC-2:2024).

You will receive a few ECC controls and a question. Rules:
- Answer ONLY from the controls provided. Do not use outside knowledge about ECC or any other framework.
- Cite every claim with the control ID in square brackets, e.g. [1-2-2] or [2-2-3-1]. Only cite IDs that appear in the provided controls.
- Answer in plain prose, 1-4 sentences. No headings, no preamble.
- If the provided controls do not answer the question, reply with exactly: I cannot answer this from the provided ECC controls."""


# --------------------------------------------------------------------------- retrieval

@lru_cache(maxsize=1)
def load_index():
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(INDEX_DIR / "embeddings.npy")
    assert len(chunks) == len(vectors), "index is stale: re-run ingest.py"
    return chunks, vectors


@lru_cache(maxsize=1)
def embedder():
    return get_model()


def detect_language(text):
    """'ar' if most letters are Arabic, else 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    arabic = sum(1 for ch in letters if ARABIC.match(ch))
    return "ar" if letters and arabic / len(letters) > 0.5 else "en"


def retrieve(question, k=TOP_K, lang=None):
    """Return the k controls (in the question's language) most similar to the question."""
    chunks, vectors = load_index()
    lang = lang or detect_language(question)
    q = embedder().encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q                       # cosine similarity (vectors are normalised)
    order = [i for i in np.argsort(-scores) if chunks[i]["lang"] == lang][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in order]


# --------------------------------------------------------------------------- generation

def format_controls(controls):
    return "\n\n".join(
        f'<control id="{c["id"]}" subdomain="{c["subdomain_id"]} {c["subdomain"]}">\n{c["text"]}\n</control>'
        for c in controls
    )


def cost_usd(model, input_tokens, output_tokens):
    price_in, price_out = PRICES.get(model, (None, None))
    if price_in is None:
        return None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def ask(question, client=None):
    import anthropic
    client = client or anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
    lang = detect_language(question)
    controls = retrieve(question, lang=lang)

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},       # short factual answers don't need deep reasoning
        messages=[{"role": "user", "content": f"{format_controls(controls)}\n\nQuestion: {question}"}],
    )
    if response.stop_reason == "refusal":
        answer = REFUSAL[lang]
    else:
        answer = "".join(b.text for b in response.content if b.type == "text").strip()

    # Normalise Arabic-Indic digits so IDs cited in Arabic answers match our ASCII IDs.
    ascii_answer = answer.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    retrieved_ids = {c["id"] for c in controls}
    cited = []
    for cid in CONTROL_ID.findall(ascii_answer):
        main = "-".join(cid.split("-")[:3])        # 2-2-3-1 -> 2-2-3
        if main in retrieved_ids and main not in cited:
            cited.append(main)

    usage = response.usage
    return {
        "question": question,
        "lang": lang,
        "answer": answer,
        "cited": [c for c in controls if c["id"] in cited],
        "retrieved": controls,
        "model": response.model,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost_usd(response.model, usage.input_tokens, usage.output_tokens),
        },
    }


def format_usage(u):
    cost = f"${u['cost_usd']:.5f}" if u["cost_usd"] is not None else "unknown price"
    return f"tokens in={u['input_tokens']} out={u['output_tokens']}  est. cost {cost}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--retrieve-only", action="store_true", help="show retrieval results, no API call")
    args = ap.parse_args()

    if args.retrieve_only:
        for c in retrieve(args.question):
            print(f"{c['score']:.3f}  {c['id']:8}  {c['subdomain']}  |  {c['text'][:90]}")
        return

    result = ask(args.question)
    print(result["answer"])
    print("\ncited:", ", ".join(c["id"] for c in result["cited"]) or "none")
    print("retrieved:", ", ".join(f"{c['id']} ({c['score']:.2f})" for c in result["retrieved"]))
    print(f"{result['model']}  {format_usage(result['usage'])}")


if __name__ == "__main__":
    main()
