"""Answer questions about the NCA Essential Cybersecurity Controls (ECC) with retrieval + an LLM.

Flow: detect the question's language -> embed it with bge-m3 -> take the top-5 most similar
controls in that language -> send them to DeepSeek with strict instructions to answer only from
them and cite control IDs, or refuse.

DeepSeek's API is OpenAI-compatible, so we call it with the `openai` SDK pointed at DeepSeek's URL.

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
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
BASE_URL = "https://api.deepseek.com"

# USD per 1M tokens at DeepSeek's PEAK rates (off-peak is half), so estimates are an upper bound.
# Source: https://api-docs.deepseek.com/quick_start/pricing/ (checked 2026-10-29).
# "hit" = input tokens served from DeepSeek's automatic prompt cache, "miss" = the rest.
PRICES = {
    "deepseek-flash": {"hit": 0.006, "miss": 0.30, "out": 1.20},
    "deepseek-v4-pro": {"hit": 0.044, "miss": 1.32, "out": 3.96},
}

CONTROL_ID = re.compile(r"\b\d{1,2}-\d{1,2}-\d{1,2}(?:-\d{1,2})?\b")
ARABIC = re.compile("[؀-ۿ]")

# Exact strings the model must return when the retrieved controls don't answer the question.
REFUSAL = {"en": "NOT COVERED", "ar": "غير مغطى في الضوابط المتاحة"}
LANGUAGE_NAME = {"en": "English", "ar": "Arabic"}

SYSTEM_PROMPT = f"""You answer questions about the Saudi National Cybersecurity Authority's Essential Cybersecurity Controls (ECC-2:2024).

You will receive some ECC controls, the language of the question, and the question.

Rules:
1. Use ONLY the controls provided. Do not use outside knowledge about ECC, other NCA documents, or other frameworks (e.g. GDPR, ISO 27001, NIST).
2. Reply in the language of the question: English questions get English answers, Arabic questions get Arabic answers.
3. Cite every claim with the control ID in square brackets using Western digits, e.g. [1-2-2] or [2-2-3-1], in both languages. Only cite IDs that appear in the provided controls.
4. Keep it to 1-4 sentences of plain prose. No headings, no preamble.
5. If the provided controls do not answer the question, reply with exactly this text and nothing else:
   - for an English question: {REFUSAL["en"]}
   - for an Arabic question: {REFUSAL["ar"]}
   Do this also when a control only mentions the topic without answering what was asked. Never combine the refusal with a partial answer."""


def is_refusal(answer, lang):
    """True if the answer is exactly the refusal string (ignoring surrounding whitespace/punctuation)."""
    return answer.strip().strip(".。؟?!\"'` ").strip() == REFUSAL[lang]


# --------------------------------------------------------------------------- retrieval

@lru_cache(maxsize=1)
def load_index():
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(INDEX_DIR / "embeddings.npy")
    owners = np.array(json.loads((INDEX_DIR / "units.json").read_text(encoding="utf-8")))
    assert len(owners) == len(vectors), "index is stale: re-run ingest.py"
    return chunks, vectors, owners


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
    chunks, vectors, owners = load_index()
    lang = lang or detect_language(question)
    q = embedder().encode([question], normalize_embeddings=True)[0]
    unit_scores = vectors @ q                  # cosine similarity (vectors are normalised)
    # A control's score is its best-matching unit (the whole control or one of its sub-clauses).
    scores = np.full(len(chunks), -1.0)
    np.maximum.at(scores, owners, unit_scores)
    order = [i for i in np.argsort(-scores) if chunks[i]["lang"] == lang][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in order]


# --------------------------------------------------------------------------- generation

def format_controls(controls):
    return "\n\n".join(
        f'<control id="{c["id"]}" subdomain="{c["subdomain_id"]} {c["subdomain"]}">\n{c["text"]}\n</control>'
        for c in controls
    )


def cost_usd(model, cache_hit_tokens, cache_miss_tokens, output_tokens):
    price = PRICES.get(model)
    if price is None:
        return None
    return (cache_hit_tokens * price["hit"] + cache_miss_tokens * price["miss"]
            + output_tokens * price["out"]) / 1_000_000


def make_client():
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def build_user_message(question, lang, controls):
    return (f"{format_controls(controls)}\n\n"
            f"Question language: {LANGUAGE_NAME[lang]}\n"
            f"Question: {question}")


def cited_ids(answer, controls):
    """Main control IDs cited in the answer, in order, keeping only IDs we actually retrieved."""
    ascii_answer = answer.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))   # in case Arabic digits are used
    retrieved_ids = {c["id"] for c in controls}
    cited = []
    for cid in CONTROL_ID.findall(ascii_answer):
        main = "-".join(cid.split("-")[:3])        # 2-2-3-1 -> 2-2-3
        if main in retrieved_ids and main not in cited:
            cited.append(main)
    return cited


def ask(question, client=None):
    client = client or make_client()
    lang = detect_language(question)
    controls = retrieve(question, lang=lang)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0,                         # most deterministic output: better for answers and evals
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, lang, controls)},
        ],
        # DeepSeek "thinks" by default; short answers from given text don't need it, and it costs tokens.
        extra_body={"thinking": {"type": "disabled"}},
    )
    choice = response.choices[0]
    if choice.finish_reason == "content_filter":   # the API's own filter blocked it (not the same as NOT COVERED)
        answer = "(The model declined to answer this request.)"
    else:
        answer = (choice.message.content or "").strip()

    refused = is_refusal(answer, lang)
    cited = [] if refused else cited_ids(answer, controls)

    usage = response.usage
    # DeepSeek reports how many prompt tokens hit its automatic cache (billed much cheaper).
    hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    miss = usage.prompt_tokens - hit if miss is None else miss
    return {
        "question": question,
        "lang": lang,
        "answer": answer,
        "refused": refused,
        "cited": [c for c in controls if c["id"] in cited],
        "retrieved": controls,
        "model": response.model,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "cached_input_tokens": hit,
            "output_tokens": usage.completion_tokens,
            "cost_usd": cost_usd(MODEL, hit, miss, usage.completion_tokens),
        },
    }


def format_usage(u):
    cost = f"${u['cost_usd']:.5f}" if u["cost_usd"] is not None else "unknown price"
    return (f"tokens in={u['input_tokens']} (cached {u['cached_input_tokens']}) "
            f"out={u['output_tokens']}  est. cost {cost}")


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
    print("\nrefused (not covered):", result["refused"])
    print("cited:", ", ".join(c["id"] for c in result["cited"]) or "none")
    print("retrieved:", ", ".join(f"{c['id']} ({c['score']:.2f})" for c in result["retrieved"]))
    print(f"{result['model']}  {format_usage(result['usage'])}")


if __name__ == "__main__":
    main()
