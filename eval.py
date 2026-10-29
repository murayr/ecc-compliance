"""Evaluate retrieval and answers against eval_set.json, per language.

Metrics (per language):
- retrieval@5:       the labelled control is among the 5 retrieved controls
- citation accuracy: the answer cites the labelled control (needs the API)

Exits with code 1 if any metric is below THRESHOLD, so CI fails on regressions.

Usage:
    python eval.py --retrieval-only    # free: embeddings only, no API calls
    python eval.py                     # full: also calls Claude for every question
"""
import argparse
import json
import sys
from collections import defaultdict

from rag import MODEL, ask, format_usage, retrieve

THRESHOLD = 0.80


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="skip the Claude API (free)")
    ap.add_argument("--file", default="eval_set.json")
    args = ap.parse_args()

    rows = json.load(open(args.file, encoding="utf-8"))
    hits = defaultdict(list)       # (lang, metric) -> [True/False, ...]
    total_in = total_out = total_cost = 0.0

    if not args.retrieval_only:
        import anthropic
        client = anthropic.Anthropic()

    for row in rows:
        lang, expected = row["lang"], row["control_id"]
        retrieved = [c["id"] for c in retrieve(row["question"], lang=lang)]
        hit = expected in retrieved
        hits[(lang, "retrieval@5")].append(hit)
        line = f"[{lang}] {'OK  ' if hit else 'MISS'} rank={retrieved.index(expected) + 1 if hit else '-'}  " \
               f"expected {expected:8} got {','.join(retrieved)}  | {row['question'][:70]}"

        if not args.retrieval_only:
            result = ask(row["question"], client=client)
            cited = [c["id"] for c in result["cited"]]
            ok = expected in cited
            hits[(lang, "citation")].append(ok)
            u = result["usage"]
            total_in += u["input_tokens"]
            total_out += u["output_tokens"]
            total_cost += u["cost_usd"] or 0
            line += f"\n      cited={cited} {'OK' if ok else 'FAIL'}  {format_usage(u)}"
        print(line)

    print("\n=== results ===")
    failed = False
    for (lang, metric), values in sorted(hits.items()):
        score = sum(values) / len(values)
        failed |= score < THRESHOLD
        print(f"{lang}  {metric:18} {score:6.1%}  ({sum(values)}/{len(values)})")
    if not args.retrieval_only:
        print(f"\nmodel {MODEL}: {len(rows)} questions, tokens in={int(total_in)} out={int(total_out)}, "
              f"est. cost ${total_cost:.4f}")
    if failed:
        print(f"\nFAIL: a metric is below {THRESHOLD:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
