"""Evaluate retrieval and answers against eval_set.json, per language.

Row types in eval_set.json:
- {"control_id": "2-2-3", ...}  answerable: the labelled control should be retrieved and cited
- {"expect": "refusal", ...}    not covered by ECC: the answer must be exactly the refusal string

Metrics (per language):
- retrieval@5:       labelled control is among the 5 retrieved controls (answerable rows)
- citation accuracy: the answer cites the labelled control (answerable rows, needs the API)
- refusal accuracy:  the answer is exactly the refusal string (refusal rows, needs the API)

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
    if args.retrieval_only:
        rows = [r for r in rows if r.get("expect") != "refusal"]   # refusals can only be scored with the API
    else:
        import anthropic
        client = anthropic.Anthropic()

    hits = defaultdict(list)       # (lang, metric) -> [True/False, ...]
    total_in = total_out = total_cost = 0

    for row in rows:
        lang, question = row["lang"], row["question"]
        expects_refusal = row.get("expect") == "refusal"
        line = f"[{row.get('id', lang)}] "

        if not expects_refusal:
            expected = row["control_id"]
            retrieved = [c["id"] for c in retrieve(question, lang=lang)]
            hit = expected in retrieved
            hits[(lang, "retrieval@5")].append(hit)
            rank = retrieved.index(expected) + 1 if hit else "-"
            line += f"{'OK  ' if hit else 'MISS'} rank={rank}  expected {expected:7} got {','.join(retrieved)}"
        else:
            line += "(refusal row)"
        line += f"  | {question[:70]}"

        if not args.retrieval_only:
            result = ask(question, client=client)
            u = result["usage"]
            total_in += u["input_tokens"]
            total_out += u["output_tokens"]
            total_cost += u["cost_usd"] or 0
            if expects_refusal:
                ok = result["refused"]      # pass = refused; fail = answered anyway
                hits[(lang, "refusal")].append(ok)
                verdict = "refused OK" if ok else "ANSWERED (should refuse)"
            else:
                cited = [c["id"] for c in result["cited"]]
                ok = expected in cited
                hits[(lang, "citation")].append(ok)
                verdict = f"cited={cited} {'OK' if ok else 'FAIL'}" + (" (refused)" if result["refused"] else "")
            line += f"\n      {verdict}  {format_usage(u)}"
            if not ok:
                line += f"\n      answer: {result['answer'][:300]}"
        print(line)

    print("\n=== results ===")
    failed = False
    for (lang, metric), values in sorted(hits.items()):
        score = sum(values) / len(values)
        failed |= score < THRESHOLD
        print(f"{lang}  {metric:12} {score:6.1%}  ({sum(values)}/{len(values)})")
    if not args.retrieval_only:
        print(f"\nmodel {MODEL}: {len(rows)} questions, tokens in={total_in} out={total_out}, "
              f"est. cost ${total_cost:.4f} (avg ${total_cost / max(len(rows), 1):.5f}/question)")
    if failed:
        print(f"\nFAIL: a metric is below {THRESHOLD:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
