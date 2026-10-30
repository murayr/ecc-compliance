"""Evaluate retrieval and answers against eval_set.json, per language.

Row types (eval_set.json, eval_hard.json):
- {"control_id": "2-2-3", ...}       answerable: the labelled control should be retrieved and cited
- {"control_ids": ["2-2-3", ...]}    answerable by several controls: ALL of them must be retrieved and cited
- {"expect": "refusal", ...}         not covered by ECC: the answer must be exactly the refusal string
- {"route": "auto", ...}             detect the question's language as production does (mixed Arabic/English)

Metrics (per language):
- retrieval@5:       every labelled control is among the 5 retrieved controls (answerable rows)
- citation accuracy: the answer cites every labelled control (answerable rows, needs the LLM API)
- refusal accuracy:  the answer is exactly the refusal string (refusal rows, needs the API)
- answer language:   the answer is in the question's language (answerable rows, needs the API)
- citation precision (reported, not gated): share of cited controls that are the labelled one,
  so an answer that cites everything it retrieved can't look perfect

Exits with code 1 if any metric is below THRESHOLD, so CI fails on regressions.

Usage:
    python eval.py --retrieval-only    # free: embeddings only, no API calls
    python eval.py                     # full: also calls DeepSeek for every question
    python eval.py --file eval_hard.json --retrieval-only
    RETRIEVAL_MODE=hybrid python eval.py ...    # how controls are found; see search.py
"""
import argparse
import json
import sys
from collections import defaultdict

from rag import MODEL, ask, detect_language, format_usage, make_client, retrieve

THRESHOLD = 0.80


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM API (free)")
    ap.add_argument("--file", default="eval_set.json")
    ap.add_argument("--budget", type=float, default=1.00,
                    help="stop the full eval once estimated spend exceeds this many USD")
    ap.add_argument("--ids", help="comma-separated row ids to run (e.g. to re-check failures cheaply)")
    ap.add_argument("--save", help="write every row's answer, citations and usage to this JSON file")
    args = ap.parse_args()

    rows = json.load(open(args.file, encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r.get("id") in wanted]
    if args.retrieval_only:
        rows = [r for r in rows if r.get("expect") != "refusal"]   # refusals can only be scored with the API
    else:
        client = make_client()

    hits = defaultdict(list)       # (lang, metric) -> [True/False, ...]
    precision = defaultdict(list)  # lang -> [share of citations that are the labelled control, ...]
    records = []
    total_in = total_out = total_cost = 0

    for row in rows:
        lang, question = row["lang"], row["question"]
        expects_refusal = row.get("expect") == "refusal"
        line = f"[{row.get('id', lang)}] "

        if not expects_refusal:
            expected = row.get("control_ids") or [row["control_id"]]
            routed = None if row.get("route") == "auto" else lang      # None: detect the language, as production does
            retrieved = [c["id"] for c in retrieve(question, lang=routed)]
            hit = all(e in retrieved for e in expected)
            hits[(lang, "retrieval@5")].append(hit)
            ranks = [retrieved.index(e) + 1 if e in retrieved else "-" for e in expected]
            line += (f"{'OK  ' if hit else 'MISS'} ranks={','.join(map(str, ranks))}  "
                     f"expected {','.join(expected):12} got {','.join(retrieved)}")
        else:
            line += "(refusal row)"
        line += f"  | {question[:70]}"

        if not args.retrieval_only:
            if total_cost > args.budget:
                print(f"\nBUDGET: stopped after ${total_cost:.4f} (limit ${args.budget:.2f}); remaining rows skipped")
                sys.exit(1)
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
                cited_ok = all(e in cited for e in expected)
                hits[(lang, "citation")].append(cited_ok)
                if cited:
                    precision[lang].append(sum(c in expected for c in cited) / len(cited))
                same_language = result["refused"] or detect_language(result["answer"]) == lang
                hits[(lang, "language")].append(same_language)
                ok = cited_ok and same_language
                verdict = (f"cited={cited} {'OK' if cited_ok else 'FAIL'}"
                           + (" (refused)" if result["refused"] else "")
                           + ("" if same_language else " WRONG LANGUAGE"))
            line += f"\n      {verdict}  {format_usage(u)}"
            if not ok:
                line += f"\n      answer: {result['answer'][:300]}"
            records.append({**row, "answer": result["answer"], "refused": result["refused"],
                            "cited": [c["id"] for c in result["cited"]],
                            "retrieved": [c["id"] for c in result["retrieved"]], "usage": u, "pass": ok})
        print(line)

    print("\n=== results ===")
    failed = False
    for (lang, metric), values in sorted(hits.items()):
        score = sum(values) / len(values)
        failed |= score < THRESHOLD
        print(f"{lang}  {metric:12} {score:6.1%}  ({sum(values)}/{len(values)})")
    for lang, values in sorted(precision.items()):
        print(f"{lang}  {'precision':12} {sum(values) / len(values):6.1%}  "
              f"(info, not gated: avg share of citations that are labelled controls)")
    if args.save and records:
        json.dump(records, open(args.save, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"saved {len(records)} rows to {args.save}")
    if not args.retrieval_only:
        print(f"\nmodel {MODEL}: {len(rows)} questions, tokens in={total_in} out={total_out}, "
              f"est. cost ${total_cost:.4f} (avg ${total_cost / max(len(rows), 1):.5f}/question)")
    if failed:
        print(f"\nFAIL: a metric is below {THRESHOLD:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
