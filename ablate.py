"""Compare retrieval modes on the question sets. Free: embeddings and keyword search only, no API calls.

    python ablate.py                          # every mode on every set
    python ablate.py --modes dense,hybrid     # only these modes
    python ablate.py --misses hybrid          # also list the questions that mode gets wrong
    python ablate.py --versus dense           # for each mode, which questions it wins and loses against dense
    python ablate.py --save ablation.json     # every row's ranks, for later analysis

Metrics, per question (a question may need several controls; `control_ids` lists them):
    hit@k    1 if ALL the needed controls are among the top k, else 0
    recall@5 the share of needed controls in the top 5
    MRR      1 / the best rank of any needed control (1.0 = the first result is right)
"""
import argparse
import json
import sys
from collections import defaultdict

import search
from rag import detect_language

SETS = {"dev": "retrieval_smoke.json", "test": "eval_set.json", "hard": "eval_hard.json"}
K = 5


def load(path):
    rows = json.load(open(path, encoding="utf-8"))
    return [r for r in rows if r.get("expect") != "refusal"]     # refusal rows have no control to find


def evaluate(row, mode):
    wanted = row.get("control_ids") or [row["control_id"]]
    lang = detect_language(row["question"]) if row.get("route") == "auto" else row["lang"]
    order = [cid for cid, _ in search.rank(row["question"], lang, mode)]
    ranks = {cid: (order.index(cid) + 1 if cid in order else None) for cid in wanted}
    found = [r for r in ranks.values() if r]
    return {
        "id": row.get("id", row["lang"]), "lang": row["lang"], "category": row.get("category", "-"),
        "routed": lang, "wanted": wanted, "ranks": ranks, "top": order[:K],
        "hit1": all(r and r <= 1 for r in ranks.values()),
        "hit5": all(r and r <= 5 for r in ranks.values()),
        "hit8": all(r and r <= 8 for r in ranks.values()),
        "hit10": all(r and r <= 10 for r in ranks.values()),
        "hit3": all(r and r <= 3 for r in ranks.values()),
        "recall5": sum(1 for r in ranks.values() if r and r <= 5) / len(wanted),
        "mrr": 1 / min(found) if found else 0.0,
    }


def mean(results, key):
    return sum(r[key] for r in results) / len(results) if results else float("nan")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(search.MODES))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--versus", help="baseline mode: list the questions each other mode wins or loses against it")
    ap.add_argument("--misses", help="comma-separated modes whose wrong answers should be listed")
    ap.add_argument("--save")
    args = ap.parse_args()
    modes, sets = args.modes.split(","), args.sets.split(",")

    data = {name: load(SETS[name]) for name in sets}
    results = {(name, mode): [evaluate(r, mode) for r in rows] for name, rows in data.items() for mode in modes}

    print(f"\n## All sets (hit@5 / hit@3 / MRR); n = {', '.join(f'{n} {len(data[n])}' for n in sets)}\n")
    print("| mode | " + " | ".join(sets) + " |\n|---|" + "---|" * len(sets))
    for mode in modes:
        cells = []
        for name in sets:
            rs = results[(name, mode)]
            cells.append(f"{mean(rs, 'hit5'):.0%} / {mean(rs, 'hit3'):.0%} / {mean(rs, 'mrr'):.2f}")
        print(f"| {mode} | " + " | ".join(cells) + " |")

    if "hard" in sets:
        cats = sorted({r["category"] for r in results[("hard", modes[0])]})
        print("\n## Hard set by category (hit@5)\n")
        print("| mode | " + " | ".join(f"{c} (n={sum(1 for r in results[('hard', modes[0])] if r['category'] == c)})" for c in cats) + " | en | ar |")
        print("|---|" + "---|" * (len(cats) + 2))
        for mode in modes:
            rs = results[("hard", mode)]
            row = [f"{mean([r for r in rs if r['category'] == c], 'hit5'):.0%}" for c in cats]
            row += [f"{mean([r for r in rs if r['lang'] == l], 'hit5'):.0%}" for l in ("en", "ar")]
            print(f"| {mode} | " + " | ".join(row) + " |")

    if "hard" in sets:
        print("\n## Hard set: share of questions with ALL needed controls in the top k\n")
        print("| mode | k=1 | k=3 | k=5 | k=8 | k=10 |\n|---|---|---|---|---|---|")
        for mode in modes:
            rs = results[("hard", mode)]
            print(f"| {mode} | " + " | ".join(f"{mean(rs, key):.0%}" for key in ("hit1", "hit3", "hit5", "hit8", "hit10")) + " |")

    if args.versus:
        print(f"\n## Paired comparison against {args.versus} (hit@5, all sets together)\n")
        for mode in modes:
            if mode == args.versus:
                continue
            wins, losses = [], []
            for name in sets:
                for base, other in zip(results[(name, args.versus)], results[(name, mode)]):
                    if other["hit5"] and not base["hit5"]:
                        wins.append(f"{name}/{base['id']}")
                    elif base["hit5"] and not other["hit5"]:
                        losses.append(f"{name}/{base['id']}")
            print(f"- {mode}: wins {len(wins)} {wins} | losses {len(losses)} {losses}")

    for mode in (args.misses.split(",") if args.misses else []):
        print(f"\n## Questions {mode} gets wrong (top {K})\n")
        for name in sets:
            for r in results[(name, mode)]:
                if not r["hit5"]:
                    print(f"- [{name}] {r['id']} wanted {r['wanted']} ranks {r['ranks']} got {r['top']}")

    if args.save:
        json.dump({f"{n}/{m}": rs for (n, m), rs in results.items()}, open(args.save, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
