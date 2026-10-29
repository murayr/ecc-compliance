# ecc-compliance

Bilingual (Arabic/English) question answering over the Saudi NCA **Essential Cybersecurity Controls (ECC-2:2024)**.
Answers cite control IDs (e.g. `2-2-3`) or refuse when the controls don't cover the question.

## How it works

1. **`ingest.py`** parses `data/ECC_2024_en.pdf` and `data/ECC_2024_ar.pdf` into one chunk per control
   (108 controls per language, sub-clauses included), embeds them locally with `BAAI/bge-m3`, and saves `index/`.
2. **`rag.py`** detects the question's language, retrieves the 5 most similar controls in that language
   (cosine similarity, plain numpy), and asks DeepSeek to answer only from them, citing IDs. DeepSeek's
   API is OpenAI-compatible, so it is called with the `openai` SDK pointed at `https://api.deepseek.com`.
3. **`eval.py`** scores retrieval@5 and citation accuracy per language against `eval_set.json`
   and exits 1 if any score is below 80%.

### Why the Arabic PDF needs its own parser

PyMuPDF's plain text for the Arabic PDF is not usable as-is:
- multi-digit numbers come out reversed (`١-٠١` is control 1-10; decree `٠٤١٧٣` is 37140),
- table cells come out in mixed order (an ID sometimes before, sometimes after its text),
- 16 ligatures are mapped to junk symbols (`السيبرا˻` = السيبراني).

So for Arabic, `ingest.py` uses the table geometry: each control is a table row (found from the drawn
cell borders), and each line is rebuilt from character x-positions right-to-left, with Latin and digit
runs flipped back. A validation step checks that both languages yield the same 108 control IDs, in order.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only, much smaller
pip install -r requirements.txt
python ingest.py                                      # downloads bge-m3 (~2 GB) on first run
export DEEPSEEK_API_KEY=...                           # never commit this
python rag.py "Who must approve the cybersecurity strategy?"
```

The model comes from `DEEPSEEK_MODEL` (default `deepseek-flash`; `deepseek-v4-pro` also has prices in
`rag.py`). Thinking mode is turned off and temperature is 0: answers are short and must come from the given text.
Cost estimates use DeepSeek's peak prices, so off-peak runs cost about half what is printed.

## Web app

```bash
uvicorn api:app --port 8000          # backend: POST /ask, GET /health
cd web && npm install && npm run dev  # front end on http://localhost:3000
```

`POST /ask {"question": "..."}` returns the answer, whether it was a refusal, the cited controls with
their full text, and token usage with estimated cost. The front end is one input; the backend detects
the language, Arabic answers render right-to-left, and cited controls are expandable cards (`<details>`).
No UI library: React + plain CSS. Set `NEXT_PUBLIC_API_URL` if the API isn't on `localhost:8000`, and
`ALLOWED_ORIGINS` on the API if the front end isn't on `localhost:3000`.

## Docker

```bash
export DEEPSEEK_API_KEY=...        # PowerShell: $env:DEEPSEEK_API_KEY="..."
docker compose up --build          # API on :8000, front end on http://localhost:3000
```

- `Dockerfile` (API): CPU-only PyTorch, runs `ingest.py` at build time, so the index and the bge-m3
  model are baked into the image and startup takes seconds.
- `web/Dockerfile`: multi-stage Next.js build; the runtime image holds only the standalone server.
- The API key is read from your shell at `docker compose up`; it is never written into an image or file.

## CI (GitHub Actions)

`.github/workflows/eval.yml` runs on every push and pull request: it builds the index and runs
`eval.py --retrieval-only`, failing if retrieval@5 drops below 80% in either language. It is free
(no API calls) and caches the bge-m3 model between runs. The full eval, which calls DeepSeek, is run
locally (`python eval.py --budget 0.25`), so the repository needs no API key secret.

## Evaluation

```bash
python eval.py --retrieval-only    # free, no API calls
python eval.py --budget 0.25       # full eval; stops if estimated spend passes $0.25
python eval.py --ids en-14,ar-r3   # re-run only some rows (cheap re-checks)
```

`eval_set.json` has 25 English + 25 Arabic answerable questions (the same 25 controls in each language;
every row quotes its `evidence` verbatim from the labelled control) and 4 + 4 questions ECC does not cover,
which must get the exact refusal string. `retrieval_smoke.json` is a separate 10 + 10 set used while
developing retrieval, so `eval_set.json` stays a held-out test.

## Eval results

### Retrieval (free, `--retrieval-only`)

| Set | Language | Before | After |
|---|---|---|---|
| eval_set (25/lang) | EN | 100% @5 (48/50 rank 1 overall) | 100% @5 (unchanged) |
| eval_set (25/lang) | AR | 100% @5 | 100% @5 (unchanged) |
| smoke (10/lang) | EN | 90% @5 | 100% @5 |
| smoke (10/lang) | AR | 100% @5 (patch question at rank 3) | 100% @5 (rank 3) |

### Full eval (`deepseek-flash`, 58 questions)

| Metric | EN before | EN after | AR before | AR after |
|---|---|---|---|---|
| citation accuracy (labelled control cited) | 100% (25/25) | 100% | 100% (25/25) | 100% |
| refusal accuracy (exact refusal string) | 100% (4/4) | 100% | 100% (4/4) | 100% |
| answer language matches question | 100% (25/25) | 100% | 100% (25/25) | 100% |
| citation precision (info, not gated) | 80.0% | 80.0% | 84.7% | 90.0% |

"Before/after" is the prompt change in failure 3 below. Average citations per answer went from 1.48 to 1.34.
One full run costs about $0.02 at peak prices (≈59k input and 3k output tokens; ~$0.0003 per question).
The hard refusal rows matter most here: ECC mentions passwords (2-2-3-1) and reporting incidents to the
NCA (2-13-3-3) but sets no password length or reporting deadline, and the model refused instead of
inventing one. Every answer was also read by hand against the control text; none contained a wrong claim.

Caveat: 100% on 58 questions written by one person means the set is too easy to separate good from
better, not that the system is perfect. Next step would be harder questions (multi-control, negations,
near-duplicate controls) and questions written by someone else.

### Failures and fixes

1. **"Should security patches be tested before they are deployed to production?"** (expected 2-10-3).
   *What broke:* embedding. The answer is one sub-clause (2-10-3-4, "verified using a non-production
   environment before being applied") inside a five-clause control about vulnerability management, so the
   control's single vector was mostly about other things; 1-6-2/1-6-3 ("updates packages before launching")
   ranked higher and 2-10-3 dropped out of the top 5.
   *Change:* each sub-clause is also embedded on its own (with its control's intro sentence), and a control
   scores as its best-matching piece. Rank went from >5 to 2 (EN); no regressions on either set.
2. **"Do we need an acceptable use policy for IT assets?"** (expected 2-1-3, rank 2 in both languages).
   *What broke:* nothing to fix. 2-1-3 says to define and approve the policy and 2-1-4 says to apply it;
   the two are almost the same text, and both are in the top 2. Label kept.
3. **"Does business continuity management need to include disaster recovery plans?"** and similar.
   *What broke:* the prompt. Answers were correct but padded with generic sibling controls, e.g. "…disaster
   recovery plans [3-1-3], and these requirements must be identified, documented, and approved [3-1-1],
   implemented [3-1-2], and periodically reviewed [3-1-4]". The citation metric still passed (it only checks
   that the labelled control is cited), so I added a precision measure to make this visible.
   *Change:* one prompt rule: answer only what was asked, without generic "documented / approved /
   implemented / reviewed" controls unless the question is about that. Arabic precision went from 84.7% to
   90.0%; English stayed at 80.0% (its remaining extra citations are mostly defensible, e.g. 2-1-4 "apply the
   acceptable use policy" next to 2-1-3 "define it"). No other metric changed.
