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
their full text, and token usage with estimated cost.

The front end is a single static page with a conversation layout, built Arabic-first for Saudi
organisations: the document is Arabic and right-to-left, Arabic labels and examples lead, and English
answers switch to left-to-right inside their own bubbles. Fonts are IBM Plex Sans Arabic and IBM Plex Sans
(designed as one family, self-hosted by `next/font`). Inline citations such as `[2-2-3]` open the matching
control card; "not covered" answers have their own state. No UI library: React and plain CSS with logical
properties, so one stylesheet serves both directions; animations respect `prefers-reduced-motion`. Set `NEXT_PUBLIC_API_URL` if the API isn't on `localhost:8000`, and
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
python eval.py --file eval_hard.json --retrieval-only   # the harder set, below
python ablate.py                   # compare retrieval modes on every set (free)
python test_search.py              # unit tests for the hand-written retrieval code (no models)
```

`eval_set.json` has 25 English + 25 Arabic answerable questions (the same 25 controls in each language;
every row quotes its `evidence` verbatim from the labelled control) and 4 + 4 questions ECC does not cover,
which must get the exact refusal string. `retrieval_smoke.json` is a separate 10 + 10 set used while
developing retrieval, so `eval_set.json` stays a held-out test.

`eval_hard.json` has 46 harder questions (written before any retrieval variant was run, so they were not
tuned to the results): near-identical lifecycle controls ("is it reviewed?" vs "is it implemented?"),
questions that need two controls at once (all must be retrieved), one sub-clause buried in a long
control, everyday wording with no shared vocabulary, and Arabic questions containing English terms
(SIEM, DDoS, Patch Management). Rows list `control_ids` and a verbatim `evidence` quote per control; the
quotes are machine-checked against the parsed text, but the labels themselves still need a human
who knows the controls to confirm them. They were written by the same person (an AI assistant) who built
the system, not independently.

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

### Retrieval experiments (free; `python ablate.py`)

Retrieval modes compared on three sets: dev (20, used while building), test (50) and hard (46).
Cells are hit@5 / hit@3 / MRR, where a question counts as a hit only if **all** the controls it needs are in
the top k. Full tables, per-question ranks and every question a mode gets wrong are in `results/`.

| mode | what it is | dev | test | hard |
|---|---|---|---|---|
| `dense_whole` | one vector per control | 95 / 95 / 0.92 | 100 / 100 / 0.98 | 89 / 70 / 0.76 |
| `dense` (default) | vectors for each control and each sub-clause | 100 / 100 / 0.94 | 100 / 100 / 0.98 | 93 / 80 / 0.79 |
| `bm25` | keyword search only | 90 / 90 / 0.76 | 100 / 100 / 0.94 | 70 / 57 / 0.73 |
| `hybrid` | dense + keyword, rank fusion | 95 / 95 / 0.95 | 100 / 100 / 0.98 | 91 / 85 / 0.81 |
| `dense_x` | dense in both languages | 100 / 100 / 0.94 | 100 / 100 / 0.98 | 93 / 85 / 0.80 |
| `hybrid_x` | hybrid in both languages | 80 / 65 / 0.68 | 90 / 80 / 0.80 | 85 / 74 / 0.76 |
| **`dense+rerank`** | dense, then a cross-encoder re-scores the top 10 | 100 / 100 / 1.00 | 100 / 100 / 0.99 | **98 / 93 / 0.91** |
| `hybrid+rerank` | hybrid, then the same reranker | 95 / 95 / 0.95 | 100 / 100 / 0.99 | 91 / 89 / 0.90 |

What the numbers say, including what they don't:

- **Sub-clause vectors help, on questions they were not tuned on.** `dense_whole` loses 3 questions to `dense`
  and wins none. The step 5 change was made on the dev set; this is the held-out confirmation.
- **Keyword search does not earn its place here.** Alone it is strong on buried detail (92%) and collapses on
  everyday wording (12%). Fused with dense it wins 1 question and loses 3 at hit@5 (it ranks the top 3 a little
  better). `hybrid_x` is worse than either: fusing four rankings gives the noisy keyword signal double weight.
  That is a design mistake I am reporting, not tuning away.
- **The reranker is the one clear win:** `dense+rerank` wins 2 hard questions and loses none, and the dev and
  test sets do not move. It reads the question and the control together instead of comparing two vectors. It
  only works because `dense` already finds every needed control by rank 10 (hit@10 is 100%); it fixes
  crowding, not blindness. `hybrid+rerank` is worse because the keyword signal pushes needed controls out of
  the 10 candidates the reranker sees.
- **With the LLM** (hard set, `deepseek-flash`, about $0.014 per run): citation accuracy 90% EN / 96% AR with
  `dense`, 95% EN / 100% AR with `dense+rerank`; the answer language was right on all 46 either way. The
  three `dense` failures were exactly its three retrieval misses, and in each the model said the controls did not
  address the point or refused, instead of inventing an answer. Citation precision fell from about 92% to
  86% with the reranker, because more related controls reach the prompt and get cited; my labels are a
  minimum set, so this is soft.
- **Cost of the reranker: about 5.6 s per question on this CPU** (0.18 s without it), plus a 2.3 GB model.
  So it is **off by default**: set `RETRIEVAL_MODE=dense+rerank` for accuracy-first use or on a GPU. It is not
  baked into the Docker image.
- **Still failing:** "What must happen to a leaver's system access and to company data on their personal
  devices?" needs 1-9-5 ("personnel's powers shall be reviewed and revoked"), which uses different words
  from the question; no mode gets it into the top 5.

How much to trust this: 46 hard questions means one question is 2.2 points, so differences of one or two
questions are suggestive, not proven. Nothing was tuned on the hard set (reciprocal rank fusion uses the
standard constant of 60; the reranker sees the top 10, fixed before running it), but the questions were written
by the system's author, and a compliance officer's own questions would be a harder test.

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
4. **"ما متطلبات الـ Patch Management؟"** (an Arabic question with an English term), found by the hard set.
   *What broke:* language routing. The product decided the language by counting letters, and "Patch Management"
   has more letters than the short Arabic words around it, so the question was treated as English: the
   English text was searched and the answer would have come back in English.
   *Change:* decide by words, with ties going to Arabic. All 124 questions across the three sets are now routed
   correctly (before: 123 of 124), and `test_search.py` covers it.
5. **Control 2-9-4 (backup requirements) in Arabic lacks the word "periodically".** The English says
   "shall be periodically reviewed"; the Arabic says only "يجب مراجعة تطبيق متطلبات الأمن السيبراني لإدارة
   النسخ الاحتياطية للجهة". I checked the raw PDF page and it is a real difference between the two official
   documents, not a parsing error, so the Arabic question about it is worded without the word.
