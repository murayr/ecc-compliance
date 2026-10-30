"""Retrieval: a few scoring signals, and ways to combine them.

Every signal scores all 108 controls of one language for a question. A control's score from a signal is
its best-matching unit (the whole control, or one of its sub-clauses). Modes combine signals:

    dense_whole  one vector per control (the original design)
    dense        vectors for the control and for each sub-clause (the current default)
    bm25         keyword search (exact words: SIEM, DDoS, "periodically")
    hybrid       dense + bm25, merged by reciprocal rank fusion
    dense_x      dense in the question's language and in the other one (control IDs are the same in both)
    hybrid_x     dense and bm25 in both languages

Any mode can end in "+rerank" (for example "dense+rerank"): the top 10 controls are re-scored by a
cross-encoder, which reads the question and the control together instead of comparing two separate vectors.

Reciprocal rank fusion: each signal ranks the controls; a control scores the sum of 1 / (60 + rank) over the
signals that ranked it. It needs no score scaling and no tuning; 60 is the value from the original paper.
"""
import json
import re
import unicodedata
from functools import lru_cache

import numpy as np

from ingest import INDEX_DIR, get_model, search_units

RRF_K = 60
OTHER = {"en": "ar", "ar": "en"}
RERANKER = "BAAI/bge-reranker-v2-m3"
RERANK_TOP = 10          # how many candidates the cross-encoder re-scores (fixed up front, not tuned)

# mode -> signals; each signal is (name, "same" language as the question or the "other" one)
MODES = {
    "dense_whole": [("dense_whole", "same")],
    "dense": [("dense", "same")],
    "bm25": [("bm25", "same")],
    "hybrid": [("dense", "same"), ("bm25", "same")],
    "dense_x": [("dense", "same"), ("dense", "other")],
    "hybrid_x": [("dense", "same"), ("dense", "other"), ("bm25", "same"), ("bm25", "other")],
}


# --------------------------------------------------------------------------- keyword signal (BM25)

AR_MARKS = re.compile("[ً-ْـ]")                 # short vowels and tatweel
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
AR_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})
TOKEN = re.compile(r"[a-z0-9]+|[ء-ي]+")
AR_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال")
AR_SUFFIXES = ("ات", "ون", "ين", "ها", "يه", "ه", "ي")
EN_STOP = set("""a an and are as at be by can do does for from has have how if in is it its may of on or our
shall should that the their there these this to us we what when which who will with within""".split())


def stem_en(word):
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def stem_ar(word):
    """Light stemming: drop one common prefix and one common suffix, keeping at least 3 letters."""
    word = word.translate(AR_LETTERS)
    for prefix in AR_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            word = word[len(prefix):]
            break
    for suffix in AR_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word


def tokenize(text):
    text = AR_MARKS.sub("", unicodedata.normalize("NFKC", text).lower().translate(AR_DIGITS))
    out = []
    for tok in TOKEN.findall(text):
        if tok.isdigit():                       # control numbers like 2-3-3-1 carry no meaning
            continue
        if tok < "؀":                      # Latin script (English words and acronyms)
            if len(tok) > 1 and tok not in EN_STOP:
                out.append(stem_en(tok))
        else:
            tok = stem_ar(tok)
            if len(tok) > 2:
                out.append(tok)
    return out


class BM25:
    """Okapi BM25 over a list of documents (each a list of tokens)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.lengths.mean()
        self.postings = {}                      # token -> {doc number: count}
        for n, doc in enumerate(docs):
            for tok in doc:
                self.postings.setdefault(tok, {})
                self.postings[tok][n] = self.postings[tok].get(n, 0) + 1
        self.size = len(docs)

    def scores(self, query_tokens):
        out = np.zeros(self.size)
        for tok in set(query_tokens):
            posting = self.postings.get(tok)
            if not posting:
                continue
            idf = np.log(1 + (self.size - len(posting) + 0.5) / (len(posting) + 0.5))
            for n, tf in posting.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[n] / self.avg)
                out[n] += idf * tf * (self.k1 + 1) / norm
        return out


# --------------------------------------------------------------------------- the index

class Side:
    """Everything needed to score the controls of one language."""

    def __init__(self, chunks, vectors, owners, texts, whole, lang):
        chunk_ids = [i for i, c in enumerate(chunks) if c["lang"] == lang]
        position = {chunk: p for p, chunk in enumerate(chunk_ids)}
        self.ids = [chunks[i]["id"] for i in chunk_ids]
        self.chunk = {chunks[i]["id"]: chunks[i] for i in chunk_ids}
        units = np.flatnonzero(np.isin(owners, chunk_ids))
        self.vectors = vectors[units]
        self.owner_pos = np.array([position[o] for o in owners[units]])
        self.is_whole = whole[units]
        self.bm25 = BM25([tokenize(texts[u]) for u in units])

    def _best_per_control(self, unit_scores):
        out = np.full(len(self.ids), -np.inf)
        np.maximum.at(out, self.owner_pos, unit_scores)
        return out

    def dense(self, q, whole_only=False):
        scores = self.vectors @ q                       # cosine similarity: vectors are normalised
        if whole_only:
            scores = np.where(self.is_whole, scores, -np.inf)
        return self._best_per_control(scores)

    def keyword(self, query_tokens):
        return self._best_per_control(self.bm25.scores(query_tokens))


class Index:
    def __init__(self):
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(INDEX_DIR / "embeddings.npy")
        owners = np.array(json.loads((INDEX_DIR / "units.json").read_text(encoding="utf-8")))
        _, texts = search_units(self.chunks)
        assert len(owners) == len(vectors) == len(texts), "index is stale: re-run ingest.py"
        whole = np.zeros(len(owners), dtype=bool)
        whole[np.unique(owners, return_index=True)[1]] = True   # the first unit of each control is the whole control
        self.side = {lang: Side(self.chunks, vectors, owners, texts, whole, lang) for lang in ("en", "ar")}


@lru_cache(maxsize=1)
def index():
    return Index()


@lru_cache(maxsize=1)
def embedder():
    return get_model()


_vectors = {}


def embed(text):
    if text not in _vectors:
        _vectors[text] = embedder().encode([text], normalize_embeddings=True)[0]
    return _vectors[text]


# --------------------------------------------------------------------------- reranking

@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder      # imported lazily: only needed for "+rerank" modes
    return CrossEncoder(RERANKER, max_length=512)


_rerank_scores = {}


def rerank(query, lang, ranked):
    """Re-order the first RERANK_TOP controls of `ranked` by cross-encoder score; the rest keep their order."""
    chunk = index().side[lang].chunk
    head, tail = ranked[:RERANK_TOP], ranked[RERANK_TOP:]
    todo = [cid for cid, _ in head if (query, lang, cid) not in _rerank_scores]
    if todo:
        pairs = [(query, chunk[cid]["subdomain"] + "\n" + chunk[cid]["text"]) for cid in todo]
        for cid, score in zip(todo, reranker().predict(pairs, batch_size=8)):
            _rerank_scores[(query, lang, cid)] = float(score)
    head = sorted(head, key=lambda item: -_rerank_scores[(query, lang, item[0])])
    return [(cid, _rerank_scores[(query, lang, cid)]) for cid, _ in head] + tail


# --------------------------------------------------------------------------- ranking

def signal_scores(name, lang, query, query_tokens):
    side = index().side[lang]
    if name == "bm25":
        return side.keyword(query_tokens)
    return side.dense(embed(query), whole_only=(name == "dense_whole"))


def fuse(rankings):
    """Reciprocal rank fusion. Each ranking is [(control id, score), ...] best first; a control scores the sum
    of 1 / (RRF_K + its rank) over the rankings that include it."""
    fused = {}
    for ranking in rankings:
        for r, (cid, _) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r)
    return sorted(fused.items(), key=lambda item: -item[1])


def rank(query, lang, mode="dense"):
    """Controls in `lang` ordered best first, as [(control id, score)]. Score is cosine similarity for a
    single dense mode, BM25 for bm25, and the fused rank score for combined modes."""
    base, _, post = mode.partition("+")
    if post:
        assert post == "rerank", f"unknown mode suffix: {post}"
        return rerank(query, lang, rank(query, lang, base))
    signals = MODES[base]
    query_tokens = tokenize(query)
    rankings = []
    for name, which in signals:
        side = index().side[lang if which == "same" else OTHER[lang]]
        scores = signal_scores(name, lang if which == "same" else OTHER[lang], query, query_tokens)
        order = [i for i in np.argsort(-scores) if scores[i] > 0 or name != "bm25"]   # keyword: matches only
        rankings.append([(side.ids[i], float(scores[i])) for i in order])
    return rankings[0] if len(rankings) == 1 else fuse(rankings)
