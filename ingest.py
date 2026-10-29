"""Parse the ECC PDFs into one chunk per control, embed them with BAAI/bge-m3, save to index/.

One chunk = one main control (e.g. 2-2-3) including its sub-clauses (2-2-3-1 ...).

The two PDFs need different parsers:
- English: PyMuPDF's plain text comes out in reading order (ID line, then its text),
  so a small line-by-line state machine is enough.
- Arabic: the plain text stream is unreliable (RTL table cells come out in mixed order,
  multi-digit numbers are reversed, some ligatures map to junk symbols). So we use the
  table geometry instead: each control is a row with a drawn cell rectangle, and we
  rebuild every line from character x-positions (right-to-left).

Usage:
    python ingest.py               # parse, embed, write index/
    python ingest.py --no-embed    # parse only, write index/chunks.json (fast, for inspection)
"""
import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path

import pymupdf

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
EMBED_MODEL = "BAAI/bge-m3"

# Main domain names, as printed in each PDF.
DOMAINS = {
    "en": {"1": "Cybersecurity Governance", "2": "Cybersecurity Defense",
           "3": "Cybersecurity Resilience", "4": "Third-Party and Cloud Computing Cybersecurity"},
    "ar": {"1": "حوكمة الأمن السيبراني", "2": "تعزيز الأمن السيبراني",
           "3": "صمود الأمن السيبراني", "4": "الأمن السيبراني المتعلق بالأطراف الخارجية والحوسبة السحابية"},
}

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
ARABIC_LETTER = re.compile("[؀-ۿﭐ-﷿ﹰ-﻿]")   # incl. presentation forms


def id_key(control_id):
    """'2-10-3' -> (2, 10, 3), so IDs sort numerically."""
    return tuple(int(p) for p in control_id.split("-"))


def tidy(text):
    """Collapse whitespace and remove spaces before punctuation."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" +([.,:;،؛)])", r"\1", text)
    text = re.sub(r"\( +", "(", text)
    return text.strip()


# --------------------------------------------------------------------------- English

EN_CONTROL = re.compile(r"^(\d+)-(\d+)-(\d+)$")       # "1-1-1" alone on a line
EN_SUBDOMAIN = re.compile(r"^(\d+)[-.](\d+)$")        # "1-1" (one heading is printed "1.7")
EN_SUBCLAUSE = re.compile(r"^(\d+)[.-](\d+)[.-](\d+)[.-](\d+)\b")   # "1.5.3.1 ..." starts a new line
EN_SKIP = {
    "Essential Cybersecurity Controls", "Document classification: Public", "TLP: White",
    "The Essential Cybersecurity Controls (ECC)", "Details of the Essential Cybersecurity Controls (ECC)",
    *DOMAINS["en"].values(),
}


def parse_english(path):
    lines = []
    in_details = False
    for page_no, page in enumerate(pymupdf.open(path), start=1):
        for line in page.get_text().splitlines():
            line = line.strip()
            if line == "Details of the Essential Cybersecurity Controls (ECC)":
                in_details = True
            elif line == "Appendices" and in_details:
                in_details = False
            if not in_details or not line or line in EN_SKIP:
                continue
            if line.isdigit() or ARABIC_LETTER.search(line):   # page numbers, Arabic header marks
                continue
            lines.append((page_no, line))

    chunks, sub, target = [], None, None
    for page_no, line in lines:
        if m := EN_CONTROL.match(line):
            cid = "-".join(m.groups())
            chunks.append({"id": cid, "page": page_no, "subdomain": sub, "lines": []})
            target = chunks[-1]["lines"]
        elif m := EN_SUBDOMAIN.match(line):
            sub = {"id": "-".join(m.groups()), "title": [], "objective": []}
            target = sub["title"]
        elif line.startswith("Objective") and sub is not None and target is sub["title"]:
            target = sub["objective"]
            rest = line[len("Objective"):].strip()
            if rest:
                target.append(rest)
        elif line == "Controls":
            target = None
        elif target is not None:
            target.append(line)

    out = []
    for c in chunks:
        text = ""
        for line in c["lines"]:
            if m := EN_SUBCLAUSE.match(line):   # normalise "1.5.3.1" to the doc's coding scheme "1-5-3-1"
                line = "-".join(m.groups()) + line[m.end():]
                text += "\n" + line
            else:
                text += " " + line
        s = c["subdomain"]
        out.append(make_chunk("en", c["id"], c["page"], s["id"], " ".join(s["title"]),
                              " ".join(s["objective"]), text))
    return out


# --------------------------------------------------------------------------- Arabic

# The Arabic font draws some letter pairs as single ligature glyphs whose Unicode mapping
# is wrong. Each mapping below was read off from context in the PDF, e.g. "أد˺" = أدنى,
# "˽كن" = يمكن, "العاˮ" = العالم, "قاɧة" = قائمة.
AR_GLYPHS = str.maketrans({
    "˻": "ني", "˴": "ما", "˼": "ير", "˾": "ين", "ɬ": "بم", "˺": "نى", "˽": "يم", "ˮ": "لم",
    "ɧ": "ئم", "˹": "نم", "ɵ": "ثر", "ɱ": "تم", "ʗ": "كي", "ɳ": "نه", "ɴ": "تي", "ɪ": "ئي",
    "ـ": "",   # tatweel (decorative stretching)
})
# Runs that read left-to-right inside an Arabic line: Latin phrases and digit sequences.
# Hyphens and slashes between numbers stay right-to-left (that is how IDs like ١-٥-٣ are laid out).
AR_LTR_RUN = re.compile(r"[A-Za-z][A-Za-z0-9 \-/.,:&'’]*[A-Za-z0-9]|[A-Za-z]|[0-9٠-٩]+")
AR_CONTROL = re.compile(r"^[٠-٩]+-[٠-٩]+-[٠-٩]+$")
AR_SUBDOMAIN = re.compile(r"^[٠-٩]+-[٠-٩]+$")


def ar_line(chars):
    """Rebuild one visual line: order chars right-to-left by x, then flip LTR runs back."""
    chars = sorted(chars, key=lambda c: -c["x"])
    s = "".join(c["c"] for c in chars)
    s = unicodedata.normalize("NFKC", s)            # presentation forms (ﺑ) -> base letters (ب)
    s = AR_LTR_RUN.sub(lambda m: m.group(0)[::-1], s)
    s = s.translate(str.maketrans("()", ")("))      # brackets are stored mirrored
    return s.translate(AR_GLYPHS)


def ar_lines(chars, tol=4):
    """Group chars into visual lines by their vertical centre."""
    rows = []
    for c in sorted(chars, key=lambda c: c["y"]):
        if rows and abs(rows[-1][-1]["y"] - c["y"]) <= tol:
            rows[-1].append(c)
        else:
            rows.append([c])
    return [line for line in (ar_line(r).strip() for r in rows) if line]


def ar_page_rows(page):
    """Yield (id_cell_text, [text lines]) for each table row on the page, top to bottom."""
    # Everything the page draws, as rectangles (lines become zero-height rectangles).
    shapes = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "re":
                shapes.append(it[1])
            elif it[0] == "l":
                shapes.append(pymupdf.Rect(it[1], it[2]).normalize())
    body = [r for r in shapes if 50 < r.y0 < 740]          # ignore header/footer decorations
    if not body:
        return
    right = max(r.x1 for r in body)                         # right edge of the table

    # The ID column is the rightmost column (25-70pt wide). Its row edges come from two kinds of
    # drawings, depending on the page: filled cell rectangles, or thin horizontal border lines.
    # (Text lines also get their own small highlight boxes; those are narrower and ignored.)
    cells = [r for r in body if r.x1 >= right - 3 and 25 <= r.width <= 70 and r.height >= 8]
    borders = [r for r in body if r.x1 >= right - 3 and r.x0 <= right - 25 and r.height < 3]
    if not cells and not borders:
        return
    id_col_left = statistics.median(r.x0 for r in cells + borders if r.width <= 70)
    edges = sorted({round(y) for r in cells for y in (r.y0, r.y1)} | {round(r.y0) for r in borders})
    edges = [y for i, y in enumerate(edges) if i == 0 or y - edges[i - 1] > 2]   # merge near-duplicates
    rows = list(zip(edges, edges[1:]))

    chars = []
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    chars.append({"c": ch["c"], "x": (x0 + x1) / 2, "y": (y0 + y1) / 2})

    for y0, y1 in rows:
        inside = [c for c in chars if y0 <= c["y"] < y1]
        id_text = " ".join(ar_lines([c for c in inside if c["x"] >= id_col_left]))
        yield id_text, ar_lines([c for c in inside if c["x"] < id_col_left])


def parse_arabic(path):
    doc = pymupdf.open(path)
    pages = [unicodedata.normalize("NFKC", p.get_text()) for p in doc]
    start = next(i for i, t in enumerate(pages) if "تفاصيل الضوابط الأساسية" in t and "الضوابط" in t
                 and re.search(r"[٠-٩]-[٠-٩]-[٠-٩]", t))
    end = next(i for i in range(start, len(pages)) if any(l.strip() == "ملاحق" for l in pages[i].splitlines()))

    chunks, sub, target = [], None, None
    for page_no in range(start, end):
        for id_text, lines in ar_page_rows(doc[page_no]):
            if AR_CONTROL.match(id_text):
                cid = id_text.translate(ARABIC_DIGITS)
                chunks.append({"id": cid, "page": page_no + 1, "subdomain": sub, "lines": []})
                target = chunks[-1]["lines"]
            elif AR_SUBDOMAIN.match(id_text):
                sub = {"id": id_text.translate(ARABIC_DIGITS), "title": [], "objective": []}
                target = sub["title"]
            elif id_text == "الهدف":
                target = sub["objective"]
            elif "الضوابط" in id_text or "الضوابط" in lines:
                target = None
                continue
            elif id_text:
                raise ValueError(f"page {page_no + 1}: unexpected ID cell {id_text!r}")
            # Empty ID cell = the previous row continues from the previous page.
            if target is not None:
                target.extend(lines)

    out = []
    for c in chunks:
        s = c["subdomain"]
        text = ""
        for line in c["lines"]:
            # a sub-clause ID at the start of a line begins a new line; otherwise it's a wrapped line
            text += ("\n" if re.match(r"^[٠-٩]+-[٠-٩]+-[٠-٩]+-[٠-٩]+", line) else " ") + line
        out.append(make_chunk("ar", c["id"], c["page"], s["id"], " ".join(s["title"]),
                              " ".join(s["objective"]), text))
    return out


# --------------------------------------------------------------------------- shared

def make_chunk(lang, cid, page, sub_id, sub_title, objective, text):
    domain = DOMAINS[lang][cid.split("-")[0]]
    sub_title, text = tidy(sub_title), tidy(text)
    return {
        "id": cid,
        "lang": lang,
        "domain": domain,
        "subdomain_id": sub_id,
        "subdomain": sub_title,
        "objective": tidy(objective),
        "page": page,
        "text": text,
        # What gets embedded: the control ID plus where it sits, then its text.
        "embed_text": f"{cid} | {domain} > {sub_id} {sub_title}\n{text}",
    }


def validate(chunks_by_lang):
    """Fail loudly if parsing went wrong, instead of silently indexing bad chunks."""
    for lang, chunks in chunks_by_lang.items():
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), f"{lang}: duplicate control IDs"
        assert ids == sorted(ids, key=id_key), f"{lang}: control IDs out of order"
        for c in chunks:
            assert c["id"].startswith(c["subdomain_id"] + "-"), f"{lang} {c['id']}: wrong subdomain {c['subdomain_id']}"
            assert len(c["text"]) > 20, f"{lang} {c['id']}: text too short: {c['text']!r}"
            assert c["subdomain"], f"{lang} {c['id']}: missing subdomain title"
    en, ar = ({c["id"] for c in chunks_by_lang[k]} for k in ("en", "ar"))
    assert en == ar, f"control IDs differ between languages: only EN {sorted(en - ar)}, only AR {sorted(ar - en)}"


def get_model():
    from sentence_transformers import SentenceTransformer   # imported lazily: slow to load
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 1024     # our chunks are short; keeps memory low
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true", help="only parse and write chunks.json")
    args = ap.parse_args()

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    ar_pdf = next(p for p in pdfs if "ar" in p.stem.lower().split("_"))
    en_pdf = next(p for p in pdfs if p != ar_pdf)
    by_lang = {"en": parse_english(en_pdf), "ar": parse_arabic(ar_pdf)}
    validate(by_lang)
    chunks = by_lang["en"] + by_lang["ar"]
    for lang, cs in by_lang.items():
        print(f"{lang}: {len(cs)} controls in {len({c['subdomain_id'] for c in cs})} subdomains")

    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.no_embed:
        return

    import numpy as np
    owners, texts = search_units(chunks)
    model = get_model()
    vectors = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    np.save(INDEX_DIR / "embeddings.npy", vectors.astype("float32"))
    (INDEX_DIR / "units.json").write_text(json.dumps(owners), encoding="utf-8")
    print(f"saved {len(chunks)} chunks and {vectors.shape} embeddings ({len(texts)} search units) to {INDEX_DIR}/")


def search_units(chunks):
    """What gets embedded. Each control is embedded whole, and each of its sub-clauses is also
    embedded on its own (with the control's intro sentence for context). A long control like
    2-10-3 has five unrelated sub-clauses, so one vector for the whole thing blurs them;
    separate vectors let a narrow question match the one sub-clause it is about.
    Returns (owners, texts): owners[i] is the index in `chunks` that texts[i] belongs to."""
    owners, texts = [], []
    for i, c in enumerate(chunks):
        owners.append(i)
        texts.append(c["embed_text"])
        intro, *subclauses = c["text"].split("\n")
        header = c["embed_text"].split("\n")[0]       # "2-10-3 | domain > subdomain"
        for line in subclauses:
            owners.append(i)
            texts.append(f"{header}\n{intro}\n{line}")
    return owners, texts


if __name__ == "__main__":
    main()
