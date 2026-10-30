"""Fast tests for the hand-written parts of retrieval. No models, no API:  python test_search.py"""
from rag import detect_language
from search import BM25, fuse, tokenize


def test_tokenize_keeps_acronyms_and_drops_numbers():
    assert tokenize("Protecting against DDoS attacks (2-5-3-9)") == ["protect", "against", "ddos", "attack"]
    assert "siem" in tokenize("تحديد التقنيات اللازمة (SIEM) لجمع سجلات الأحداث")


def test_tokenize_arabic_normalises_forms():
    # the same word with and without the definite article, with a different alef and ta marbuta
    assert tokenize("الأمن السيبراني")[0] == tokenize("امن")[0]
    assert tokenize("الحماية") == tokenize("حماية")


def test_arabic_indic_digits_are_dropped_like_ascii_ones():
    assert tokenize("٢-٣-٣-١ التقييد الحازم") == tokenize("2-3-3-1 التقييد الحازم")
    assert all(not t.isdigit() for t in tokenize("٢-٣-٣-١"))


def test_bm25_prefers_the_rarer_matching_word():
    docs = [tokenize(t) for t in ["backup requirements reviewed", "backup and recovery", "firewall requirements"]]
    scores = BM25(docs).scores(tokenize("recovery"))
    assert scores.argmax() == 1 and scores[0] == 0 and scores[2] == 0


def test_fusion_rewards_agreement():
    a = [("x", 0.9), ("y", 0.8), ("z", 0.1)]
    b = [("y", 5.0), ("x", 4.0), ("w", 1.0)]
    order = [cid for cid, _ in fuse([a, b])]
    assert order[:2] == ["x", "y"] or order[:2] == ["y", "x"]
    assert set(order) == {"x", "y", "z", "w"}           # a control only one signal found still appears


def test_language_is_decided_by_words_not_letters():
    assert detect_language("ما متطلبات الـ Patch Management؟") == "ar"
    assert detect_language("When is multi-factor authentication required?") == "en"
    assert detect_language("هل نحتاج SIEM لتجميع السجلات؟") == "ar"
    assert detect_language("What does the ECC say about MFA?") == "en"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
