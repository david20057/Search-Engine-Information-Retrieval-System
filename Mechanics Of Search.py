from __future__ import annotations

from pathlib import Path
import re
import csv
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, DefaultDict, Iterable

# -----------------------------
# Paths 
# -----------------------------
BOOKS_DIR = Path(r"C:\Users\User\Desktop\search_engine\data\books")
META_CSV = Path(r"C:\Users\User\Desktop\search_engine\data\metadata.csv")
OUT_DIR = Path(r"C:\Users\User\Desktop\search_engine\results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Text cleaning + tokenisation
# -----------------------------
START_RE = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", flags=re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF.*?\*\*\*", flags=re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^a-z0-9äöüß]+", flags=re.IGNORECASE)
WS_RE = re.compile(r"\s+")


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Many Gutenberg files contain a header/footer with license text.
    I remove it so queries like "Gutenberg" don't dominate ranking.
    """
    start = START_RE.search(text)
    end = END_RE.search(text)
    if start and end and start.end() < end.start():
        return text[start.end(): end.start()]
    return text


def preprocess(text: str) -> List[str]:
    """
    Basic preprocessing:
    - lowercase
    - keep letters/numbers (including German umlauts for "Dornröschen")
    - split on whitespace
    """
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    return [t for t in text.split() if t]


def norm_field(s: str | None) -> str:
    """Normalise metadata fields for matching (lowercase + tidy whitespace)."""
    s = (s or "").lower()
    return WS_RE.sub(" ", s).strip()


# -----------------------------
# Preview extraction for TSV
# -----------------------------
BAD_LINE_PREFIXES = (
    "produced by", "e-text prepared by", "etext prepared by",
    "transcribed by", "this ebook", "project gutenberg",
    "copyright", "license", "http", "www.gutenberg.org"
)


def find_preview_and_line(doc_id: str, query: str, max_chars: int = 200) -> Tuple[str, str]:
    """
    Finds a 'nice' line to show in the TSV:
    - choose a line that matches MORE query terms (not just the first match)
    - skip common Gutenberg boilerplate lines
    Returns (preview, start_line). If not found, returns ("","").
    """
    path = BOOKS_DIR / f"{doc_id}.txt"
    if not path.exists():
        return "", ""

    q_terms = set(preprocess(query))
    if not q_terms:
        return "", ""

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = strip_gutenberg_boilerplate(text)
    lines = text.splitlines()

    best_preview = ""
    best_line_no = 0
    best_score = 0.0

    for i, line in enumerate(lines, 1):
        clean = WS_RE.sub(" ", line).strip()
        if not clean:
            continue

        low = clean.lower()
        if low.startswith(BAD_LINE_PREFIXES):
            continue

        hits = sum(1 for t in q_terms if t in low)
        if hits == 0:
            continue

        # Slight preference for informative lines over tiny ones
        length_bonus = min(len(clean), 120) / 120.0
        score = hits + 0.2 * length_bonus

        if score > best_score:
            preview = clean[:max_chars].rstrip()
            if len(clean) > max_chars:
                preview += "..."
            best_preview = preview
            best_line_no = i
            best_score = score

    if best_preview:
        return best_preview, str(best_line_no)

    return "", ""


# -----------------------------
# Index building (inverted index + stats)
# -----------------------------
InvertedIndex = DefaultDict[str, Dict[str, int]]  # term -> {doc_id: tf}


def build_index(limit_docs: int | None = None) -> Tuple[InvertedIndex, Dict[str, int], Dict[str, int]]:
    """
    Builds:
    - inverted index: term -> {doc_id: term_frequency}
    - doc_len: doc_id -> number of tokens
    - df: term -> number of docs containing the term
    """
    index: InvertedIndex = defaultdict(lambda: defaultdict(int))
    doc_len: Dict[str, int] = {}
    df: Dict[str, int] = defaultdict(int)

    files = list(BOOKS_DIR.glob("*.txt"))
    if limit_docs is not None:
        files = files[:limit_docs]

    print("Indexing docs:", len(files))

    for i, f in enumerate(files, 1):
        doc_id = f.stem
        text = f.read_text(encoding="utf-8", errors="ignore")
        text = strip_gutenberg_boilerplate(text)
        tokens = preprocess(text)

        doc_len[doc_id] = len(tokens)

        seen_terms = set()
        for t in tokens:
            index[t][doc_id] += 1
            if t not in seen_terms:
                df[t] += 1
                seen_terms.add(t)

        if i % 5000 == 0:
            print("...indexed", i)

    return index, doc_len, df


# -----------------------------
# Retrieval models
# -----------------------------
def structured_search(query: str, top_k: int = 100) -> List[Tuple[str, float]]:
    """
    Structured retrieval = search in metadata (title/author/bookshelf/language).
    Simple weighted matching for tokens.
    """
    q_tokens = preprocess(query)
    scores: Dict[str, float] = defaultdict(float)

    with open(META_CSV, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            book_id = str(row["gutenberg_id"])
            title = norm_field(row.get("title"))
            author = norm_field(row.get("author"))
            shelf = norm_field(row.get("gutenberg_bookshelf"))
            lang = norm_field(row.get("language"))

            s = 0.0
            for t in q_tokens:
                if t in title:
                    s += 3.0
                if t in author:
                    s += 2.0
                if t in shelf:
                    s += 1.0
                if t == lang:
                    s += 0.5

            if s > 0:
                scores[book_id] = s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def tfidf_search(query: str, index: InvertedIndex, df: Dict[str, int], N: int, top_k: int = 100) -> List[Tuple[str, float]]:
    """
    Vector Space Model using TF-IDF and cosine similarity.
    IDF is smoothed: log((N+1)/(df+1)) + 1
    """
    q_tokens = preprocess(query)
    if not q_tokens:
        return []

    q_tf = Counter(q_tokens)

    q_vec: Dict[str, float] = {}
    for term, tf in q_tf.items():
        if term in df:
            idf = math.log((N + 1) / (df[term] + 1)) + 1.0
            q_vec[term] = tf * idf

    if not q_vec:
        return []

    q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
    scores: Dict[str, float] = defaultdict(float)
    doc_norm_sq: Dict[str, float] = defaultdict(float)

    for term, q_w in q_vec.items():
        idf = math.log((N + 1) / (df[term] + 1)) + 1.0
        for doc_id, tf in index.get(term, {}).items():
            d_w = tf * idf
            scores[doc_id] += q_w * d_w
            doc_norm_sq[doc_id] += d_w * d_w

    ranked = []
    for doc_id, dot in scores.items():
        d_norm = math.sqrt(doc_norm_sq[doc_id])
        if d_norm > 0:
            ranked.append((doc_id, dot / (q_norm * d_norm)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def bm25_search(
    query: str,
    index: InvertedIndex,
    df: Dict[str, int],
    doc_len: Dict[str, int],
    N: int,
    avgdl: float,
    top_k: int = 100,
    k1: float = 1.5,
    b: float = 0.75
) -> List[Tuple[str, float]]:
    """
    Okapi BM25.
    """
    q_tokens = preprocess(query)
    if not q_tokens:
        return []

    q_tf = Counter(q_tokens)
    scores: Dict[str, float] = defaultdict(float)

    for term in q_tf.keys():
        if term not in df:
            continue

        idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
        postings = index[term]

        for doc_id, tf in postings.items():
            dl = doc_len[doc_id]
            denom = tf + k1 * (1 - b + b * (dl / avgdl))
            scores[doc_id] += idf * (tf * (k1 + 1) / denom)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# -----------------------------
# TSV writing
# -----------------------------
def write_tsv(query_nr: int, model_name: str, query_text: str, results: List[Tuple[str, float]]) -> None:
    out_path = OUT_DIR / f"{query_nr}_{model_name}.tsv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("rank\tbook_id\tscore\tpreview\tstart_line\n")
        for rank, (doc_id, score) in enumerate(results, 1):
            preview, line_no = find_preview_and_line(doc_id, query_text)
            f.write(f"{rank}\t{doc_id}\t{score}\t{preview}\t{line_no}\n")
    print("Wrote:", out_path.name)


if __name__ == "__main__":
    # The 6 queries from the assignment sheet
    queries = [
        "to be, or not to be",
        "English Grammar",
        "Philip K Dick",
        "Jabberwocky",
        "Gutenberg",
        "Dornröschen",
    ]

    # None for the full run of the 70,772 files 
    LIMIT_DOCS = None

    index, doc_len, df = build_index(limit_docs=LIMIT_DOCS)
    N = len(doc_len)
    avgdl = sum(doc_len.values()) / N

    for i, q in enumerate(queries, 1):
        write_tsv(i, "structured", q, structured_search(q, top_k=100))
        write_tsv(i, "tfidf", q, tfidf_search(q, index, df, N, top_k=100))
        write_tsv(i, "bm25", q, bm25_search(q, index, df, doc_len, N, avgdl, top_k=100))