"""CDE-Match-inspired keyword candidate generator (Stage 0/1).

A LOCAL, reproducible lexical candidate generator that runs several matching
rules IN PARALLEL and UNIONS their candidates — it never stops early when one
rule fires (unlike a sequential exact->fuzzy->containment cascade). Every
candidate carries rule/field provenance and a per-rule score, so a learned
reranker (HGBC/XGBoost) can later combine the signals.

Rules implemented (Stage 1):
  - exact      : normalized whole-string equality, per field (hash map)
  - charngram  : char_wb 3-5 gram TF-IDF cosine top-K, per field (sklearn)
  - token      : token inverted-index overlap (Jaccard), per field
  - containment / rev_containment : substring direction on already-retrieved
                 candidates (provenance enrichment, no extra index scan)
  - pv         : PV-token overlap (query PV_BLOCK_SDE vs CDE PV_SUMMARY)

Catalog text comes ONLY from public CDE fields in cde_master_enriched.parquet:
  SHORT_NAME, LONG_NAME, DEC_LONG_NAME, DEFINITION, PREFERRED_QUESTION_TEXT, PV_SUMMARY.
No gold / true_cde_id / is_label / outcome columns are read. Vectorizers are fit
on the CDE catalog ONLY, never on queries.

The index is cacheable under artifacts_v3_cdisc/keyword_retriever/catalog_index/.
"""
from __future__ import annotations

import contextlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import regex as re
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from demap_repro.text.normalize import normalize_query_text
from demap_repro.text.pv_parser import parse_pv_block

# Public CDE catalog fields (NO gold, NO outcome columns).
TEXT_FIELDS = ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT", "DEFINITION"]
# Fields used for char-ngram fuzzy retrieval (short/medium text only; DEFINITION
# excluded — long free text makes char-ngram cosine noisy and memory-heavy).
CHARNGRAM_FIELDS = ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT"]
# Fields used for word 1-2 gram TF-IDF retrieval (v2; complements char-ngram).
WORDNGRAM_FIELDS = ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT"]
_PV_SPLIT_RE = re.compile(r"\s*\|?\s*PV_TYPE\s*:", re.IGNORECASE)


def strip_query_pv(text: object) -> str:
    """Drop a trailing ``| PV_TYPE: ...`` permissible-value block from a query
    string so name-matching rules see only the query NAME (PVs are routed to the
    dedicated PV rule). Returns the input unchanged if no PV block is present."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return _PV_SPLIT_RE.split(str(text), maxsplit=1)[0].strip()
# Fields used for exact whole-string match (DEFINITION excluded — never exact).
EXACT_FIELDS = ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT"]
PV_FIELD = "PV_SUMMARY"

CATALOG_TEXT_COLS = TEXT_FIELDS + [PV_FIELD]
# Hard guard: these may never be referenced as catalog inputs.
FORBIDDEN_CATALOG_COLS = {
    "true_cde_id", "is_label", "is_injected_gold", "true_rank", "true_score",
    "gold_cde_id", "gold_rank", "gold_score",
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# ---------------------------------------------------------------------------
# Deterministic ordering contract
# ---------------------------------------------------------------------------
# Retrieval must not depend on the process environment. Within every
# (query, rule, field) group, candidates are ordered by an explicit TOTAL order:
#
#   1. ``rule_score`` DESCENDING  — the primary retrieval score;
#   2. catalog row index ASCENDING — the candidate's position in
#      ``KeywordCatalogIndex.cde_ids``, which is unique, frozen at index-build
#      time (``drop_duplicates("cde_id").reset_index(drop=True)``) and shipped
#      inside the index, so it maps 1:1 to a CDE id.
#
# ``top_k_per_rule`` truncation keeps the first K under that order. Because the
# second key is unique the order is total: no tie can survive to be resolved by
# an unordered Python structure, a hash value, a numpy sort's internal pivoting,
# or the environment.
#
# Before this contract, `_token_overlap` and the PV rule built a ``Counter`` by
# iterating a ``set`` of query tokens; CPython randomizes string hashing per
# process (PEP 456), so the candidate array layout — and therefore which
# exactly-tied candidates survived truncation — varied between processes.
KEYWORD_TIE_BREAK_POLICY_VERSION = "keyword_tie_break_v1_score_desc_then_catalog_row_asc"


def _order_by_score_then_row(rows: np.ndarray, scores: np.ndarray,
                             top_k: Optional[int] = None):
    """Apply the deterministic ordering contract to one candidate group.

    ``rows`` are catalog row indices, ``scores`` their retrieval scores. Returns
    ``(rows, scores)`` ordered by score descending then catalog row ascending,
    truncated to ``top_k`` when given. ``np.lexsort`` is used (last key is
    primary) so the order depends only on the two explicit keys.
    """
    if rows.size == 0:
        return rows, scores
    order = np.lexsort((rows, -scores))
    if top_k is not None and order.size > top_k:
        order = order[:top_k]
    return rows[order], scores[order]


def normalize_text(s: object) -> str:
    """Whole-string normalization for exact match: NFKC + quote/dash fold
    (via normalize_query_text), then lowercase + whitespace collapse."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    txt = normalize_query_text(str(s))
    return re.sub(r"\s+", " ", txt).strip().lower()


def tokenize(s: object) -> List[str]:
    """Lowercased alphanumeric tokens, splitting camelCase and underscores so
    code-like SHORT_NAME / query_field strings tokenize sensibly."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    txt = normalize_query_text(str(s))
    txt = _CAMEL_RE.sub(" ", txt).replace("_", " ").lower()
    return _WORD_RE.findall(txt)


def _pv_tokens(pv_block: object) -> List[str]:
    """Normalized permissible-value tokens from a PV block (query or CDE side)."""
    if pv_block is None or (isinstance(pv_block, float) and np.isnan(pv_block)):
        return []
    try:
        parsed = parse_pv_block(str(pv_block))
    except Exception:
        return []
    vals = getattr(parsed, "pv_values_norm", None) or []
    return [v for v in vals if v]


@dataclass
class KeywordCatalogIndex:
    """Holds per-field exact maps, token inverted indices, char-ngram TF-IDF
    matrices, and a PV inverted index over the public CDE catalog."""

    cde_ids: np.ndarray                                   # (n_cde,) str
    # exact: field -> {norm_text: [row_idx, ...]}
    exact_maps: Dict[str, Dict[str, List[int]]]
    # token inverted index: field -> {token: np.ndarray[row_idx]}
    token_index: Dict[str, Dict[str, np.ndarray]]
    token_counts: Dict[str, np.ndarray]                   # field -> (n_cde,) |token set|
    norm_field_text: Dict[str, List[str]]                 # field -> per-cde normalized string
    charngram_vec: Dict[str, TfidfVectorizer]             # field -> fitted vectorizer
    charngram_mat: Dict[str, sparse.csr_matrix]           # field -> (n_cde, V) L2-normalized
    pv_index: Dict[str, np.ndarray]                       # pv_token -> row_idx array
    pv_counts: np.ndarray                                 # (n_cde,) |pv token set|
    word_vec: Dict[str, TfidfVectorizer] = dc_field(default_factory=dict)   # v2 word 1-2gram
    word_mat: Dict[str, sparse.csr_matrix] = dc_field(default_factory=dict)
    meta: Dict[str, object] = dc_field(default_factory=dict)

    @property
    def n_cde(self) -> int:
        return len(self.cde_ids)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

_MAX_DF_TOKEN = 5000   # drop tokens appearing in > this many CDEs (too generic)


def build_index(
    cde_master_path: Path,
    *,
    charngram_ngram=(3, 5),
    charngram_min_df: int = 2,
    word_ngram: bool = False,
    eligibility: str = "none",
    verbose: bool = True,
) -> KeywordCatalogIndex:
    from demap_repro.data import eligibility as _elig

    if eligibility not in _elig.ELIGIBILITY_MODES:
        raise ValueError(f"unknown eligibility mode {eligibility!r}; "
                         f"expected one of {_elig.ELIGIBILITY_MODES}")
    cols = ["cde_id"] + CATALOG_TEXT_COLS
    elig_report = None
    if eligibility == "none":
        df = pd.read_parquet(cde_master_path, columns=cols)
    else:
        # The keyword candidate UNIVERSE is constrained here, before any index
        # structure exists, so an ineligible (retired/archived or TEST/Training)
        # CDE can never be retrieved, ranked, or refill a truncated top-K.
        df = pd.read_parquet(
            cde_master_path,
            columns=cols + [_elig.DEFAULT_WORKFLOW_STATUS_COL, _elig.DEFAULT_CONTEXT_COL])
        df, elig_report = _elig.filter_eligible(df, mode=eligibility, return_report=True)
        df = df[cols]
        if verbose:
            print(_elig.format_report(elig_report))
    # Leakage guard: never load gold/outcome columns.
    bad = set(df.columns) & FORBIDDEN_CATALOG_COLS
    assert not bad, f"forbidden catalog columns loaded: {sorted(bad)}"
    df = df.drop_duplicates("cde_id").reset_index(drop=True)
    cde_ids = df["cde_id"].astype(str).to_numpy()
    n = len(df)
    if verbose:
        print(f"[index] catalog CDEs: {n}")

    exact_maps: Dict[str, Dict[str, List[int]]] = {}
    token_index: Dict[str, Dict[str, np.ndarray]] = {}
    token_counts: Dict[str, np.ndarray] = {}
    norm_field_text: Dict[str, List[str]] = {}

    for fld in TEXT_FIELDS:
        raw = df[fld].tolist() if fld in df.columns else [""] * n
        norm = [normalize_text(x) for x in raw]
        norm_field_text[fld] = norm
        # exact map
        if fld in EXACT_FIELDS:
            em: Dict[str, List[int]] = defaultdict(list)
            for i, t in enumerate(norm):
                if t:
                    em[t].append(i)
            exact_maps[fld] = dict(em)
        # token inverted index + per-cde token-set size
        postings: Dict[str, List[int]] = defaultdict(list)
        tcounts = np.zeros(n, dtype=np.int32)
        for i, t in enumerate(raw):
            toks = set(tokenize(t))
            tcounts[i] = len(toks)
            for tok in toks:
                postings[tok].append(i)
        # drop overly generic tokens
        token_index[fld] = {
            tok: np.asarray(idx, dtype=np.int32)
            for tok, idx in postings.items() if len(idx) <= _MAX_DF_TOKEN
        }
        token_counts[fld] = tcounts
        if verbose:
            print(f"[index] {fld}: exact_keys="
                  f"{len(exact_maps.get(fld, {})):>6} tokens={len(token_index[fld]):>7}")

    charngram_vec: Dict[str, TfidfVectorizer] = {}
    charngram_mat: Dict[str, sparse.csr_matrix] = {}
    for fld in CHARNGRAM_FIELDS:
        texts = [normalize_text(x) for x in (df[fld].tolist() if fld in df.columns else [""] * n)]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=charngram_ngram,
                              min_df=charngram_min_df, lowercase=False)
        try:
            mat = vec.fit_transform(texts)  # row-L2-normalized (norm='l2' default)
        except ValueError:  # empty vocabulary
            continue
        charngram_vec[fld] = vec
        charngram_mat[fld] = mat.tocsr()
        if verbose:
            print(f"[index] {fld}: charngram vocab={len(vec.vocabulary_):>8} nnz={mat.nnz}")

    # word 1-2 gram TF-IDF (v2; complementary phrase-level retrieval)
    word_vec: Dict[str, TfidfVectorizer] = {}
    word_mat: Dict[str, sparse.csr_matrix] = {}
    if word_ngram:
        for fld in WORDNGRAM_FIELDS:
            texts = [normalize_text(x) for x in (df[fld].tolist() if fld in df.columns else [""] * n)]
            vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                  min_df=2, token_pattern=r"(?u)\b\w+\b", lowercase=False)
            try:
                mat = vec.fit_transform(texts)
            except ValueError:
                continue
            word_vec[fld] = vec
            word_mat[fld] = mat.tocsr()
            if verbose:
                print(f"[index] {fld}: wordngram vocab={len(vec.vocabulary_):>8} nnz={mat.nnz}")

    # PV inverted index from PV_SUMMARY
    pv_postings: Dict[str, List[int]] = defaultdict(list)
    pv_counts = np.zeros(n, dtype=np.int32)
    pv_raw = df[PV_FIELD].tolist() if PV_FIELD in df.columns else [""] * n
    for i, blk in enumerate(pv_raw):
        toks = set(_pv_tokens(blk))
        pv_counts[i] = len(toks)
        for tok in toks:
            pv_postings[tok].append(i)
    pv_index = {tok: np.asarray(idx, dtype=np.int32)
                for tok, idx in pv_postings.items() if len(idx) <= _MAX_DF_TOKEN}
    if verbose:
        print(f"[index] PV tokens={len(pv_index)}  CDEs_with_pv={int((pv_counts>0).sum())}")

    return KeywordCatalogIndex(
        cde_ids=cde_ids, exact_maps=exact_maps, token_index=token_index,
        token_counts=token_counts, norm_field_text=norm_field_text,
        charngram_vec=charngram_vec, charngram_mat=charngram_mat,
        pv_index=pv_index, pv_counts=pv_counts,
        word_vec=word_vec, word_mat=word_mat,
        meta={"cde_master": str(cde_master_path), "n_cde": n,
              "charngram_ngram": list(charngram_ngram), "word_ngram": word_ngram,
              "eligibility": eligibility, "eligibility_report": elig_report},
    )


# ---------------------------------------------------------------------------
# Cache safety: atomic write + advisory lock
# ---------------------------------------------------------------------------
# The keyword-index cache is written by build jobs that frequently run as Slurm
# ARRAY tasks sharing ONE --out-dir (Steps B/D and the HGBC / cross-encoder
# feature builds all call load_or_build_index on the same path). Two tasks each
# doing ``joblib.dump`` to the SAME file concurrently interleave their bytes and
# leave a CORRUPT joblib that still unpickles -- but balloons on load (observed:
# a 329 MB cache expanding past 62 GiB and OOM-killing every reader). To make
# that impossible:
#   * every writer dumps to a UNIQUE temp file in the cache dir and then
#     ``os.replace`` it into place -- an atomic rename within one filesystem, so
#     a reader only ever sees the previous file or a fully-written new one,
#     never a torn/partial/interleaved file;
#   * an advisory ``flock`` serialises builders so only one rebuilds while the
#     others wait and then load the finished cache. The lock is best-effort
#     (some networked filesystems ignore flock); the atomic replace is what
#     GUARANTEES integrity even when the lock is a no-op.

def _atomic_joblib_dump(obj, dest: Path, *, compress: int = 3) -> None:
    """Dump ``obj`` to ``dest`` atomically: write to a unique temp file in the
    same directory, then ``os.replace`` into place. Never exposes a partial or
    interleaved file to concurrent readers/writers."""
    import os
    import tempfile

    import joblib
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=dest.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp, compress=compress)   # compress>0 => single file
        os.replace(tmp, dest)                       # atomic within one filesystem
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


@contextlib.contextmanager
def _index_cache_lock(cache_file: Path, *, timeout: float = 3600.0,
                      poll: float = 2.0, verbose: bool = True):
    """Advisory exclusive lock on a sidecar ``<cache>.lock`` file so only one
    builder runs at a time. Best-effort: if the filesystem does not support
    ``flock`` (or we time out) we proceed anyway -- the atomic write in
    :func:`_atomic_joblib_dump` still guarantees cache integrity, at worst a
    harmless redundant rebuild. Yields True iff the lock was acquired."""
    import fcntl
    import os
    import time

    lock_path = Path(str(cache_file) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "w")
    acquired = False
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.time() - start > timeout:
                    if verbose:
                        print(f"[index] WARNING: timed out after {timeout:.0f}s waiting for "
                              f"cache lock {lock_path}; proceeding (atomic write keeps it safe)")
                    break
                if verbose:
                    print(f"[index] waiting for keyword-index cache lock held by another job: "
                          f"{lock_path}")
                time.sleep(poll)
        yield acquired
    finally:
        try:
            if acquired:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def _assert_index_eligibility(idx: "KeywordCatalogIndex", eligibility: str,
                              cache_file: Path) -> None:
    """Fail fast if a cached index was built under a different eligibility mode.

    A pre-eligibility cache (no ``eligibility`` key in meta) is treated as
    ``'none'`` — it was built from the unfiltered catalog. Loading such a cache
    when the caller asked for a filtered universe would silently reintroduce
    retired/archived CDEs into the candidate pool, so it is a hard error."""
    got = (getattr(idx, "meta", None) or {}).get("eligibility") or "none"
    if got != eligibility:
        raise RuntimeError(
            f"keyword-index cache {cache_file} was built with eligibility={got!r} "
            f"but eligibility={eligibility!r} was requested; delete the cache or "
            f"use a cache_name that encodes the eligibility mode")


def load_or_build_index(
    cde_master_path: Path,
    cache_dir: Path,
    *,
    rebuild: bool = False,
    word_ngram: bool = False,
    eligibility: str = "none",
    cache_name: str = "keyword_index.joblib",
    verbose: bool = True,
) -> KeywordCatalogIndex:
    """Load a cached keyword index, else build and cache it.

    Concurrency-safe (see the cache-safety note above): concurrent callers
    (e.g. Slurm array tasks that share a cache dir) are serialised by an
    advisory ``flock`` and the cache is written atomically via a temp file +
    ``os.replace``, so a reader never sees a partial or interleaved joblib.

    ``eligibility`` constrains the candidate universe at index-build time (see
    ``build_index``); a cached index whose recorded eligibility differs from the
    requested one is rejected rather than silently reused.
    """
    import joblib
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / cache_name
    if cache_file.exists() and not rebuild:
        if verbose:
            print(f"[index] loading cached index: {cache_file}")
        idx = joblib.load(cache_file)
        _assert_index_eligibility(idx, eligibility, cache_file)
        return idx
    # Cache miss (or forced rebuild): serialise builders on a sidecar lock.
    with _index_cache_lock(cache_file, verbose=verbose):
        # Re-check under the lock: a concurrent task may have built it while we
        # waited -- load its (atomically written) result instead of rebuilding.
        if cache_file.exists() and not rebuild:
            if verbose:
                print(f"[index] loading cached index (built by concurrent job): {cache_file}")
            idx = joblib.load(cache_file)
            _assert_index_eligibility(idx, eligibility, cache_file)
            return idx
        idx = build_index(cde_master_path, word_ngram=word_ngram,
                          eligibility=eligibility, verbose=verbose)
        _atomic_joblib_dump(idx, cache_file, compress=3)
        if verbose:
            print(f"[index] cached (atomic write) -> {cache_file}")
    return idx


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

_CHARNGRAM_BLOCK = 256


def _vec_topk(vec, mat, q_norm: List[str], top_k: int):
    """Cosine top-K per query against an L2-normalized TF-IDF matrix (char or word)."""
    if vec is None:
        return [(np.empty(0, int), np.empty(0, float))] * len(q_norm)
    out = []
    for start in range(0, len(q_norm), _CHARNGRAM_BLOCK):
        chunk = q_norm[start:start + _CHARNGRAM_BLOCK]
        Q = vec.transform(chunk)            # (b, V), L2-normalized
        sims = (Q @ mat.T)                  # (b, n_cde) cosine
        sims = sims.toarray()
        for r in range(sims.shape[0]):
            row = sims[r]
            nz = np.flatnonzero(row > 0)
            if nz.size == 0:
                out.append((np.empty(0, int), np.empty(0, float)))
                continue
            # Deterministic ordering contract: score desc, catalog row asc.
            out.append(_order_by_score_then_row(nz, row[nz], top_k))
    return out


def _token_overlap(index_field: Dict[str, np.ndarray], counts: np.ndarray,
                   q_tokens: List[str], top_k: int):
    """Jaccard token overlap retrieval for one query. Returns (idx_arr, score_arr)."""
    if not q_tokens:
        return np.empty(0, int), np.empty(0, float)
    cnt = Counter()
    # ``sorted`` makes the token sweep — and therefore the Counter's key
    # insertion order — depend only on the query text, never on set iteration
    # order (which CPython randomizes per process).
    qtokens_unique = sorted(set(q_tokens))
    for tok in qtokens_unique:
        post = index_field.get(tok)
        if post is not None:
            cnt.update(post.tolist())
    if not cnt:
        return np.empty(0, int), np.empty(0, float)
    cand = np.fromiter(cnt.keys(), dtype=np.int64)
    overlap = np.fromiter(cnt.values(), dtype=np.float64)
    qn = len(qtokens_unique)
    denom = qn + counts[cand] - overlap
    jac = np.where(denom > 0, overlap / denom, 0.0)
    # Deterministic ordering contract: score desc, catalog row asc (total order,
    # so truncation at top_k keeps a well-defined subset of any tied group).
    return _order_by_score_then_row(cand, jac, top_k)


def _containment_rows(idx: KeywordCatalogIndex, q_norm: str, cde_row: int):
    """Substring containment provenance for one (query, cde) over text fields.
    Yields (rule, field, score)."""
    if not q_norm:
        return
    for fld in TEXT_FIELDS:
        ft = idx.norm_field_text[fld][cde_row]
        if not ft:
            continue
        if q_norm == ft:
            continue  # exact already captured
        if q_norm in ft:
            # query contained in CDE field (reverse containment)
            yield ("rev_containment", fld, len(q_norm) / max(1, len(ft)))
        elif ft in q_norm:
            yield ("containment", fld, len(ft) / max(1, len(q_norm)))


def generate_candidates(
    queries: pd.DataFrame,
    idx: KeywordCatalogIndex,
    *,
    top_k_per_rule: int = 50,
    query_text_col: str = "query_text_q3",
    query_field_col: str = "query_field",
    pv_col: str = "PV_BLOCK_SDE",
    strip_query_pv_block: bool = False,
    use_word_ngram: bool = False,
    include_containment: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all rules in parallel; return long provenance rows:
    [query_id, cde_id, rule, field, rule_rank, rule_score].

    ``strip_query_pv_block`` (v2): drop the trailing ``| PV_TYPE: ...`` block so
    name rules see only the query name. ``use_word_ngram`` (v2): add a word 1-2
    gram TF-IDF retrieval rule (requires an index built with word_ngram=True)."""
    q = queries.copy()
    q["query_id"] = q["query_id"].astype(str)
    qids = q["query_id"].tolist()
    raw_texts = q[query_text_col].fillna("").astype(str).tolist()
    # name text used by name rules (optionally PV-block-stripped)
    q_texts = [strip_query_pv(t) for t in raw_texts] if strip_query_pv_block else raw_texts
    q_norm = [normalize_text(t) for t in q_texts]
    q_tokens = [tokenize(t) for t in q_texts]
    has_field = query_field_col in q.columns
    q_field = (q[query_field_col].fillna("").astype(str).tolist() if has_field
               else [""] * len(q))
    has_pv = pv_col in q.columns
    q_pv = (q[pv_col].tolist() if has_pv else [None] * len(q))

    cde_ids = idx.cde_ids
    rows: List[tuple] = []  # (qi, cde_row, rule, field, score)

    # --- exact (whole-string), per field; also code-like query_field vs SHORT_NAME
    for fi in range(len(qids)):
        seen_exact = set()
        for fld in EXACT_FIELDS:
            hit = idx.exact_maps.get(fld, {}).get(q_norm[fi])
            if hit:
                for cr in hit:
                    rows.append((fi, cr, "exact", fld, 1.0))
                    seen_exact.add(cr)
        if has_field and q_field[fi]:
            fnorm = normalize_text(q_field[fi])
            hit = idx.exact_maps.get("SHORT_NAME", {}).get(fnorm)
            if hit:
                for cr in hit:
                    rows.append((fi, cr, "exact", "SHORT_NAME@query_field", 1.0))

    # --- char-ngram fuzzy, per field (batched)
    for fld in CHARNGRAM_FIELDS:
        per_q = _vec_topk(idx.charngram_vec.get(fld), idx.charngram_mat.get(fld),
                          q_norm, top_k_per_rule)
        for fi, (cand, sc) in enumerate(per_q):
            for cr, s in zip(cand.tolist(), sc.tolist()):
                rows.append((fi, cr, "charngram", fld, float(s)))

    # --- word 1-2 gram fuzzy, per field (v2; batched)
    if use_word_ngram and idx.word_mat:
        for fld in WORDNGRAM_FIELDS:
            if fld not in idx.word_mat:
                continue
            per_q = _vec_topk(idx.word_vec.get(fld), idx.word_mat.get(fld),
                              q_norm, top_k_per_rule)
            for fi, (cand, sc) in enumerate(per_q):
                for cr, s in zip(cand.tolist(), sc.tolist()):
                    rows.append((fi, cr, "wordngram", fld, float(s)))

    # --- token-overlap, per field
    for fld in TEXT_FIELDS:
        tindex = idx.token_index[fld]
        tcounts = idx.token_counts[fld]
        for fi in range(len(qids)):
            cand, sc = _token_overlap(tindex, tcounts, q_tokens[fi], top_k_per_rule)
            for cr, s in zip(cand.tolist(), sc.tolist()):
                rows.append((fi, cr, "token", fld, float(s)))

    # --- PV-token overlap (queries with a PV block)
    if has_pv:
        for fi in range(len(qids)):
            # sorted(): PV-token sweep order depends only on the query's PV block
            qpv = sorted(set(_pv_tokens(q_pv[fi])))
            if not qpv:
                continue
            cnt = Counter()
            for tok in qpv:
                post = idx.pv_index.get(tok)
                if post is not None:
                    cnt.update(post.tolist())
            if not cnt:
                continue
            cand = np.fromiter(cnt.keys(), dtype=np.int64)
            overlap = np.fromiter(cnt.values(), dtype=np.float64)
            denom = len(qpv) + idx.pv_counts[cand] - overlap
            jac = np.where(denom > 0, overlap / denom, 0.0)
            # Deterministic ordering contract: score desc, catalog row asc.
            cand, jac = _order_by_score_then_row(cand, jac, top_k_per_rule)
            for cr, s in zip(cand.tolist(), jac.tolist()):
                rows.append((fi, int(cr), "pv", "PV_SUMMARY", float(s)))

    # --- containment provenance on already-retrieved (query, cde) pairs
    # (only annotates candidates already retrieved by other rules -> does NOT
    # change the candidate set/ceiling; skippable for speed at large top_k).
    retrieved: Dict[int, set] = defaultdict(set)
    if include_containment:
        for fi, cr, *_ in rows:
            retrieved[fi].add(cr)
    for fi in sorted(retrieved):
        qn = q_norm[fi]
        if not qn:
            continue
        # sorted(): containment rows are emitted in ascending catalog row order,
        # not in set-iteration order.
        for cr in sorted(retrieved[fi]):
            for rule, fld, sc in _containment_rows(idx, qn, cr):
                rows.append((fi, cr, rule, fld, sc))

    if not rows:
        return pd.DataFrame(columns=["query_id", "cde_id", "rule", "field",
                                     "rule_rank", "rule_score"])

    arr = pd.DataFrame(rows, columns=["_qi", "_cr", "rule", "field", "rule_score"])
    arr["query_id"] = np.asarray(qids, dtype=object)[arr["_qi"].to_numpy()]
    arr["cde_id"] = cde_ids[arr["_cr"].to_numpy()]
    # Deterministic ordering contract, applied once to the whole provenance frame:
    # within each (query, rule, field), score DESC then catalog row ASC. The
    # explicit keys make both the emitted row order and rule_rank independent of
    # how the rows happened to be appended above.
    arr = arr.sort_values(["_qi", "rule", "field", "rule_score", "_cr"],
                          ascending=[True, True, True, False, True],
                          kind="stable").reset_index(drop=True)
    arr["rule_rank"] = arr.groupby(["_qi", "rule", "field"], sort=False).cumcount() + 1
    out = arr[["query_id", "cde_id", "rule", "field", "rule_rank", "rule_score"]].reset_index(drop=True)
    if verbose:
        print(f"[gen] queries={len(qids)} provenance_rows={len(out)} "
              f"unique_pairs={out[['query_id','cde_id']].drop_duplicates().shape[0]}")
    return out
