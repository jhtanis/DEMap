#!/usr/bin/env python3
"""V1 / inferred-helper Python clone of NCI **CDE Match**.

A faithful-as-possible, *readable* (not optimized) sequential reimplementation of
the caDSR "CDE Match" PL/SQL procedures, reconstructed from:

  scripts/cde_match/cdeMatch_matchFlow.sql           (controller: spDSMatchFlow)
  scripts/cde_match/cdeMatch_dynmcSqlQueryBuilder.sql (rules 1-15: spDSSubDynmc)
  scripts/cde_match/cdeMatch_longestWordLogic.sql     (rules 16-18 longest word)
  scripts/cde_match/cdeMatch_pvvmLogic.sql            (PV/VM qualify: spDSSubPVVM)
  scripts/cde_match/CDE_Match_Logic_04-13-2026.pdf    (flow diagram; confirms flow)

The PDF could not be text-extracted locally (subsetted font, no ToUnicode map),
but the diagram confirms the high-level flow that the SQL encodes:
  exact PT/QT/Alt -> like PT/QT/Alt -> reverse-like PT/QT/Alt
  -> (concept/synonym, currently gated off) -> longest-word fallback,
  with PV/VM qualification for enumerated sources/CDEs, unique-CDE accumulation,
  and a hard stop once the match limit is reached.

EVERYTHING that could not be recovered from the available files is implemented in
the clearly-marked "INFERRED HELPERS / CONSTANTS" section below and documented in
artifacts_v3_cdisc/cde_match_clone/CDE_MATCH_CLONE_ASSUMPTIONS.md. All inferred
numeric constants are configurable on the public API (match_limit, min_like_len,
pv_match_pct) and are never silently hard-coded into the matching logic.

This V1 deliberately does NOT improve the algorithm, run parallel-rule ablations,
add SapBERT/PV-semantic features, or do any HGBC hybrid work.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from demap_repro.data import eligibility as elig

# ===========================================================================
# INFERRED HELPERS / CONSTANTS  (not recoverable from the available SQL/PDF)
# ---------------------------------------------------------------------------
# Each item here stands in for a CDE Match helper / bind variable whose body was
# not in the provided files. Behavior is inferred and documented; see
# CDE_MATCH_CLONE_ASSUMPTIONS.md. Tunable constants are surfaced as parameters
# on CdeMatchClone, not frozen here.
# ===========================================================================

DEFAULT_MATCH_LIMIT = 10      # v_mtch_lmt   (our production CDE Match top-10)
DEFAULT_MIN_LIKE_LEN = 3      # v_mtch_min_len  INFERRED: 'like'/'reverse' rules
#   require length(cde_term) > min_like_len. The real value is unknown; 3 keeps
#   1-2 char fragments (e.g. "ID", "OS") from producing junk substring hits while
#   not being so large it suppresses real short clinical terms. Configurable.
DEFAULT_PV_MATCH_PCT = 0.5    # v_mtch_pcnt  INFERRED: fractional PV-overlap
#   threshold used by the 'percent match' PV/VM tier. Configurable.

# v_reg_str / v_reg_str_ds : the two regex cleansing patterns applied to the
# source term (one for exact, one for the like staging). Their exact bodies are
# unrecoverable. INFERENCE: both upper-case and strip punctuation to spaces; V1
# uses ONE normalizer for both the source side and the CDE side so that the
# DB-side pre-normalized columns (MTCH_TERM / MTCH_TERM_ADV) are reproduced
# consistently. Documented as a faithfulness risk.
_NONALNUM_RE = re.compile(r"[^A-Z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize_term(s: object) -> str:
    """V1/FP1 normalization (the original INFERRED normalizer).

    upper-case, replace any run of non-alphanumeric characters with a single
    space, collapse whitespace, trim. Used by ``--normalization v1`` for all
    fields. (Fidelity Pass 2 below replaces this with the exact per-field SQL
    regexes; this is kept for A/B comparison.)
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = _NONALNUM_RE.sub(" ", str(s).upper())
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# Fidelity Pass 2: EXACT SQL normalization (CONFIRMED from S74_NCI_DS.sql +
# cdeMatch_tableViewDefinitions.sql). Each regex DELETES the matched chars
# (no space substitution, no whitespace collapse) — exactly as the DB triggers /
# bind variables do.
#   v_reg_str_ds  = delete  ( ) ; - _ | : $ [ ] ' " % * & # @ { }   (keep spaces,
#                   commas, periods, '?', '/', '+' ...). Used for the SOURCE term
#                   (v_entty_nm), REF MTCH_TERM, and (inferred) long-name MTCH_TERM.
#   v_reg_str     = [^ A-Za-z0-9]  (delete all non-alnum except space). ALT_NMS MTCH_TERM.
#   v_reg_str_adv = [^A-Za-z0-9]   (delete all non-alnum incl. space). MTCH_TERM_ADV
#                   / longest-word rules 16-18.
# NOTE: '+' is NOT in the v_reg_str_ds set, so it is kept (verified — no plus-sign
# handling in the SQL constant).
# ---------------------------------------------------------------------------
_DS_CHARS = "();-_|:$[]'\"%*&#@{}"          # the v_reg_str_ds deletion set (19 chars)
_RE_DS = re.compile("[" + re.escape(_DS_CHARS) + "]")
_RE_REG = re.compile(r"[^ A-Za-z0-9]")
_RE_ADV = re.compile(r"[^A-Za-z0-9]")


def _to_str(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s)


def norm_ds(s: object) -> str:
    """v_reg_str_ds: upper, delete the 19-char set; keep spaces/commas/periods/?/+."""
    return _RE_DS.sub("", _to_str(s).upper())


def norm_reg(s: object) -> str:
    """v_reg_str: upper, delete all non-alphanumeric except space (ALT_NMS MTCH_TERM)."""
    return _RE_REG.sub("", _to_str(s).upper())


def norm_adv(s: object) -> str:
    """v_reg_str_adv: upper, delete all non-alphanumeric incl. space (MTCH_TERM_ADV)."""
    return _RE_ADV.sub("", _to_str(s).upper())


# Per-mode, per-role normalizer selection. 'P'/'Q'/'A' = field normalizers (CDE
# side); 'adv' = longest-word field; 'src' = source main term (v_entty_nm);
# 'word' = source basis for the longest-word token split (v_entty_nm_like).
NORMALIZERS = {
    "v1": {"P": normalize_term, "Q": normalize_term, "A": normalize_term,
           "adv": normalize_term, "src": normalize_term, "word": normalize_term},
    "exact_sql": {"P": norm_ds, "Q": norm_ds, "A": norm_reg,
                  "adv": norm_adv, "src": norm_ds, "word": norm_reg},
}


# Registration-status ordering. The DB sorts candidates within a rule by
# de.regstr_stus_id ASC and subtracts regstr_stus_id*0.05 from the score
# (cdeMatch_dynmcSqlQueryBuilder.sql: score = v_score - de.regstr_stus_id*0.05;
# order by de.regstr_stus_id asc) -- lower regstr_stus_id = more registered =
# ranks first. The numeric ids are not in any provided file, but they were
# RECOVERED EMPIRICALLY (2026-06-05) from the fractional parts of the official CDE
# Match scores: score - floor(score) = 1 - (regstr_stus_id*0.05 mod 1), so a
# fractional of .85 -> id 3, .90 -> id 2, .55 -> id 9, etc. Joining 112k official
# candidate rows to cde_master_enriched.registration_status and taking the mode of
# the inferred id per status name gave the mapping below at 99-100% purity. This
# replaces the earlier inferred 0-based rank so the clone's scores match the real
# official scores. (Only the score VALUE/intra-rule order changes; the cde_id
# tie-break for EXACT-score ties is Oracle physical/group-by order, which is not
# reproducible from any available column -- see CDE_MATCH_CLONE_ASSUMPTIONS.md.)
REGISTRATION_STATUS_RANK: Dict[str, int] = {
    "PREFERRED STANDARD": 1,   # purity 100% (n=131)
    "STANDARD": 2,             # purity 100% (n=28,986)
    "QUALIFIED": 3,            # purity 99%  (n=37,601)
    "CANDIDATE": 5,            # purity 100% (n=54)
    "APPLICATION": 9,          # purity 99%  (n=27,054)
}
# Default for statuses not observed in official output (Historical ~18% of the
# catalog, Superceded, Incomplete, etc.): a neutral mid id. Historical CDEs barely
# appear in match results, so this value does not affect any official-observed
# status ordering; kept at 6 (between Candidate=5 and Application=9).
_DEFAULT_STATUS_RANK = 6


def registration_rank(status: object) -> int:
    if status is None or (isinstance(status, float) and pd.isna(status)):
        return _DEFAULT_STATUS_RANK
    return REGISTRATION_STATUS_RANK.get(str(status).strip().upper(), _DEFAULT_STATUS_RANK)


CURRENT_VERSION_MODES = ("off", "highest")  # plus dynamic 'indicator:<col>'


def _apply_current_version(master: pd.DataFrame, *, mode: str = "off",
                           indicator_col: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """Restrict the master catalog to current-version CDEs (Issue 1).

    ``mode='off'`` (DEFAULT) returns the frame unchanged (legacy: all versions kept).
    ``mode='highest'`` keeps the max ``cde_version`` per ``cde_publicid`` -- a documented
    APPROXIMATION of the production ``currnt_ver_ind = 1`` restriction (no real current-
    version indicator exists in the caDSR XML export). ``mode='indicator:<col>'`` filters
    on a real current-version column (truthy / ==1) IF present; missing column raises.

    Returns ``(filtered_master, report)``. Version comparison is numeric when the
    version parses as a number, else lexicographic (stable, documented).
    """
    n_before = int(len(master))
    n_pub_before = int(master["cde_publicid"].astype(str).nunique())
    report = {"mode": mode, "n_before": n_before, "n_publicid": n_pub_before,
              "n_after": n_before, "n_removed": 0, "approximation": False}
    if mode == "off":
        return master, report

    if mode.startswith("indicator:"):
        col = indicator_col or mode.split(":", 1)[1]
        if col not in master.columns:
            raise KeyError(
                f"current_version_mode={mode!r} needs column {col!r}; present: "
                f"{list(master.columns)}. The caDSR XML export carries no current-version "
                f"indicator -- supply a supplemental column or use mode='highest'.")
        truthy = master[col].map(
            lambda v: str(v).strip().lower() in ("1", "true", "yes", "y", "t"))
        out = master[truthy].copy()
    elif mode == "highest":
        report["approximation"] = True
        ver_num = pd.to_numeric(master["cde_version"], errors="coerce")
        if ver_num.notna().all():
            order_key = ver_num
        else:  # fall back to zero-padded lexicographic on the raw version string
            order_key = master["cde_version"].astype(str).str.zfill(12)
        tmp = master.assign(_ver_key=order_key.values)
        idx = tmp.groupby(master["cde_publicid"].astype(str).values)["_ver_key"].idxmax()
        out = master.loc[sorted(idx)].copy()
    else:
        raise ValueError(
            f"unknown current_version_mode {mode!r}; expected 'off', 'highest', "
            f"or 'indicator:<col>'")

    report["n_after"] = int(len(out))
    report["n_removed"] = n_before - int(len(out))
    return out, report


def parse_pv_set(block: object) -> FrozenSet[str]:
    """Parse a pipe-delimited permissible-value block into a normalized set.

    Source side: split_v3 column PV_BLOCK_SDE (or PV_BLOCK_CDE for a CDE). The DB
    distinguishes PV-name vs value-meaning (VM); V1 unifies them into one set per
    entity (see _build_cde_pv_index). INFERRED: we ignore excl_pv_ind (the
    "excluded PV" flag, nci_ds_dtl.excl_pv_ind) because it is not present in the
    offline split data -- documented.
    """
    if block is None or (isinstance(block, float) and pd.isna(block)):
        return frozenset()
    parts = [normalize_term(p) for p in str(block).split("|")]
    return frozenset(p for p in parts if p)


# ===========================================================================
# RULE INVENTORY  (full 18; rules 10-15 represented but disabled in current flow)
# ===========================================================================

@dataclass(frozen=True)
class Rule:
    rule_id: int
    rule_name: str
    mode: str            # 'E' exact | 'L' like | 'R' reverse-like | 'LW' longest-word
    term: str            # 'P' long name | 'Q' question text | 'A' alt name | 'C' concept | 'S' synonym
    mtch_type: str       # exact | like | reverse | longest_word
    matched_on: str      # longname | qtext | altname | concept | synonym
    enabled_in_current_flow: bool
    disabled_reason: str = ""
    rule_desc: str = ""


# rule_desc strings mirror the v_rule_desc text emitted by the SQL.
RULES: List[Rule] = [
    Rule(1, "Long Name Exact Match", "E", "P", "exact", "longname", True,
         rule_desc="1. Long Name Exact Match"),
    Rule(2, "Question Text Exact Match", "E", "Q", "exact", "qtext", True,
         rule_desc="2. Question Text Exact Match"),
    Rule(3, "Alternate Name Exact Match", "E", "A", "exact", "altname", True,
         rule_desc="3. Alternate Name Exact Match"),
    Rule(4, "Long Name Like Match", "L", "P", "like", "longname", True,
         rule_desc="4. Long Name Like Match"),
    Rule(5, "Question Text Like Match", "L", "Q", "like", "qtext", True,
         rule_desc="5. Question Text Like Match"),
    Rule(6, "Alternate Name Like Match", "L", "A", "like", "altname", True,
         rule_desc="6. Alternate Name Like Match"),
    Rule(7, "Long Name Reverse Like Match", "R", "P", "reverse", "longname", True,
         rule_desc="7. Long Name Reverse Like Match"),
    Rule(8, "Question Text Reverse Like Match", "R", "Q", "reverse", "qtext", True,
         rule_desc="8. Question Text Reverse Like Match"),
    Rule(9, "Alternate Name Reverse Like Match", "R", "A", "reverse", "altname", True,
         rule_desc="9. Alternate Name Reverse Like Match"),
    # ---- Concept / Synonym rules: defined in spDSSubDynmc but the calls are
    #      commented/gated off (v_cncpt_ind) in spDSMatchFlow. Represented here so
    #      the inventory stays complete; NOT executed by the active flow, and not
    #      implementable offline (no DE<->concept / synonym tables available).
    Rule(10, "Concept Exact Match", "E", "C", "exact", "concept", False,
         "commented/gated off in matchFlow", "10. Concept Exact Match"),
    Rule(11, "Synonym Exact Match", "E", "S", "exact", "synonym", False,
         "commented/gated off in matchFlow", "11. Synonym Exact Match"),
    Rule(12, "Concept Like Match", "L", "C", "like", "concept", False,
         "commented/gated off in matchFlow", "12. Concept Like Match"),
    Rule(13, "Synonym Like Match", "L", "S", "like", "synonym", False,
         "commented/gated off in matchFlow", "13. Synonym Like Match"),
    Rule(14, "Concept Reverse Match", "R", "C", "reverse", "concept", False,
         "commented/gated off in matchFlow", "14. Concept Reverse Match"),
    Rule(15, "Synonym Reverse Match", "R", "S", "reverse", "synonym", False,
         "commented/gated off in matchFlow", "15. Synonym Reverse Match"),
    # ---- Longest-word fallback (enumerated source, only when 0 core results). ----
    Rule(16, "Longest Term Like Match", "LW", "P", "longest_word", "longname", True,
         rule_desc="16. Longest Term Like Match"),
    Rule(17, "Longest Term Question Text Like Match", "LW", "Q", "longest_word", "qtext", True,
         rule_desc="17. Longest Term Question Text Like Match"),
    Rule(18, "Longest Term Alt Name Like Match", "LW", "A", "longest_word", "altname", True,
         rule_desc="18. Longest Term Alt Name Like Match"),
]
RULES_BY_ID: Dict[int, Rule] = {r.rule_id: r for r in RULES}

# Active core flow = rules 1-9 in order (concept/synonym 10-15 gated off).
ACTIVE_CORE_RULES: List[Rule] = [RULES_BY_ID[i] for i in range(1, 10)]
# Longest-word rules with their fixed scores (from longestWordLogic.sql).
LONGEST_WORD_RULES = [(RULES_BY_ID[16], 20.0), (RULES_BY_ID[17], 15.0), (RULES_BY_ID[18], 10.0)]

# field key 'P'/'Q'/'A' -> which CdeCatalog field frame to use.
_TERM_TO_FIELD = {"P": "P", "Q": "Q", "A": "A"}


# ===========================================================================
# CDE CATALOG  (builds the CDE-side match indexes from repo parquets)
# ===========================================================================

def _apply_q_holdout(qf: pd.DataFrame, holdout_terms=None, holdout_pairs=None) -> pd.DataFrame:
    """Leakage-aware EVAL helper (Mode A): drop question/reference-text rows that
    exactly reuse a source query text.

    - holdout_terms (set of normalized texts): drop any Q row whose term is in the
      set -> 'any' mode (exclude exact source-text reuse for ANY CDE).
    - holdout_pairs (set of (cde_id, normalized text)): drop only Q rows where the
      (CDE, text) pair is in the set -> 'gold' mode (exclude exact reuse only for
      that query's gold CDE).
    This is an evaluation instrument, NOT an algorithmic change; the default build
    passes neither and behaves exactly as Fidelity Pass 1.
    """
    if qf.empty or (not holdout_terms and not holdout_pairs):
        return qf
    mask = pd.Series(False, index=qf.index)
    if holdout_terms:
        mask = mask | qf["term"].isin(holdout_terms)
    if holdout_pairs:
        mask = mask | pd.Series(
            [(c, t) in holdout_pairs for c, t in zip(qf["cde_id"], qf["term"])],
            index=qf.index)
    return qf[~mask].reset_index(drop=True)


@dataclass
class CdeCatalog:
    """CDE-side data needed to reproduce the rule WHERE-clauses.

    fields['P'|'Q'|'A'] : DataFrame[cde_id, term, term_len, is_enum, regstr_rank]
        one row per CDE for P; MANY rows per CDE for Q (Fidelity Pass 1: all
        question/reference texts) and A (alt names). All frames are restricted to
        current-version catalog CDEs (mirrors the SQL ver_nr join).
    cde_pv : dict cde_id -> frozenset of normalized PV names/value-meanings.
    question_source : 'ref_docs' (Fidelity Pass 1, multi-row question text) or
        'master_pqt' (V1, single PREFERRED_QUESTION_TEXT).
    """
    fields: Dict[str, pd.DataFrame]
    cde_pv: Dict[str, FrozenSet[str]]
    n_cdes: int
    question_source: str = "master_pqt"
    normalization: str = "v1"
    adv_fields: Optional[Dict[str, pd.DataFrame]] = None  # MTCH_TERM_ADV frames (longest-word)
    eligibility: str = "none"                       # production-CDE-Match eligibility mode
    eligibility_report: Optional[dict] = None       # before/after counts (None when not applied)
    # ---- prod-like additions (Issue 1 / Issue 2); defaults reproduce legacy ----
    current_version_mode: str = "off"               # 'off' | 'highest' | 'indicator:<col>'
    current_version_report: Optional[dict] = None   # before/after counts (None when 'off')
    cde_pv_names: Optional[Dict[str, FrozenSet[str]]] = None  # PV names only (perm_val_nm)
    cde_vm: Optional[Dict[str, FrozenSet[str]]] = None        # value-meanings only (item_nm)

    @classmethod
    def build(
        cls,
        cde_master_path: Path,
        alt_names_path: Optional[Path] = None,
        pv_path: Optional[Path] = None,
        ref_docs_path: Optional[Path] = None,
        question_source: str = "ref_docs",
        holdout_q_terms=None,
        holdout_q_pairs=None,
        normalization: str = "exact_sql",
        eligibility: str = "none",
        current_version_mode: str = "off",
        pv_vm_index: str = "unified",
    ) -> "CdeCatalog":
        """Build the CDE-side match indexes.

        Fidelity Pass 1 (question_source='ref_docs'): the question-text field (Q,
        used by rules 2/5/8) is built from ALL question/reference texts in the
        reference-documents table -- Preferred + Alternate + Application-Standard
        Question Text -- mirroring the SQL ``ref r ... obj_key_desc like
        '%QUESTION%'`` join, instead of the single ``PREFERRED_QUESTION_TEXT``.
        Every field frame is inner-joined to the current-version catalog (mirrors
        the SQL ``de.ver_nr = r.ver_nr`` + ``currnt_ver_ind = 1`` join).

        Fidelity Pass 2 (normalization='exact_sql', DEFAULT): use the exact
        per-field SQL regexes -- P/Q via v_reg_str_ds (norm_ds), A via v_reg_str
        (norm_reg) -- and build separate MTCH_TERM_ADV frames (norm_adv) for the
        longest-word rules. This is the default because it matches the confirmed SQL
        regex constants and improves official-overlap. normalization='v1' keeps the
        original single inferred normalizer (for A/B reproducibility).

        Eligibility (eligibility='none', DEFAULT): no eligibility filter -- the catalog
        spans the full master, preserving existing behavior/artifacts.
        eligibility='production_cde_match' applies the getDSFilterString universe
        (drop RETIRED-family workflow_status + TEST/Training context), reproducing the
        SQL admin-status/context exclusion; see demap_repro.data.eligibility. The filter
        is applied to the master BEFORE the field frames are built, so ineligible CDEs
        are removed from P and (via the current-version inner-join) from Q and A.

        Current-version (current_version_mode='off', DEFAULT): keep EVERY (publicid,
        version) row present in the master -- the legacy behavior (byte-for-byte).
        'highest' reduces the catalog to the max version per publicid, a documented
        APPROXIMATION of the production ``currnt_ver_ind = 1`` restriction (the real
        current-version indicator is a ONEDATA DB column absent from the caDSR XML
        export, so the ~110 CDEs whose highest version is not the current one cannot be
        identified from our data -- see the prod-like assumptions doc). 'indicator:<col>'
        filters on a real current-version column (==1/True) IF the master carries one;
        it raises if the column is missing rather than silently guessing.

        PV/VM index (pv_vm_index='unified', DEFAULT): fold PV names and value-meanings
        into one set per CDE (legacy ``cde_pv``). 'separate' ALSO builds ``cde_pv_names``
        (perm_val_nm) and ``cde_vm`` (item_nm) so the prod-like clone can compute PV
        overlap and VM overlap independently (spDSSubPVVM), matching production.
        """
        norm = NORMALIZERS[normalization]
        base_cols = ["cde_id", "cde_publicid", "cde_version", "LONG_NAME",
                     "PREFERRED_QUESTION_TEXT", "VALUE_DOMAIN_TYPE", "registration_status"]
        elig_cols = [elig.DEFAULT_WORKFLOW_STATUS_COL, elig.DEFAULT_CONTEXT_COL]
        cv_col = current_version_mode.split(":", 1)[1] if current_version_mode.startswith("indicator:") else None
        read_cols = base_cols + (elig_cols if eligibility != "none" else [])
        # Only request the current-version indicator column if the parquet actually has
        # it; otherwise let _apply_current_version raise a clear, actionable KeyError
        # instead of a low-level parquet read error.
        if cv_col:
            import pyarrow.parquet as _pq
            avail = set(_pq.read_schema(cde_master_path).names)
            if cv_col in avail and cv_col not in read_cols:
                read_cols = read_cols + [cv_col]
        master = pd.read_parquet(cde_master_path, columns=read_cols).copy()
        master["cde_id"] = master["cde_id"].astype(str)
        master = master.drop_duplicates("cde_id", keep="first")

        # Production-CDE-Match eligibility filter (default OFF). Applied here, before
        # meta/field frames are derived, so ineligible CDEs propagate out everywhere.
        master, eligibility_report = elig.filter_eligible(
            master, mode=eligibility, return_report=True)

        # Current-version restriction (Issue 1; default OFF = keep all versions).
        master, current_version_report = _apply_current_version(
            master, mode=current_version_mode, indicator_col=cv_col)
        master["is_enum"] = master["VALUE_DOMAIN_TYPE"].astype(str).str.upper().eq("ENUMERATED")
        master["regstr_rank"] = master["registration_status"].map(registration_rank).astype(int)
        meta = master.set_index("cde_id")[["is_enum", "regstr_rank"]]

        def _field_frame(text_series: pd.Series, cde_ids: pd.Series, normfn) -> pd.DataFrame:
            term = text_series.map(normfn)
            df = pd.DataFrame({"cde_id": cde_ids.astype(str).values, "term": term.values})
            df = df[df["term"] != ""].copy()
            # current-version restriction (Fidelity Pass 1): inner-join to the
            # master catalog so only current-version CDE rows survive -- drops
            # version-mismatched alt-name / ref-doc rows, matching the SQL join on
            # (item_id, ver_nr) with currnt_ver_ind = 1.
            df = df.merge(meta, left_on="cde_id", right_index=True, how="inner")
            df["term_len"] = df["term"].str.len()
            df = df.drop_duplicates(["cde_id", "term"])   # collapse repeated texts
            return df[["cde_id", "term", "term_len", "is_enum", "regstr_rank"]].reset_index(drop=True)

        # Resolve the raw text for each field once, then normalize per (field, mode).
        long_txt, long_ids = master["LONG_NAME"], master["cde_id"]
        q_used = "master_pqt"
        if question_source == "ref_docs" and ref_docs_path is not None and Path(ref_docs_path).exists():
            ref = pd.read_parquet(
                ref_docs_path,
                columns=["cde_publicid", "cde_version", "document_type", "document_text"],
            ).copy()
            ref = ref[ref["document_type"].astype(str).str.upper().str.contains("QUESTION", na=False)]
            ref["cde_id"] = ref["cde_publicid"].astype(str) + "::" + ref["cde_version"].astype(str)
            q_txt, q_ids = ref["document_text"], ref["cde_id"]
            q_used = "ref_docs"
        else:
            q_txt, q_ids = master["PREFERRED_QUESTION_TEXT"], master["cde_id"]
        if alt_names_path is not None and Path(alt_names_path).exists():
            alt = pd.read_parquet(
                alt_names_path, columns=["cde_publicid", "cde_version", "alternate_name"]
            ).copy()
            alt["cde_id"] = (alt["cde_publicid"].astype(str) + "::" + alt["cde_version"].astype(str))
            a_txt, a_ids = alt["alternate_name"], alt["cde_id"]
        else:
            a_txt, a_ids = None, None

        def _A_frame(normfn):
            if a_txt is None:
                return master.head(0).assign(term="", term_len=0)[
                    ["cde_id", "term", "term_len", "is_enum", "regstr_rank"]]
            return _field_frame(a_txt, a_ids, normfn)

        fields: Dict[str, pd.DataFrame] = {
            "P": _field_frame(long_txt, long_ids, norm["P"]),
            "Q": _field_frame(q_txt, q_ids, norm["Q"]),
            "A": _A_frame(norm["A"]),
        }
        # Leakage-aware eval hook (Mode A): default build passes nothing -> no-op.
        fields["Q"] = _apply_q_holdout(fields["Q"], holdout_q_terms, holdout_q_pairs)

        # MTCH_TERM_ADV frames for the longest-word rules. In v1 the longest-word
        # rules used the standard frames, so alias to `fields` to preserve behavior.
        if normalization == "v1":
            adv_fields = fields
        else:
            adv_fields = {
                "P": _field_frame(long_txt, long_ids, norm["adv"]),
                "Q": _field_frame(q_txt, q_ids, norm["adv"]),
                "A": _A_frame(norm["adv"]),
            }

        cde_pv: Dict[str, FrozenSet[str]] = {}
        cde_pv_names: Optional[Dict[str, FrozenSet[str]]] = None
        cde_vm: Optional[Dict[str, FrozenSet[str]]] = None
        if pv_path is not None and Path(pv_path).exists():
            if pv_vm_index == "separate":
                cde_pv, cde_pv_names, cde_vm = cls._build_cde_pv_index_separate(Path(pv_path))
            else:
                cde_pv = cls._build_cde_pv_index(Path(pv_path))

        return cls(fields=fields, cde_pv=cde_pv, n_cdes=int(master["cde_id"].nunique()),
                   question_source=q_used, normalization=normalization, adv_fields=adv_fields,
                   eligibility=eligibility,
                   eligibility_report=(eligibility_report if eligibility != "none" else None),
                   current_version_mode=current_version_mode,
                   current_version_report=(current_version_report
                                           if current_version_mode != "off" else None),
                   cde_pv_names=cde_pv_names, cde_vm=cde_vm)

    @staticmethod
    def _build_cde_pv_index(pv_path: Path) -> Dict[str, FrozenSet[str]]:
        pv = pd.read_parquet(
            pv_path, columns=["cde_publicid", "cde_version", "valid_value", "value_meaning"]
        ).copy()
        pv["cde_id"] = pv["cde_publicid"].astype(str) + "::" + pv["cde_version"].astype(str)
        # INFERRED: the DB matches source PVs against either CDE PV-name (perm_val_nm)
        # or value-meaning (item_nm) -- spDSSubPVVM runs PV then VM. We fold both
        # into a single normalized set per CDE; membership is equivalent.
        out: Dict[str, FrozenSet[str]] = {}
        vv = pv["valid_value"].map(normalize_term)
        vm = pv["value_meaning"].map(normalize_term)
        tmp = pd.DataFrame({"cde_id": pv["cde_id"].values, "vv": vv.values, "vm": vm.values})
        for cde_id, grp in tmp.groupby("cde_id", sort=False):
            s = set(grp["vv"]) | set(grp["vm"])
            s.discard("")
            out[cde_id] = frozenset(s)
        return out

    @staticmethod
    def _build_cde_pv_index_separate(
        pv_path: Path,
    ) -> Tuple[Dict[str, FrozenSet[str]], Dict[str, FrozenSet[str]], Dict[str, FrozenSet[str]]]:
        """Build PV-name, value-meaning, and unified sets per CDE, SEPARATELY.

        Production ``spDSSubPVVM`` compares the source values against the CDE's PV
        NAMES (``perm_val_nm`` = ``valid_value``) and its VALUE MEANINGS (``item_nm`` =
        ``value_meaning``) as two independent overlap tests, in a cascade. This returns
        ``(unified, pv_names, vm)`` so both the legacy unified overlap (for audit) and
        the prod-like separate overlaps are available. Empty strings are dropped.
        """
        pv = pd.read_parquet(
            pv_path, columns=["cde_publicid", "cde_version", "valid_value", "value_meaning"]
        ).copy()
        pv["cde_id"] = pv["cde_publicid"].astype(str) + "::" + pv["cde_version"].astype(str)
        vv = pv["valid_value"].map(normalize_term)
        vm = pv["value_meaning"].map(normalize_term)
        tmp = pd.DataFrame({"cde_id": pv["cde_id"].values, "vv": vv.values, "vm": vm.values})
        unified: Dict[str, FrozenSet[str]] = {}
        pv_names: Dict[str, FrozenSet[str]] = {}
        vms: Dict[str, FrozenSet[str]] = {}
        for cde_id, grp in tmp.groupby("cde_id", sort=False):
            p = {x for x in grp["vv"] if x}
            m = {x for x in grp["vm"] if x}
            pv_names[cde_id] = frozenset(p)
            vms[cde_id] = frozenset(m)
            unified[cde_id] = frozenset(p | m)
        return unified, pv_names, vms


# ===========================================================================
# CANDIDATE + MATCH ENGINE
# ===========================================================================

_OUTPUT_COLUMNS = [
    "query_id", "split", "cde_id", "cdematch_rank", "cdematch_score",
    "in_cdematch_topk", "rule_id", "rule_desc", "mtch_type", "matched_on",
    "num_pv_match", "enabled_in_current_flow",
]

# Prod-like variant output: legacy columns + audit fields for Issues 1-3
# (current-version eligibility, separate PV/VM overlap, top-N unique score group).
_OUTPUT_COLUMNS_PRODLIKE = [
    "query_id", "split", "cde_id", "cde_publicid", "cde_version",
    "cdematch_rank", "cdematch_score", "unique_score_rank", "in_cdematch_topk",
    "rule_id", "rule_desc", "mtch_type", "matched_on", "passed_current_version",
    "num_pv_match", "pv_overlap", "vm_overlap", "num_pv_match_unified",
    "pvvm_component", "enabled_in_current_flow",
]

# Provenance columns added ONLY when an optional fuzzy fallback is applied. The
# baseline (no fallback) output keeps _OUTPUT_COLUMNS byte-for-byte.
_FUZZY_PROVENANCE_COLUMNS = [
    "cand_source",        # "clone" | "fuzzy_fallback"
    "fuzzy_rule",         # charngram | token (fuzzy rows; secondary on clone-exact rows)
    "fuzzy_field",        # PREFERRED_QUESTION_TEXT | LONG_NAME | DEC_LONG_NAME
    "fuzzy_score",        # fuzzy similarity (charngram cosine / token Jaccard)
    "fuzzy_also_matched", # True on a clone-exact row the fuzzy fallback also surfaced
]
FUZZY_OUTPUT_COLUMNS = _OUTPUT_COLUMNS + _FUZZY_PROVENANCE_COLUMNS

# Tested fuzzy signals that recovered CIMAC v2 non-exact gold-present wins
# (independent keyword retriever, June-18). Each is (keyword_rule, catalog_field).
_FUZZY_ALL = (
    ("charngram", "PREFERRED_QUESTION_TEXT"),
    ("token",     "PREFERRED_QUESTION_TEXT"),
    ("charngram", "LONG_NAME"),
    ("charngram", "DEC_LONG_NAME"),
    ("token",     "DEC_LONG_NAME"),
)
# Named fallback variants (the production variant + rule/field ablations).
FUZZY_FALLBACK_SIGNALS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "charngram_qtext":     (("charngram", "PREFERRED_QUESTION_TEXT"),),
    "token_qtext":         (("token", "PREFERRED_QUESTION_TEXT"),),
    "charngram_names":     (("charngram", "LONG_NAME"), ("charngram", "DEC_LONG_NAME")),
    "token_dec_long_name": (("token", "DEC_LONG_NAME"),),
    "keyword_fuzzy_all":   _FUZZY_ALL,
    "keyword_v1":          _FUZZY_ALL,   # production variant (alias of keyword_fuzzy_all)
}


@dataclass(frozen=True)
class FuzzyFallback:
    """Spec for the optional keyword fuzzy fallback appended after the clone's
    exact-rule candidates. ``signals`` is a tuple of (keyword_rule, catalog_field)
    pairs drawn from a KeywordCatalogIndex; the generation settings mirror the
    June-18 keyword baseline so the fallback reproduces its tested candidates."""

    name: str
    signals: Tuple[Tuple[str, str], ...]
    top_k_per_rule: int = 500
    query_text_col: str = "query_text_raw"
    strip_query_pv: bool = True
    use_word_ngram: bool = True
    fuzzy_top_n: int = 1000          # cap on fuzzy candidates kept per query

    @classmethod
    def from_name(cls, name: str, **overrides) -> "FuzzyFallback":
        if name not in FUZZY_FALLBACK_SIGNALS:
            raise ValueError(
                f"unknown fuzzy fallback variant {name!r}; "
                f"choose from {sorted(FUZZY_FALLBACK_SIGNALS)}")
        return cls(name=name, signals=FUZZY_FALLBACK_SIGNALS[name], **overrides)


# Candidate-side text fields the exact-query-match control may operate on.
EXACT_MATCH_CONTROLLABLE_FIELDS = ("P", "Q", "A")   # P=long name, Q=qtext, A=alt name


@dataclass(frozen=True)
class _QueryBlock:
    """Per-query specification of which exact-matching metadata rows to hide.

    A row in a field frame is blocked iff its normalized ``term`` equals the
    (field-appropriate) normalized source text AND it belongs to a targeted CDE.
    ``any_scope`` ignores CDE identity (diagnostic). ``ver_ids`` matches versioned
    cde_ids; ``pub_ids`` matches any version of those public ids.
    """
    fields: FrozenSet[str]
    any_scope: bool
    ver_ids: FrozenSet[str]
    pub_ids: FrozenSet[str]


@dataclass
class ExactMatchControl:
    """Query-level exact-query-text metadata leakage control (additive; default no-op).

    For caDSR-derived evaluation splits the gold CDE often carries a metadata row
    (long name / reference question text / alternate name) whose normalized text is
    IDENTICAL to the source query text -- a leakage path the clone's EXACT rules
    exploit. This control simulates a realistic non-caDSR exact-match rate by
    KEEPING a calibrated fraction of queries' exact gold-metadata rows and BLOCKING
    the rest, **per query** (the catalog is never mutated; two queries that share a
    metadata row are decided independently).

    Semantics (locked-in decisions, 2026-06-18):
      - allow_rate = fraction of queries KEPT (1.0 = baseline/no-op, 0.0 = block all).
      - Only the exact-matching metadata row(s) of the targeted CDE are hidden, and
        only for that query; the candidate CDE and its non-matching rows remain.
      - scope='gold' targets the query's gold CDE only; scope='any' is diagnostic.
      - fields restricts which of P/Q/A are controlled (PVs are never controlled).
      - level resolves gold matching: 'versioned' (exact cde_id), 'publicid' (any
        version), 'auto' (versioned when the gold carries a version, else publicid).
      - Deterministic: the per-query allow/block decision is a pure hash of
        (seed, query_id) -- no RNG state; reproducible; different seeds may differ.
    """
    allow_rate: float = 1.0
    seed: int = 42
    fields: Tuple[str, ...] = EXACT_MATCH_CONTROLLABLE_FIELDS
    scope: str = "gold"        # 'gold' | 'any'
    level: str = "auto"        # 'auto' | 'versioned' | 'publicid'

    def __post_init__(self):
        bad = [f for f in self.fields if f not in EXACT_MATCH_CONTROLLABLE_FIELDS]
        if bad:
            raise ValueError(f"exact-match fields must be subset of "
                             f"{EXACT_MATCH_CONTROLLABLE_FIELDS}; got {bad}")
        if self.scope not in ("gold", "any"):
            raise ValueError(f"scope must be 'gold' or 'any'; got {self.scope}")
        if self.level not in ("auto", "versioned", "publicid"):
            raise ValueError(f"level must be auto|versioned|publicid; got {self.level}")
        self._fields = frozenset(self.fields)

    def is_active(self) -> bool:
        """True iff the control would change any output (allow_rate < 1.0)."""
        return self.allow_rate < 1.0

    def query_allowed(self, query_id: object) -> bool:
        """Deterministic per-query gate: keep (True) the query's exact gold row(s)?

        Delegates to the SHARED nested mask (``exact_match_mask.query_allowed``),
        which both lexical methods must use: pure function of (seed, query_id),
        allow iff hash01 < allow_rate, clamped at the ends (>=1.0 keeps all,
        <=0.0 blocks all). Byte-identical to the historical in-class formula."""
        from demap_repro.lexical import mask as emm
        return emm.query_allowed(query_id, self.allow_rate, self.seed)

    def build_block(self, golds: Set[str]) -> Optional[_QueryBlock]:
        """Build the row-block spec for a *blocked* query, or None if nothing to do."""
        if self.scope == "any":
            return _QueryBlock(self._fields, True, frozenset(), frozenset())
        golds = {str(g) for g in golds if g is not None and str(g) not in ("", "nan", "None")}
        if not golds:
            return None
        if self.level == "versioned":
            ver = {g for g in golds if "::" in g}
            pub: Set[str] = set()
        elif self.level == "publicid":
            ver = set()
            pub = {g.split("::")[0] for g in golds}
        else:  # auto
            ver = {g for g in golds if "::" in g}
            pub = {g.split("::")[0] for g in golds if "::" not in g}
        if not ver and not pub:
            return None
        return _QueryBlock(self._fields, False, frozenset(ver), frozenset(pub))


@dataclass
class CdeMatchClone:
    catalog: CdeCatalog
    match_limit: int = DEFAULT_MATCH_LIMIT
    min_like_len: int = DEFAULT_MIN_LIKE_LEN
    pv_match_pct: float = DEFAULT_PV_MATCH_PCT
    # ---- prod-like knobs (Issues 2 & 3); defaults reproduce legacy exactly ----
    variant: str = "legacy"              # 'legacy' | 'prod_like'
    pv_vm_mode: str = "unified"          # 'unified' (legacy) | 'separate' (Issue 2)
    limit_mode: str = "match_limit"      # 'match_limit' (legacy) | 'top_n_scores' (Issue 3)
    top_n_scores: int = 10               # number of UNIQUE score groups kept in top_n_scores mode
    top_n_discovery_cap: int = 5000      # safety cap on candidates discovered per query in top_n mode
    #   Raised from 1000 -> 5000 (2026-07-14): the V1 canonical sweep observed up to 4,266
    #   candidates/query within the top-10 score groups on the large lexical splits (test/cctg/
    #   cdash), so a 1000 cap silently truncated top-N score groups there; 5000 was never
    #   reached. If a query DOES hit the cap, run_over_queries logs a WARNING and records it in
    #   ``last_run_stats`` (top-N score groups may be incomplete for those queries).

    def __post_init__(self):
        # Source-side normalizers must match the catalog's mode: 'src' = main term
        # (v_entty_nm); 'word' = longest-word token basis (v_entty_nm_like).
        norm = NORMALIZERS.get(getattr(self.catalog, "normalization", "v1"), NORMALIZERS["v1"])
        self._norm_src = norm["src"]
        self._norm_word = norm["word"]
        self._norm_adv = norm["adv"]      # adv-frame normalizer (longest-word block term)
        # Cap-reached telemetry (top_n_scores mode): per-query flag + per-run summary.
        self._cap_reached_last = False
        self.last_run_stats: dict = {}
        # 'prod_like' is a convenience preset: separate PV/VM overlap unless the caller
        # explicitly overrode pv_vm_mode. limit_mode/top_n stay independently settable.
        if self.variant == "prod_like" and self.pv_vm_mode == "unified":
            self.pv_vm_mode = "separate"
        if self.variant not in ("legacy", "prod_like"):
            raise ValueError(f"unknown variant {self.variant!r}; expected legacy|prod_like")
        if self.pv_vm_mode not in ("unified", "separate"):
            raise ValueError(f"unknown pv_vm_mode {self.pv_vm_mode!r}; expected unified|separate")
        if self.limit_mode not in ("match_limit", "top_n_scores"):
            raise ValueError(f"unknown limit_mode {self.limit_mode!r}; expected "
                             f"match_limit|top_n_scores")

    @property
    def _discovery_cap(self) -> int:
        """How many unique CDEs to accumulate before stopping the flow.

        match_limit mode reproduces legacy behavior (stop at match_limit). top_n_scores
        mode keeps discovering (bounded by ``top_n_discovery_cap``) so the top-N unique
        score groups are complete rather than truncated by the arbitrary match limit.
        """
        return self.match_limit if self.limit_mode == "match_limit" else self.top_n_discovery_cap

    @staticmethod
    def _blocked_indices(frame: pd.DataFrame, src_term: str, block: "_QueryBlock"):
        """Index labels in ``frame`` to hide for a blocked query, or None.

        Targets only rows whose normalized ``term`` exactly equals ``src_term``
        (the exact-match leakage row) and whose CDE is in scope. The ``term``
        equality is the same predicate the EXACT rule uses, so this removes exactly
        the rows that would exact-match -- never non-matching rows of the CDE.
        """
        if frame.empty or not src_term:
            return None
        term_mask = frame["term"].values == src_term
        if not term_mask.any():
            return None
        idx = frame.index.values[term_mask]
        if block.any_scope:
            return idx
        ids = frame["cde_id"].values[term_mask]
        keep = np.fromiter(
            (cid in block.ver_ids or cid.split("::")[0] in block.pub_ids for cid in ids),
            dtype=bool, count=len(ids))
        sel = idx[keep]
        return sel if len(sel) else None

    # ---- inferred helper analogues (clearly named after the SQL helpers) ----
    @staticmethod
    def isDSEnum(pv_attached: object, pv_n: object) -> bool:
        """INFERRED helper isDSEnum: a source is enumerated iff it carries PVs."""
        if isinstance(pv_attached, (bool, np.bool_)):
            if bool(pv_attached):
                return True
        try:
            return int(pv_n) > 0
        except (TypeError, ValueError):
            return False

    def _pv_tier(self, overlap: int, src_n: int) -> Optional[tuple]:
        """spDSSubPVVM qualification cascade collapsed to (score_adj, overlap).

        full overlap -> 0.00 ; >= pct -> -0.01 ; any (>0) -> -0.02 ; else drop.
        (PV-name and value-meaning tiers are unified; see parse_pv_set.)
        """
        if src_n <= 0 or overlap <= 0:
            return None
        if overlap >= src_n:
            return (0.0, overlap)
        if overlap >= src_n * self.pv_match_pct:
            return (-0.01, overlap)
        return (-0.02, overlap)

    def _pv_tier_separate(self, pv_ov: int, vm_ov: int, src_n: int) -> Optional[tuple]:
        """Prod-like spDSSubPVVM cascade with PV and VM overlap computed SEPARATELY.

        Reproduces the SQL fall-through order (cdeMatch_pvvmLogic.sql): full-PV -> full-VM
        -> pct-PV -> pct-VM -> any-PV -> any-VM. The score adjustment depends only on the
        TIER (full=0.00, pct=-0.01, any=-0.02), identical for PV and VM. Unlike the unified
        tier, a CDE reaches 'full'/'pct' only if PV ALONE (or VM ALONE) covers the source
        values -- combining a PV hit and a VM hit on different source values no longer
        counts, matching production.

        Returns (score_adj, winning_overlap, pv_ov, vm_ov, component) or None.
        component in {'PV','VM'}.
        """
        if src_n <= 0:
            return None
        thr = src_n * self.pv_match_pct
        if pv_ov >= src_n:
            return (0.0, pv_ov, pv_ov, vm_ov, "PV")
        if vm_ov >= src_n:
            return (0.0, vm_ov, pv_ov, vm_ov, "VM")
        if pv_ov > 0 and pv_ov >= thr:
            return (-0.01, pv_ov, pv_ov, vm_ov, "PV")
        if vm_ov > 0 and vm_ov >= thr:
            return (-0.01, vm_ov, pv_ov, vm_ov, "VM")
        if pv_ov > 0:
            return (-0.02, pv_ov, pv_ov, vm_ov, "PV")
        if vm_ov > 0:
            return (-0.02, vm_ov, pv_ov, vm_ov, "VM")
        return None

    def _qualify_pv(self, cde_id: str, src_pv: FrozenSet[str], src_n: int) -> Optional[tuple]:
        """Dispatch PV/VM qualification for one CDE -> (score_adj, cand_pv_kwargs) or None.

        In 'unified' mode reproduces the legacy single-set tier. In 'separate' mode uses
        the independent PV/VM cascade and returns the richer overlap audit fields.
        """
        if self.pv_vm_mode == "separate":
            if self.catalog.cde_pv_names is None:
                raise ValueError(
                    "pv_vm_mode='separate' needs a catalog built with pv_vm_index='separate' "
                    "(cde_pv_names/cde_vm are unset) to qualify an enumerated source. Rebuild "
                    "the catalog with pv_vm_index='separate'.")
            pv_set = self.catalog.cde_pv_names.get(cde_id, frozenset())
            vm_set = self.catalog.cde_vm.get(cde_id, frozenset())
            tier = self._pv_tier_separate(len(pv_set & src_pv), len(vm_set & src_pv), src_n)
            if tier is None:
                return None
            adj, win_ov, pv_ov, vm_ov, comp = tier
            unified_ov = len((pv_set | vm_set) & src_pv)
            return adj, {"num_pv": win_ov, "pv_overlap": pv_ov, "vm_overlap": vm_ov,
                         "num_pv_unified": unified_ov, "pvvm_component": comp}
        tier = self._pv_tier(len(self.catalog.cde_pv.get(cde_id, frozenset()) & src_pv), src_n)
        if tier is None:
            return None
        adj, overlap = tier
        return adj, {"num_pv": overlap}

    def _field_match(self, f: pd.DataFrame, src: str, src_sp: str,
                     mode: str, bidirectional: bool) -> pd.DataFrame:
        """Apply one rule's WHERE predicate to a field frame; return matched rows.

        Reproduces spDSSubDynmc match conditions:
          E : src == term
          L : (src in term) OR (term in src)         [bidirectional]
              else (non-enum Q/A): (src + ' ') in term  [forward only]
          R : term in (src + ' ')
        Like/reverse additionally require length(term) > min_like_len.
        """
        if f.empty:
            return f
        terms = f["term"].values
        if mode == "E":
            mask = f["term"].values == src
        elif mode == "L":
            longer = f["term_len"].values > self.min_like_len
            if bidirectional:
                src_in_term = f["term"].str.contains(src, regex=False, na=False).values
                term_in_src = np.fromiter((t in src for t in terms), dtype=bool, count=len(terms))
                mask = (src_in_term | term_in_src) & longer
            else:
                src_sp_in_term = f["term"].str.contains(src_sp, regex=False, na=False).values
                mask = src_sp_in_term & longer
        elif mode == "R":
            longer = f["term_len"].values > self.min_like_len
            term_in_src_sp = np.fromiter((t in src_sp for t in terms), dtype=bool, count=len(terms))
            mask = term_in_src_sp & longer
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"unknown mode {mode}")
        return f[mask]

    def _run_rule(self, rule: Rule, src: str, src_sp: str, enum_pass: bool,
                  src_pv: FrozenSet[str], block_idx: Optional[dict] = None) -> List[dict]:
        """One (rule, enum/non-enum pass) -> ordered candidate dicts (pre-dedup).

        ``block_idx`` (optional) maps field key -> index labels to hide for this
        query (exact-query-match leakage control); applied before any matching so
        the hidden rows are invisible to every rule, never changing other rows.
        """
        f = self.catalog.fields[_TERM_TO_FIELD[rule.term]]
        if f.empty:
            return []
        if block_idx is not None:
            excl = block_idx.get(rule.term)
            if excl is not None:
                f = f.drop(index=excl, errors="ignore")
                if f.empty:
                    return []
        if enum_pass:
            f = f[f["is_enum"].values]          # enumerated CDEs only (val_dom 16,17)
            if f.empty:
                return []
        # rule 4 (Long Name Like) is bidirectional regardless of pass; rules 5/6
        # are bidirectional only on the enum pass (SQL v_enum branch).
        bidirectional = (rule.rule_id == 4) or enum_pass
        sub = self._field_match(f, src, src_sp, rule.mode, bidirectional)
        if sub.empty:
            return []
        g = sub.groupby("cde_id", as_index=False)["regstr_rank"].min()  # one row/CDE
        base = 100.0 - (0.0 if enum_pass else 1.0) - (rule.rule_id - 1) * 5.0
        out: List[dict] = []
        if enum_pass:
            src_n = len(src_pv)
            for cde_id, regstr in zip(g["cde_id"].values, g["regstr_rank"].values):
                q = self._qualify_pv(cde_id, src_pv, src_n)
                if q is None:
                    continue
                pv_adj, pvkw = q
                out.append(self._cand(cde_id, rule, base + pv_adj - regstr * 0.05, **pvkw))
        else:
            for cde_id, regstr in zip(g["cde_id"].values, g["regstr_rank"].values):
                out.append(self._cand(cde_id, rule, base - regstr * 0.05, 0))
        out.sort(key=lambda c: (-c["cdematch_score"], -c["num_pv_match"], c["cde_id"]))
        return out

    def _run_longest_word(self, src_word: str, src_pv: FrozenSet[str],
                          block_idx_adv: Optional[dict] = None) -> List[dict]:
        """Longest-word fallback (rules 16-18): match the longest source token
        against the MTCH_TERM_ADV frames (space-removed), enumerated CDEs, PV/VM
        qualify. ``src_word`` is the source normalized for token splitting
        (v_entty_nm_like). ``block_idx_adv`` hides this query's blocked exact-match
        rows from the adv frames too (consistency with the core rules)."""
        words = src_word.split()
        if not words:
            return []
        sel = max(words, key=len)            # v_sel_word: longest token (first on tie)
        sel_n = len(sel)
        src_n = len(src_pv)
        adv = self.catalog.adv_fields or self.catalog.fields
        out: List[dict] = []
        for rule, base in LONGEST_WORD_RULES:
            f = adv[_TERM_TO_FIELD[rule.term]]
            if f.empty:
                continue
            if block_idx_adv is not None:
                excl = block_idx_adv.get(rule.term)
                if excl is not None:
                    f = f.drop(index=excl, errors="ignore")
                    if f.empty:
                        continue
            f = f[f["is_enum"].values]       # val_dom_typ_id = 17 (enumerated)
            if f.empty:
                continue
            terms = f["term"].values
            longer = f["term_len"].values > self.min_like_len
            len_ge = f["term_len"].values >= sel_n
            sel_in_term = f["term"].str.contains(sel, regex=False, na=False).values
            term_in_sel = np.fromiter((t in sel for t in terms), dtype=bool, count=len(terms))
            sub = f[(sel_in_term | term_in_sel) & longer & len_ge]
            if sub.empty:
                continue
            g = sub.groupby("cde_id", as_index=False)["regstr_rank"].min()
            for cde_id, regstr in zip(g["cde_id"].values, g["regstr_rank"].values):
                q = self._qualify_pv(cde_id, src_pv, src_n)
                if q is None:
                    continue
                pv_adj, pvkw = q
                out.append(self._cand(cde_id, rule, base + pv_adj - regstr * 0.05, **pvkw))
        out.sort(key=lambda c: (-c["cdematch_score"], -c["num_pv_match"], c["cde_id"]))
        return out

    @staticmethod
    def _cand(cde_id: str, rule: Rule, score: float, num_pv: int,
              pv_overlap: object = pd.NA, vm_overlap: object = pd.NA,
              num_pv_unified: object = pd.NA, pvvm_component: object = "none") -> dict:
        return {
            "cde_id": str(cde_id),
            "cdematch_score": round(float(score), 4),
            "rule_id": rule.rule_id,
            "rule_desc": rule.rule_desc,
            "mtch_type": rule.mtch_type,
            "matched_on": rule.matched_on,
            "num_pv_match": int(num_pv),
            "enabled_in_current_flow": rule.enabled_in_current_flow,
            # prod-like audit fields (PV/VM separate); NA in unified/non-enum paths.
            "pv_overlap": pv_overlap,
            "vm_overlap": vm_overlap,
            "num_pv_match_unified": num_pv_unified,
            "pvvm_component": pvvm_component,
        }

    def match_flow(self, name: object, src_pv: FrozenSet[str], is_enum_src: bool,
                   block: Optional["_QueryBlock"] = None) -> List[dict]:
        """spDSMatchFlow for ONE source element. Returns ordered candidate dicts.

        Sequential rules; unique-CDE accumulation (an earlier/higher-priority rule
        wins and a CDE is never re-added). In ``limit_mode='match_limit'`` (legacy)
        there is a hard stop once ``match_limit`` unique CDEs accumulate. In
        ``limit_mode='top_n_scores'`` the flow instead keeps discovering (bounded by
        ``top_n_discovery_cap``) and afterwards keeps only candidates whose score is
        within the top ``top_n_scores`` UNIQUE score values (all ties at the Nth score
        included), assigning a ``unique_score_rank`` per score group.

        ``block`` (optional) is the per-query exact-query-match leakage control spec;
        it hides only this query's exact-matching gold-metadata rows (never the CDE
        or its other rows). The default (None) reproduces baseline behavior exactly.
        """
        self._cap_reached_last = False       # reset per-query (top_n telemetry)
        src = self._norm_src(name)
        if not src:
            return []
        src_sp = src + " "
        src_word = self._norm_word(name)     # basis for longest-word token split
        cap = self._discovery_cap
        # Precompute per-field index labels to hide for this query (once, not per rule).
        block_idx = block_idx_adv = None
        if block is not None:
            adv = self.catalog.adv_fields or self.catalog.fields
            src_adv = self._norm_adv(name)
            block_idx, block_idx_adv = {}, {}
            for fld in block.fields:
                fk = _TERM_TO_FIELD[fld]
                bi = self._blocked_indices(self.catalog.fields[fk], src, block)
                if bi is not None:
                    block_idx[fk] = bi
                bia = self._blocked_indices(adv[fk], src_adv, block)
                if bia is not None:
                    block_idx_adv[fk] = bia
            block_idx = block_idx or None
            block_idx_adv = block_idx_adv or None
        results: List[dict] = []
        seen: set = set()

        def add(cands: List[dict]) -> None:
            for c in cands:
                if len(seen) >= cap:
                    return
                if c["cde_id"] in seen:
                    continue
                seen.add(c["cde_id"])
                results.append(c)

        if is_enum_src:
            # Enumerated source: each rule tries enumerated CDEs (PV/VM-qualified)
            # first, then non-enumerated CDEs; stop at the cap.
            for rule in ACTIVE_CORE_RULES:
                if len(seen) >= cap:
                    break
                add(self._run_rule(rule, src, src_sp, enum_pass=True, src_pv=src_pv,
                                   block_idx=block_idx))
                if len(seen) >= cap:
                    break
                add(self._run_rule(rule, src, src_sp, enum_pass=False, src_pv=src_pv,
                                   block_idx=block_idx))
            if len(seen) == 0:               # extend matching to Longest Word
                add(self._run_longest_word(src_word, src_pv, block_idx_adv=block_idx_adv))
        else:
            # Non-enumerated source: rules 1-9 against all CDEs, no PV step,
            # no longest-word fallback.
            for rule in ACTIVE_CORE_RULES:
                if len(seen) >= cap:
                    break
                add(self._run_rule(rule, src, src_sp, enum_pass=False, src_pv=src_pv,
                                   block_idx=block_idx))

        if self.limit_mode == "top_n_scores":
            # Discovery stopped at the safety cap iff we accumulated >= cap unique CDEs.
            # When that happens the top-N score groups may be INCOMPLETE (a higher-priority
            # rule's later candidates never got added), so flag it for the batch driver.
            self._cap_reached_last = len(seen) >= cap
            results = self._apply_top_n_scores(results)
        for i, c in enumerate(results, start=1):
            c["cdematch_rank"] = i           # rank = rule-priority accumulation order
            c["in_cdematch_topk"] = True
        return results

    def _apply_top_n_scores(self, results: List[dict]) -> List[dict]:
        """Keep candidates within the top ``top_n_scores`` UNIQUE score values (Issue 3).

        The accumulation order (rule priority, then intra-rule score/cde_id order) is the
        clone's deterministic proxy for the arbitrary Oracle physical order among equal
        scores; it is PRESERVED within each score group. We then group by the rounded
        score value, keep the highest ``top_n_scores`` distinct scores, and tag each row
        with ``unique_score_rank`` (1 = highest score group). All ties at the Nth score
        are retained, so cross-analysis against official output is not truncated by the
        arbitrary match limit.
        """
        if not results:
            return results
        # distinct scores, highest first; keep the top-N of them.
        distinct = sorted({c["cdematch_score"] for c in results}, reverse=True)
        keep_scores = distinct[: self.top_n_scores]
        rank_of = {s: i + 1 for i, s in enumerate(keep_scores)}
        keep_set = set(keep_scores)
        kept = [c for c in results if c["cdematch_score"] in keep_set]
        for c in kept:
            c["unique_score_rank"] = rank_of[c["cdematch_score"]]
        return kept

    # ---- batch driver ----
    def run_over_queries(
        self,
        queries: pd.DataFrame,
        split: str,
        query_name_col: str = "query_text_raw",
        pv_block_col: str = "PV_BLOCK_SDE",
        dedup_query_id: bool = True,
        exact_match_control: Optional[ExactMatchControl] = None,
        gold_col: str = "cde_id",
        fuzzy_fallback: Optional["FuzzyFallback"] = None,
        keyword_index: object = None,
    ) -> pd.DataFrame:
        """Run the clone over a split DataFrame; return the long candidate table.

        Fidelity Pass 1: input rows are deduplicated by ``query_id`` before
        matching (a query_id is a hash of the source, so duplicate rows carry the
        same source text/PVs and only differ in gold). This stops the same query
        from emitting its candidate set multiple times; recall (set-based over the
        unique query_ids) is unchanged.

        ``exact_match_control`` (optional): query-level exact-query-match leakage
        control. When None or inactive (allow_rate>=1.0) the output is byte-for-byte
        the baseline. When active, each query is independently allowed/blocked by a
        deterministic hash of (seed, query_id); a blocked query has only its gold
        CDE's exact-matching metadata rows hidden, for that query alone.
        """
        ctrl = exact_match_control if (exact_match_control is not None
                                       and exact_match_control.is_active()) else None
        # gold-by-query (pre-dedup) so multi-gold query_ids are fully covered.
        gold_by_qid: Dict[str, Set[str]] = {}
        if ctrl is not None and ctrl.scope == "gold" and gold_col in queries.columns:
            for qid, g in zip(queries["query_id"].astype(str), queries[gold_col].astype(str)):
                gold_by_qid.setdefault(qid, set()).add(g)
        if dedup_query_id and "query_id" in queries.columns:
            queries = queries.drop_duplicates("query_id")
        rows: List[dict] = []
        name_col = query_name_col if query_name_col in queries.columns else "query_text_q3"
        has_pv_block = pv_block_col in queries.columns
        has_pv_attached = "pv_attached" in queries.columns
        has_pv_n = "PV_N" in queries.columns
        cap_reached_qids: List[str] = []      # top_n telemetry: queries that hit the discovery cap
        for r in queries.itertuples(index=False):
            rd = r._asdict()
            name = _strip_pv_block(rd.get(name_col))
            src_pv = parse_pv_set(rd.get(pv_block_col)) if has_pv_block else frozenset()
            is_enum = self.isDSEnum(
                rd.get("pv_attached") if has_pv_attached else None,
                rd.get("PV_N") if has_pv_n else None,
            )
            qid = str(rd.get("query_id"))
            block = None
            if ctrl is not None and not ctrl.query_allowed(qid):
                golds = gold_by_qid.get(qid, set()) if ctrl.scope == "gold" else set()
                block = ctrl.build_block(golds)
            cands = self.match_flow(name, src_pv, is_enum, block=block)
            if self.limit_mode == "top_n_scores" and self._cap_reached_last:
                cap_reached_qids.append(qid)
            for c in cands:
                c = dict(c)
                c["query_id"] = qid
                c["split"] = split
                rows.append(c)
        # top_n telemetry: record + warn if any query hit the discovery cap (its top-N
        # score groups may be incomplete). No-op / empty for match_limit mode.
        self.last_run_stats = {
            "split": split,
            "limit_mode": self.limit_mode,
            "top_n_discovery_cap": self.top_n_discovery_cap,
            "n_queries_cap_reached": len(cap_reached_qids),
            "cap_reached_query_ids": cap_reached_qids[:100],  # bounded sample for the summary
        }
        if self.limit_mode == "top_n_scores" and cap_reached_qids:
            print(f"WARNING: top_n_discovery_cap={self.top_n_discovery_cap} reached for "
                  f"{len(cap_reached_qids)} query(ies) on split '{split}'; their top-N score "
                  f"groups may be INCOMPLETE. Raise --top-n-discovery-cap to fully cover them.")
        prodlike = (self.variant == "prod_like" or self.pv_vm_mode == "separate"
                    or self.limit_mode == "top_n_scores")
        out_cols = _OUTPUT_COLUMNS_PRODLIKE if prodlike else _OUTPUT_COLUMNS
        if not rows:
            df = pd.DataFrame(columns=out_cols)
        elif prodlike:
            df = pd.DataFrame(rows)
            # derive publicid/version + current-version audit flag; keep NA where the
            # discovery path never set an audit field (e.g. non-enum -> no pv/vm overlap).
            parts = df["cde_id"].astype(str).str.split("::", n=1, expand=True)
            df["cde_publicid"] = parts[0]
            df["cde_version"] = parts[1] if parts.shape[1] > 1 else pd.NA
            cv_on = self.catalog.current_version_mode != "off"
            df["passed_current_version"] = True if cv_on else pd.NA
            if "unique_score_rank" not in df.columns:
                df["unique_score_rank"] = pd.NA
            for c in ("num_pv_match", "cdematch_rank", "unique_score_rank",
                      "pv_overlap", "vm_overlap", "num_pv_match_unified"):
                df[c] = df[c].astype("Int64")
            df = df[out_cols].sort_values(
                ["query_id", "cdematch_rank"]).reset_index(drop=True)
        else:
            df = pd.DataFrame(rows)
            df["num_pv_match"] = df["num_pv_match"].astype("Int64")
            df["cdematch_rank"] = df["cdematch_rank"].astype("Int64")
            df = df[_OUTPUT_COLUMNS].sort_values(
                ["query_id", "cdematch_rank"]).reset_index(drop=True)
        # Optional keyword fuzzy fallback (opt-in; baseline output is unchanged).
        if fuzzy_fallback is not None:
            if keyword_index is None:
                raise ValueError("fuzzy_fallback requires a keyword_index")
            fz = generate_fuzzy_candidates(queries, keyword_index, fuzzy_fallback)
            df = apply_fuzzy_fallback(df, fz, fuzzy_fallback, split=split)
        return df


_PV_SPLIT_RE = re.compile(r"\s\|\s*PV(_TYPE)?\b", re.IGNORECASE)


def _strip_pv_block(name: object) -> str:
    """Defensively drop a trailing ' | PV_TYPE: ...; PV: ...' block if the chosen
    source-name column happens to be the q3 representation rather than the raw
    name. The CDE Match source term is the element name/question only.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    return _PV_SPLIT_RE.split(str(name), maxsplit=1)[0].strip()


def compute_refdoc_holdout(queries: pd.DataFrame, query_name_col: str = "query_text_raw",
                           gold_col: str = "cde_id", normalization: str = "exact_sql"):
    """Build leakage-aware holdout sets from an eval split (Mode A inputs).

    Returns (terms, pairs):
      - terms: set of normalized source query texts (for 'any' holdout).
      - pairs: set of (gold_cde_id, normalized source text) (for 'gold' holdout).
    The normalization MUST match the catalog's Q-field normalizer (the holdout is
    compared against Q terms): v1 -> normalize_term; exact_sql -> norm_ds (the
    v_reg_str_ds the question field uses). Strip a trailing PV block first.
    """
    qnorm = NORMALIZERS.get(normalization, NORMALIZERS["v1"])["Q"]
    col = query_name_col if query_name_col in queries.columns else "query_text_q3"
    normed = queries[col].map(lambda s: qnorm(_strip_pv_block(s)))
    keep = normed != ""
    terms = set(normed[keep])
    pairs = set()
    if gold_col in queries.columns:
        pairs = set(zip(queries.loc[keep, gold_col].astype(str), normed[keep]))
    return terms, pairs


# ===========================================================================
# SUMMARIES  (per-rule counts, query coverage, recall@K)
# ===========================================================================

def per_rule_counts(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["rule_id", "rule_desc", "mtch_type",
                                     "enabled_in_current_flow", "n_candidates"])
    g = (candidates.groupby(["rule_id", "rule_desc", "mtch_type", "enabled_in_current_flow"],
                            dropna=False).size().reset_index(name="n_candidates"))
    return g.sort_values("rule_id").reset_index(drop=True)


def query_coverage(candidates: pd.DataFrame, queries: pd.DataFrame) -> dict:
    n_q = int(queries["query_id"].astype(str).nunique())
    n_cov = int(candidates["query_id"].nunique()) if not candidates.empty else 0
    cands_per_q = (candidates.groupby("query_id").size() if not candidates.empty
                   else pd.Series(dtype=int))
    return {
        "n_queries": n_q,
        "n_queries_with_candidates": n_cov,
        "coverage_frac": (n_cov / n_q) if n_q else 0.0,
        "mean_candidates_per_covered_query": float(cands_per_q.mean()) if len(cands_per_q) else 0.0,
        "max_candidates_per_query": int(cands_per_q.max()) if len(cands_per_q) else 0,
    }


def recall_at_k(candidates: pd.DataFrame, queries: pd.DataFrame,
                ks=(5, 10, 20, 100), gold_col: str = "cde_id") -> dict:
    """Standalone Recall@K of the clone candidates vs gold (versioned + publicid).

    Denominator = queries that have a gold label. A query counts as a hit at K if
    the gold CDE is among the query's top-K candidates by rank.
    """
    if gold_col not in queries.columns:
        return {"note": "no gold column present; recall not computed"}
    gold = queries[["query_id", gold_col]].copy()
    gold["query_id"] = gold["query_id"].astype(str)
    gold["gold_cde"] = gold[gold_col].astype(str)
    gold["gold_pub"] = gold["gold_cde"].str.split("::").str[0]
    denom = int(gold["query_id"].nunique())
    out = {"n_queries_with_gold": denom}
    if candidates.empty or denom == 0:
        for k in ks:
            out[f"recall@{k}"] = 0.0
            out[f"recall_publicid@{k}"] = 0.0
        return out
    cand = candidates.merge(gold[["query_id", "gold_cde", "gold_pub"]], on="query_id", how="inner")
    cand["cand_pub"] = cand["cde_id"].str.split("::").str[0]
    for k in ks:
        topk = cand[cand["cdematch_rank"] <= k]
        hit_v = topk[topk["cde_id"] == topk["gold_cde"]]["query_id"].nunique()
        hit_p = topk[topk["cand_pub"] == topk["gold_pub"]]["query_id"].nunique()
        out[f"recall@{k}"] = hit_v / denom
        out[f"recall_publicid@{k}"] = hit_p / denom
    return out


# ===========================================================================
# OPTIONAL KEYWORD FUZZY FALLBACK  (opt-in; baseline clone output unchanged)
# ===========================================================================

def generate_fuzzy_candidates(queries: pd.DataFrame, keyword_index: object,
                              spec: "FuzzyFallback", *, verbose: bool = False) -> pd.DataFrame:
    """Generate fuzzy candidates for ``spec``'s (rule, field) signals via the
    keyword retriever's tested scorers (charngram cosine / token Jaccard), then
    collapse to the single best signal per (query_id, cde_id).

    Returns ``[query_id, cde_id, fuzzy_rule, fuzzy_field, fuzzy_score]`` (possibly
    empty). No gold/outcome columns are read; the retriever fits only on the CDE
    catalog index, so this is deployment-safe.
    """
    from demap_repro.lexical.cde_match import keyword_retriever as kr  # lazy: heavy sklearn import

    empty = pd.DataFrame(columns=["query_id", "cde_id", "fuzzy_rule",
                                  "fuzzy_field", "fuzzy_score"])
    if keyword_index is None or queries is None or len(queries) == 0:
        return empty
    qcol = (spec.query_text_col if spec.query_text_col in queries.columns
            else "query_text_q3")
    kw = kr.generate_candidates(
        queries, keyword_index,
        top_k_per_rule=spec.top_k_per_rule,
        query_text_col=qcol,
        strip_query_pv_block=spec.strip_query_pv,
        use_word_ngram=spec.use_word_ngram,
        include_containment=False,
        verbose=verbose,
    )
    if kw.empty:
        return empty
    sig = set(spec.signals)
    mask = [(r, f) in sig for r, f in zip(kw["rule"].values, kw["field"].values)]
    kw = kw.loc[mask, ["query_id", "cde_id", "rule", "field", "rule_score"]]
    if kw.empty:
        return empty
    kw = kw.copy()
    kw["query_id"] = kw["query_id"].astype(str)
    kw["cde_id"] = kw["cde_id"].astype(str)
    # best signal per (query, cde): max score; deterministic tie-break (rule, field).
    kw = (kw.sort_values(["rule_score", "rule", "field"], ascending=[False, True, True])
            .drop_duplicates(["query_id", "cde_id"]))
    return kw.rename(columns={"rule": "fuzzy_rule", "field": "fuzzy_field",
                              "rule_score": "fuzzy_score"}).reset_index(drop=True)


def apply_fuzzy_fallback(clone_candidates: pd.DataFrame, fuzzy_candidates: pd.DataFrame,
                         spec: "FuzzyFallback", *, split: object = None) -> pd.DataFrame:
    """Merge clone candidates with fuzzy fallback candidates per query using a
    three-tier ranking and recompute ``cdematch_rank``:

      tier 1  clone EXACT-rule hits (``mtch_type == 'exact'``), pinned in clone order
      tier 2  fuzzy fallback candidates, by ``fuzzy_score`` desc (capped at
              ``spec.fuzzy_top_n``), excluding any cde already in tier 1
      tier 3  clone non-exact candidates (reverse/like/longest_word/pv), in clone
              order, excluding cdes already placed in tier 1 or tier 2

    Dedup is by ``cde_id`` with that tier priority (clone-exact primary). A cde
    that is both a clone-exact hit and a fuzzy hit stays clone-exact primary and
    records the fuzzy provenance as secondary (``fuzzy_also_matched=True``).

    Returns a long table with :data:`FUZZY_OUTPUT_COLUMNS`.
    """
    clone = clone_candidates.copy()
    if len(clone):
        clone["query_id"] = clone["query_id"].astype(str)
        clone["cde_id"] = clone["cde_id"].astype(str)
    fz = fuzzy_candidates
    has_fz = fz is not None and len(fz) > 0
    clone_by_q = {q: g for q, g in clone.groupby("query_id", sort=False)} if len(clone) else {}
    fz_by_q = {q: g for q, g in fz.groupby("query_id", sort=False)} if has_fz else {}

    # clone query order first, then any fuzzy-only queries.
    qids = list(clone_by_q.keys())
    qids += [q for q in fz_by_q if q not in clone_by_q]

    def _clone_row(rec, rank, sval, also=False, fr=pd.NA, ff=pd.NA, fs=pd.NA):
        d = {c: getattr(rec, c) for c in _OUTPUT_COLUMNS}
        d["cdematch_rank"] = rank
        d["cand_source"] = "clone"
        d["fuzzy_rule"] = fr
        d["fuzzy_field"] = ff
        d["fuzzy_score"] = fs
        d["fuzzy_also_matched"] = bool(also)
        return d

    rows: List[dict] = []
    for qid in qids:
        cg = clone_by_q.get(qid)
        if cg is not None and len(cg):
            sval = cg["split"].iloc[0] if "split" in cg.columns else split
            exact = cg[cg["mtch_type"] == "exact"].sort_values("cdematch_rank")
            weak = cg[cg["mtch_type"] != "exact"].sort_values("cdematch_rank")
        else:
            sval, exact, weak = split, None, None
        exact_ids = set(exact["cde_id"]) if exact is not None and len(exact) else set()

        # fuzzy provenance for this query (full, incl. clone-exact ids for secondary).
        fuzzy_full: Dict[str, tuple] = {}
        fuzzy_tier_ids: List[str] = []
        fzq = fz_by_q.get(qid)
        if fzq is not None and len(fzq):
            fzq = fzq.sort_values(["fuzzy_score", "cde_id"], ascending=[False, True])
            for rec in fzq.itertuples(index=False):
                fuzzy_full[rec.cde_id] = (rec.fuzzy_rule, rec.fuzzy_field, float(rec.fuzzy_score))
            fuzzy_tier_ids = [c for c in fzq["cde_id"].tolist()
                              if c not in exact_ids][:spec.fuzzy_top_n]
        fuzzy_tier_set = set(fuzzy_tier_ids)

        rank = 0
        # tier 1: clone exact, pinned in clone order.
        if exact is not None:
            for rec in exact.itertuples(index=False):
                rank += 1
                prov = fuzzy_full.get(rec.cde_id)
                if prov is not None:
                    rows.append(_clone_row(rec, rank, sval, also=True,
                                           fr=prov[0], ff=prov[1], fs=round(prov[2], 4)))
                else:
                    rows.append(_clone_row(rec, rank, sval))
        # tier 2: fuzzy fallback, by score desc.
        for cde in fuzzy_tier_ids:
            fr, ff, fs = fuzzy_full[cde]
            rank += 1
            rows.append({
                "query_id": qid, "split": sval, "cde_id": cde,
                "cdematch_rank": rank, "cdematch_score": round(float(fs), 4),
                "in_cdematch_topk": True, "rule_id": -1,
                "rule_desc": f"fuzzy_fallback:{spec.name}", "mtch_type": fr,
                "matched_on": ff, "num_pv_match": 0, "enabled_in_current_flow": False,
                "cand_source": "fuzzy_fallback", "fuzzy_rule": fr, "fuzzy_field": ff,
                "fuzzy_score": round(float(fs), 4), "fuzzy_also_matched": False,
            })
        # tier 3: clone non-exact, excluding cdes already placed.
        if weak is not None:
            for rec in weak.itertuples(index=False):
                if rec.cde_id in exact_ids or rec.cde_id in fuzzy_tier_set:
                    continue
                rank += 1
                rows.append(_clone_row(rec, rank, sval))

    if not rows:
        return pd.DataFrame(columns=FUZZY_OUTPUT_COLUMNS)
    out = pd.DataFrame(rows)
    out["num_pv_match"] = out["num_pv_match"].astype("Int64")
    out["cdematch_rank"] = out["cdematch_rank"].astype("Int64")
    out["fuzzy_also_matched"] = out["fuzzy_also_matched"].astype(bool)
    return out[FUZZY_OUTPUT_COLUMNS].sort_values(
        ["query_id", "cdematch_rank"]).reset_index(drop=True)
