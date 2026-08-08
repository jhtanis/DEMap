"""Reusable parser/normalizer for permissible-value (PV) text blocks.

Foundation for future PV-synonym-overlap features documented in
``artifacts_v2_balanced/pv_synonym_overlap/PV_SYNONYM_AND_CALIBRATION_DESIGN.md``
(§3). Pure-Python, dependency-light: stdlib only. Designed to be called
in tight loops over candidate-union rows without per-call surprises.

The canonical input format is::

    PV_TYPE: <TYPE>(n=<N>); PV: <pv1> | <pv2> | ... | <pvK>

But the parser tolerates missing pieces, malformed text, NaN, blank
strings, and trailing/leading whitespace. It never raises on input
shape — callers should check the returned ``ParsedPV.is_well_formed``
flag if they care.

Accepted column names (shared across SDE and CDE sides):
``PV_BLOCK_SDE``, ``PV_BLOCK_CDE``, ``PV_SUMMARY``. The parser is
agnostic to which one it sees.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

__all__ = [
    "ParsedPV",
    "PVValueInfo",
    "parse_pv_block",
    "extract_pv_values",
    "normalize_pv_value",
    "dedupe_preserve_order",
    "is_generic_pv",
    "is_code_like_pv",
    "is_numeric_pv",
    "parse_numeric_pv",
    "classify_pv_value",
    "numeric_pv_set",
    "numeric_overlap_features",
    "GENERIC_PV_TOKENS",
]


# Canonical generic / non-informative PV tokens. Compared after
# normalize_pv_value(). Kept deliberately small and conservative —
# clinically-meaningful values must never land here. Extend only after
# auditing real data; the design note (§3.3) names these explicitly.
GENERIC_PV_TOKENS = frozenset(
    {
        "n/a",
        "na",
        "n.a",  # "N.A." after trailing-period stripping
        "not applicable",
        "unknown",
        "unk",
        "not reported",
        "not provided",
        "not specified",
        "unspecified",
        "other",
        "other, specify",
        "none",
        "missing",
    }
)

# Header form: "PV_TYPE: <TYPE>(n=<N>); PV: <body>"
# Spacing is fixed in the producer; we still tolerate a missing "(n=<N>)" and
# a missing ";" so partial / hand-rolled inputs don't crash.
_HEADER_RE = re.compile(
    r"""
    ^\s*
    PV_TYPE\s*:\s*(?P<pv_type>[A-Za-z0-9_\-]*)\s*
    (?:\(\s*n\s*=\s*(?P<pv_n>\d+)\s*\))?
    \s*;?\s*
    (?:PV\s*:\s*(?P<body>.*))?
    \s*$
    """,
    re.VERBOSE | re.DOTALL,
)

# Whitespace collapse helper.
_WS_RE = re.compile(r"\s+")

# Punctuation we strip from the *ends* of normalized values. Internal
# punctuation (commas, parentheticals, colons) is preserved — they often
# disambiguate (e.g. "Yes, and the condition is still present").
_NORM_STRIP_CHARS = " \t\r\n\".,;:'"

# Pure-numeric detection: integer or simple decimal. Used by both
# is_numeric_pv (strict numeric check) and _CODE_PATTERNS (broader
# code-like check). Bare numbers are *numeric and* code-like;
# alphanumeric codes (C12345, MDR0042) are *only* code-like.
#
# The pattern accepts ``3``, ``3.5``, ``.5``, ``3.``, and ``-3.5``.
# Bare-numeric callers (is_numeric_pv / parse_numeric_pv) must apply
# this BEFORE outer-punctuation stripping or after a numeric-preserving
# normalization — otherwise ``.5`` is silently stripped to ``5`` and
# misread as 5.0 instead of 0.5. See ``_numeric_normalize``.
_NUMERIC_RE = re.compile(r"^-?(?:\d+\.\d*|\.\d+|\d+)$")

# Code-like detection patterns. Conservative — only fire on values that
# look unambiguously code-shaped. A value containing prose alongside a
# code is NOT code-like (e.g. "irCR (Immune-Related Response Criteria)").
_CODE_PATTERNS = (
    # Pure numeric (any length): 0, 1, 12, 42, 3.5
    _NUMERIC_RE,
    # NCI Thesaurus / EVS-style concept code: C12345
    re.compile(r"^[Cc]\d{2,7}$"),
    # ICD-10-style: A00, A00.0, Z99.89 — letter + digits + optional .digits
    re.compile(r"^[A-Za-z]\d{2}(?:\.\d{1,4})?$"),
    # Generic short alphanumeric code: 1-3 letters + 2-7 digits with no
    # spaces, e.g. AB12, ICD9-250, MDR0042. Underscores/dashes allowed.
    re.compile(r"^[A-Za-z]{1,4}[-_]?\d{2,7}[A-Za-z0-9]*$"),
)


@dataclass
class ParsedPV:
    """Structured result of parse_pv_block.

    Attributes are populated even on malformed input — empty list / empty
    string / None — so callers can read them unconditionally.
    """

    pv_type: str = ""
    parsed_n: Optional[int] = None
    pv_values_raw: List[str] = field(default_factory=list)
    pv_values_norm: List[str] = field(default_factory=list)
    is_well_formed: bool = False

    @property
    def n_values(self) -> int:
        return len(self.pv_values_norm)

    @property
    def n_matches(self) -> Optional[bool]:
        """True iff a declared ``n=<N>`` matches the count of parsed
        values. None when ``n`` is absent. The check uses the post-dedup
        normalized count; mismatches commonly arise from duplicate raw
        values or from generic placeholders the producer counted
        differently."""
        if self.parsed_n is None:
            return None
        return self.parsed_n == self.n_values


def _is_missing(value: object) -> bool:
    """True for None, NaN, or strings that are empty / whitespace-only."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def normalize_pv_value(value: str) -> str:
    """Lowercase, collapse internal whitespace, strip outer punctuation.

    Parentheticals are preserved (design note §3.2 #4). Empty / None
    input yields ``""``.
    """
    if _is_missing(value):
        return ""
    s = str(value)
    s = s.lower()
    s = _WS_RE.sub(" ", s)
    s = s.strip(_NORM_STRIP_CHARS)
    return s


def dedupe_preserve_order(values: List[str]) -> List[str]:
    """Return values in input order, with duplicates removed.

    Equality is whatever ``==`` says — so the caller controls whether
    dedup happens before or after normalization. ``parse_pv_block``
    dedupes on the normalized form while preserving the first raw
    occurrence.
    """
    seen = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def is_generic_pv(value: str) -> bool:
    """True iff the normalized value is in the canonical generic-token
    set. Conservative — clinically-meaningful values must never match.
    """
    if _is_missing(value):
        return False
    return normalize_pv_value(value) in GENERIC_PV_TOKENS


def is_code_like_pv(value: str) -> bool:
    """True iff the value looks like a bare code rather than a
    natural-language label. Conservative; only fires on patterns that
    are unambiguous.

    Examples that match: ``"0"``, ``"42"``, ``"C12345"``, ``"A00.1"``,
    ``"icd9-250"``. Examples that do not match: ``"irCR"``,
    ``"irCR (Immune-Related Response Criteria)"``, ``"5 stars"``,
    ``"Grade 3"``.
    """
    if _is_missing(value):
        return False
    s = normalize_pv_value(value)
    if not s:
        return False
    if " " in s:
        return False
    return any(p.match(s) for p in _CODE_PATTERNS)


def extract_pv_values(text: Optional[str]) -> List[str]:
    """Return the raw pipe-delimited PV values from a PV block.

    Tolerant: returns ``[]`` for None / NaN / blank / malformed.
    Whitespace around each value is stripped; truly empty entries
    between consecutive pipes are dropped. Order preserved; no dedup.
    """
    if _is_missing(text):
        return []
    s = str(text)
    m = _HEADER_RE.match(s)
    body: Optional[str]
    if m is not None:
        body = m.group("body")
    else:
        # Fallback: accept a bare "<v1> | <v2>" body without the
        # PV_TYPE prefix. Some callers may want to pass just the body.
        if "|" in s and ":" not in s.split("|", 1)[0]:
            body = s
        else:
            return []
    if body is None or body.strip() == "":
        return []
    raw = [part.strip() for part in body.split("|")]
    return [r for r in raw if r != ""]


def parse_pv_block(text: Optional[str]) -> ParsedPV:
    """Parse a PV block into a ``ParsedPV``.

    Always returns a ``ParsedPV``; never raises on input shape. Use
    ``ParsedPV.is_well_formed`` to distinguish "the header matched and
    we extracted values cleanly" from "we recovered some values via
    fallback parsing or returned an empty structure".
    """
    if _is_missing(text):
        return ParsedPV()
    s = str(text)
    m = _HEADER_RE.match(s)
    if m is None:
        # Fallback: still try to extract values if the body looks like a
        # pipe-delimited list. is_well_formed stays False.
        raw_values = extract_pv_values(s)
        norm = [normalize_pv_value(v) for v in raw_values]
        pairs = list(zip(raw_values, norm))
        kept_raw: List[str] = []
        kept_norm: List[str] = []
        seen: set = set()
        for r, n in pairs:
            if n == "" or n in seen:
                continue
            seen.add(n)
            kept_raw.append(r)
            kept_norm.append(n)
        return ParsedPV(
            pv_type="",
            parsed_n=None,
            pv_values_raw=kept_raw,
            pv_values_norm=kept_norm,
            is_well_formed=False,
        )

    pv_type = (m.group("pv_type") or "").upper()
    parsed_n_str = m.group("pv_n")
    parsed_n = int(parsed_n_str) if parsed_n_str is not None else None
    body = m.group("body")
    if body is None:
        raw_values: List[str] = []
    else:
        raw_values = [part.strip() for part in body.split("|")]
        raw_values = [r for r in raw_values if r != ""]

    # Normalize and dedupe by normalized form, keeping the first raw form
    # that produced each unique normalized value.
    kept_raw = []
    kept_norm = []
    seen = set()
    for r in raw_values:
        n = normalize_pv_value(r)
        if n == "" or n in seen:
            continue
        seen.add(n)
        kept_raw.append(r)
        kept_norm.append(n)

    return ParsedPV(
        pv_type=pv_type,
        parsed_n=parsed_n,
        pv_values_raw=kept_raw,
        pv_values_norm=kept_norm,
        is_well_formed=True,
    )


# ---------------------------------------------------------------------------
# Numeric / code-like helpers
#
# Numeric handling rationale (see design note §3 "Numeric / code-like PV
# handling"): bare integers / floats are common (grades, dose levels,
# counts) and SHOULD NOT be dropped. They are weak-to-moderate structured
# evidence — exact set overlap is informative ("SDE=[2,4] and CDE=[1,3,4,5]
# share 4") but SapBERT cosine on bare numbers is not. Downstream features
# should expose numeric overlap separately from text-synonym overlap and
# let XGBoost weight them independently.
# ---------------------------------------------------------------------------


def _numeric_normalize(value: object) -> str:
    """Numeric-preserving normalization.

    Lowercases and collapses whitespace, but does NOT strip leading or
    trailing periods — doing so would silently turn ``".5"`` into
    ``"5"`` (a different number). Returns ``""`` for missing inputs.
    """
    if _is_missing(value):
        return ""
    s = str(value).lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def is_numeric_pv(value: object) -> bool:
    """True iff the value is a bare integer or simple decimal.

    Strict: does not match values with letters, whitespace, or other
    structure. Examples of NON-matches: ``"Grade 3"``, ``"Stage IV"``,
    ``"I-125"``, ``"PD-103"``, ``"R175C"``, ``"5 stars"``, ``""``,
    ``"1,000"``, ``"1.0e3"``, ``"+3"``.

    Note: every value that is_numeric_pv is also is_code_like_pv (the
    numeric pattern is the first code-like pattern). The reverse is not
    true — alphanumeric codes are code-like but not numeric.
    """
    s = _numeric_normalize(value)
    if not s:
        return False
    return _NUMERIC_RE.match(s) is not None


def parse_numeric_pv(value: object) -> Optional[float]:
    """Return the value as a float if it is a bare numeric PV, else None.

    Never raises. ``"3"`` → ``3.0``; ``"3.5"`` → ``3.5``; ``"-2"`` →
    ``-2.0``; ``".5"`` → ``0.5``; ``"3."`` → ``3.0``;
    ``"Grade 3"`` → ``None``; ``""`` → ``None``; ``None`` → ``None``.
    """
    s = _numeric_normalize(value)
    if not s or _NUMERIC_RE.match(s) is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):  # pragma: no cover — regex blocks bad input
        return None


@dataclass
class PVValueInfo:
    """Per-value classification result from ``classify_pv_value``.

    Held as a dataclass so callers can read fields by name without
    juggling dict key constants. ``numeric_value`` is ``None`` whenever
    ``is_numeric`` is False.
    """

    normalized: str = ""
    is_generic: bool = False
    is_code_like: bool = False
    is_numeric: bool = False
    numeric_value: Optional[float] = None


def classify_pv_value(value: object) -> PVValueInfo:
    """Classify a single PV value across all known flags.

    Cheap wrapper around the individual classifiers — useful when a
    caller needs every flag for the same value and wants to avoid
    re-normalizing four times. Missing / blank inputs return an empty
    ``PVValueInfo``.
    """
    if _is_missing(value):
        return PVValueInfo()
    norm = normalize_pv_value(value)  # type: ignore[arg-type]
    if norm == "":
        return PVValueInfo()
    # Numeric detection uses the numeric-preserving form so ``".5"``
    # parses to 0.5, not 5.0. The general `normalized` field still uses
    # the synonym-friendly form.
    numeric_value = parse_numeric_pv(value)
    is_num = numeric_value is not None
    return PVValueInfo(
        normalized=norm,
        is_generic=norm in GENERIC_PV_TOKENS,
        is_code_like=(" " not in norm) and any(p.match(norm) for p in _CODE_PATTERNS),
        is_numeric=is_num,
        numeric_value=numeric_value,
    )


def numeric_pv_set(values: List[object]) -> Set[float]:
    """Return the set of bare-numeric values from a PV list.

    Non-numeric values (labels, alphanumeric codes, missing entries)
    are silently skipped — they belong in the text-synonym pipeline,
    not the numeric-overlap pipeline.
    """
    out: Set[float] = set()
    for v in values:
        f = parse_numeric_pv(v)
        if f is not None:
            out.add(f)
    return out


def numeric_overlap_features(
    sde_values: List[object],
    cde_values: List[object],
) -> Dict[str, object]:
    """Compute exact numeric-overlap features for a (SDE, CDE) pair.

    Inputs are PV lists from either side (raw or normalized — the
    function re-normalizes). Returns a dict with stable keys, suitable
    for joining into a candidate-pair feature row. All counts are
    ints; all fractions are floats in [0, 1]; presence flags are bool.
    Fractions are 0.0 when their denominator side has no numeric PVs.

    Feature semantics:
      - ``n_numeric_sde`` / ``n_numeric_cde``: how many bare numbers
        each side carries (post-dedup as a set).
      - ``n_numeric_overlap``: |sde ∩ cde|.
      - ``numeric_jaccard``: |∩| / |∪|; 0.0 when union is empty.
      - ``numeric_sde_overlap_fraction``: |∩| / |sde_set|; 0.0 when
        ``n_numeric_sde == 0``.
      - ``numeric_cde_overlap_fraction``: |∩| / |cde_set|; 0.0 when
        ``n_numeric_cde == 0``.
      - ``both_have_numeric_pvs``: convenience presence flag.
      - ``only_sde_has_numeric_pvs`` / ``only_cde_has_numeric_pvs``:
        asymmetric presence flags. All three flags are False when
        neither side has numerics — callers should pair these with the
        higher-level ``pv_missing_*`` sentinels.
    """
    sde_set = numeric_pv_set(sde_values)
    cde_set = numeric_pv_set(cde_values)
    overlap = sde_set & cde_set
    union = sde_set | cde_set
    n_sde = len(sde_set)
    n_cde = len(cde_set)
    n_overlap = len(overlap)
    return {
        "n_numeric_sde": n_sde,
        "n_numeric_cde": n_cde,
        "n_numeric_overlap": n_overlap,
        "numeric_jaccard": (n_overlap / len(union)) if union else 0.0,
        "numeric_sde_overlap_fraction": (n_overlap / n_sde) if n_sde else 0.0,
        "numeric_cde_overlap_fraction": (n_overlap / n_cde) if n_cde else 0.0,
        "both_have_numeric_pvs": (n_sde > 0) and (n_cde > 0),
        "only_sde_has_numeric_pvs": (n_sde > 0) and (n_cde == 0),
        "only_cde_has_numeric_pvs": (n_cde > 0) and (n_sde == 0),
    }
