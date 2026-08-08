#!/usr/bin/env python3
"""Paper v13 §38: recompute the S4 dataset-statistics tables from the canonical data.

Datasets (8 rows, GDC as ONE combined row):
  Train (data/processed/splits/train.parquet)
  Validation Dev (data/processed/splits/val_dev.parquet)
  Test, CCTG, OID ALT, CDASH, GDC combined, CIMAC (data/processed/eval_canonical/)

Table A — provenance shift (one row per dataset):
  n_queries; %ALT-provenance rows; mean query tokens; %code-like queries;
  top family and its share.
Table B — style shift, Train as reference:
  dominant style; mean query tokens; %code-like; %underscore-containing;
  %question-mark; token-unigram Jensen-Shannon divergence vs Train.

Definitions (stated in the table notes; computed per unique query):
  query text     = query_text with any ALT_NAME:/REF_TEXT: prefix stripped.
  tokens         = [A-Za-z0-9]+ matches.
  code-like      = single schema-style identifier: no internal whitespace
                   (matches ^[A-Za-z0-9_.\\-]+$).
  ALT provenance = query_source == 'ALT' (caDSR alternate-name harvest);
                   external sets report their own source values verbatim.
  dominant style = schema_like if code-like >= 60%, natural_language if
                   code-like <= 40%, else mixed.
  JSD            = Jensen-Shannon divergence (natural log) between the
                   dataset's token-unigram distribution and Train's,
                   lowercased, additive smoothing 1e-8.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/vf/users/nextgen2/james/tasks/cde_project/demap")
OUT = REPO / ".scratch/demap/paper_v13_scientific_audit"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CODE_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

DATASETS = [
    ("Train", REPO / "data/processed/splits/train.parquet"),
    ("Validation Dev", REPO / "data/processed/splits/val_dev.parquet"),
    ("Test", REPO / "data/processed/eval_canonical/test.parquet"),
    ("CCTG", REPO / "data/processed/eval_canonical/cctg.parquet"),
    ("OID ALT", REPO / "data/processed/eval_canonical/oid_alt.parquet"),
    ("CDASH", REPO / "data/processed/eval_canonical/cdash.parquet"),
    ("GDC combined", REPO / "data/processed/eval_canonical/gdc_combined.parquet"),
    ("CIMAC", REPO / "data/processed/eval_canonical/cimac_v2.parquet"),
]


def strip_prefix(q) -> str:
    q = str(q) if q is not None else ""
    for p in ("ALT_NAME:", "REF_TEXT:"):
        if q.startswith(p):
            return q[len(p):].strip()
    return q.strip()


def js_divergence(p_counts: Counter, q_counts: Counter, alpha: float = 1e-8) -> float:
    vocab = sorted(set(p_counts) | set(q_counts))
    p = np.array([p_counts.get(t, 0) for t in vocab], dtype=float) + alpha
    q = np.array([q_counts.get(t, 0) for t in vocab], dtype=float) + alpha
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def per_dataset(name: str, path: Path):
    df = pd.read_parquet(path)
    # style is measured on the BARE source string (no ALT_NAME:/REF_TEXT: label,
    # no appended PV/value-domain block) — query_text_raw where available.
    qcol = "query_text_raw" if "query_text_raw" in df.columns else "query_text"
    d = df.drop_duplicates("query_id").copy()
    txt = d[qcol].map(strip_prefix)
    toks = txt.map(lambda s: TOKEN_RE.findall(s))
    code_like = txt.map(lambda s: bool(CODE_RE.match(s)) and len(s) > 0)
    src = (df["query_source"].astype(str).value_counts(normalize=True)
           if "query_source" in df.columns else pd.Series(dtype=float))
    alt_share = float(src.get("ALT", 0.0))
    source_desc = ", ".join(f"{k} {v:.1%}" for k, v in src.head(3).items())
    if "family" in d.columns and d["family"].notna().any():
        fam = d["family"].astype(str).value_counts(normalize=True)
        top_family, top_share = fam.index[0], float(fam.iloc[0])
    else:
        top_family, top_share = "—", np.nan
    counts = Counter(t.lower() for ts in toks for t in ts)
    row = {
        "dataset": name,
        "n_queries": int(d["query_id"].nunique()),
        "n_rows": int(len(df)),
        "alt_provenance_share": alt_share,
        "query_source_values": source_desc,
        "mean_query_tokens": float(toks.map(len).mean()),
        "code_like_share": float(code_like.mean()),
        "underscore_share": float(txt.str.contains("_").mean()),
        "question_mark_share": float(txt.str.contains(r"\?").mean()),
        "top_family": top_family,
        "top_family_share": top_share,
    }
    return row, counts


def main(argv=None) -> int:
    """Compute the Table S1 / Table S2 dataset-characterization statistics."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=str(REPO))
    ap.add_argument('--out-dir', default=str(OUT))
    args = ap.parse_args(argv)
    globals()['REPO'] = Path(args.root)
    globals()['OUT'] = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, token_counts = [], {}
    for name, path in DATASETS:
        r, c = per_dataset(name, path)
        rows.append(r)
        token_counts[name] = c

    t = pd.DataFrame(rows)
    train_counts = token_counts["Train"]
    t["token_jsd_vs_train"] = [js_divergence(token_counts[n], train_counts)
                               if n != "Train" else 0.0 for n in t["dataset"]]
    t["dominant_style"] = np.where(t["code_like_share"] >= 0.60, "schema_like",
                                   np.where(t["code_like_share"] <= 0.40,
                                            "natural_language", "mixed"))

    prov_cols = ["dataset", "n_queries", "alt_provenance_share", "query_source_values",
                 "mean_query_tokens", "code_like_share", "top_family", "top_family_share"]
    style_cols = ["dataset", "dominant_style", "mean_query_tokens", "code_like_share",
                  "underscore_share", "question_mark_share", "token_jsd_vs_train"]
    t[prov_cols].to_csv(OUT / "s4_provenance_shift_v13.csv", index=False)
    t[style_cols].to_csv(OUT / "s4_style_shift_v13.csv", index=False)
    print("=== S4 Table A: provenance shift ===")
    print(t[prov_cols].round(3).to_string(index=False))
    print("\n=== S4 Table B: style shift (Train reference) ===")
    print(t[style_cols].round(3).to_string(index=False))
    print(f"\nwrote {OUT}/s4_provenance_shift_v13.csv and s4_style_shift_v13.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
