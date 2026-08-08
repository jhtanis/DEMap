#!/usr/bin/env python3
"""Generate the two editable manuscript diagrams as PowerPoint sources.

Outputs (all under manuscript/figures/):
  figure1_workflow.pptx                 — Figure 1, raw caDSR -> final evaluation
  figure_reranker_architecture.pptx     — detailed reranker architecture (Section 4.7)

Both are ordinary PowerPoint files: every box, arrow and label is a native shape
the author can move, retype or restyle. Run this script to regenerate them, or
edit the .pptx directly — the .pptx is the source of record.

Export to PNG/PDF with LibreOffice (see export_diagrams.sh).

Font sizes are chosen so the inserted figure stays >= 11 pt at manuscript width.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- style knobs
TITLE_PT = 20
BOX_TITLE_PT = 14
BOX_BODY_PT = 12
STAGE_LABEL_PT = 13
CAPTION_PT = 11

C_DATA = RGBColor(0xDD, 0xEB, 0xF7)      # data / inputs (light blue)
C_DATA_L = RGBColor(0x1F, 0x4E, 0x79)
C_MODEL = RGBColor(0xE2, 0xF0, 0xD9)     # models / training (light green)
C_MODEL_L = RGBColor(0x37, 0x56, 0x23)
C_RETR = RGBColor(0xFF, 0xF2, 0xCC)      # retrieval (light amber)
C_RETR_L = RGBColor(0x7F, 0x60, 0x00)
C_RANK = RGBColor(0xFB, 0xE4, 0xD5)      # reranking (light orange)
C_RANK_L = RGBColor(0x83, 0x3C, 0x0B)
C_EVAL = RGBColor(0xE6, 0xE0, 0xEC)      # evaluation (light purple)
C_EVAL_L = RGBColor(0x40, 0x30, 0x51)
C_LINE = RGBColor(0x59, 0x59, 0x59)


def _box(slide, x, y, w, h, title, body="", fill=C_DATA, line=C_DATA_L,
         title_pt=BOX_TITLE_PT, body_pt=BOX_BODY_PT, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    s.line.width = Pt(1.5)
    s.shadow.inherit = False
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Emu(45720)
    tf.margin_top = tf.margin_bottom = Emu(27432)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = title
    r.font.size = Pt(title_pt)
    r.font.bold = True
    r.font.color.rgb = line
    if body:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = body
        r2.font.size = Pt(body_pt)
        r2.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return s


def _arrow(slide, x1, y1, x2, y2, width_pt=2.0):
    from pptx.enum.shapes import MSO_CONNECTOR
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                   Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = C_LINE
    c.line.width = Pt(width_pt)
    # arrowhead
    ln = c.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "h": "med"})
    ln.append(tail)
    return c


def _label(slide, x, y, w, text, pt=STAGE_LABEL_PT, bold=True,
           color=RGBColor(0x25, 0x25, 0x25), align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.34))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(pt)
    r.font.bold = bold
    r.font.color.rgb = color
    return tb


def _blank_slide(width_in, height_in):
    prs = Presentation()
    prs.slide_width = Inches(width_in)
    prs.slide_height = Inches(height_in)
    return prs, prs.slides.add_slide(prs.slide_layouts[6])


# ---------------------------------------------------------------------------
# Figure 1 — end-to-end workflow
# ---------------------------------------------------------------------------
def build_workflow() -> Path:
    """Figure 1.

    Kept in step with manuscript/figures/make_figure1_workflow.py, which is the
    matplotlib generator that produces the vector PDF / high-resolution PNG
    actually embedded in the manuscript (LibreOffice/PowerPoint export is not
    available on the cluster). If you change one, change the other.

    Row 3 branches the candidate pool into three PARALLEL feature families that
    each feed the HGBC directly. Do not redraw it as a sequential chain: that
    would imply the bi-encoder and keyword features are outputs of the
    FT-MedCPT cross-encoder, which they are not.
    """
    W, H = 13.7, 8.7
    prs, s = _blank_slide(W, H)
    _label(s, 0.35, 0.14, 13.0,
           "Source-to-CDE mapping workflow", pt=TITLE_PT, align=PP_ALIGN.CENTER)

    # ---- Row 1: data -------------------------------------------------------
    y = 0.72
    _label(s, 0.30, y - 0.06, 4.6, "1  Data and benchmark construction",
           pt=STAGE_LABEL_PT, color=C_DATA_L)
    for x, title, body in (
        (0.30, "caDSR export",
         "January 12, 2026\n79,479-record benchmark-\nconstruction catalog"),
        (3.65, "Benchmark construction",
         "ALT / REF source-like queries,\nexpert-linked gold CDEs,\n"
         "69,102 constructed query-CDE pairs"),
        (7.00, "Canonical deduplication and\nreachable-gold filtering",
         "68,142 analysis-ready pairs"),
        (10.35, "Production catalog",
         "June 18, 2026\n62,976 records,\n62,858 unique CDE public IDs"),
    ):
        _box(s, x, y + 0.32, 3.05, 1.12, title, body, C_DATA, C_DATA_L)
    for x0 in (3.35, 6.70, 10.05):
        _arrow(s, x0, y + 0.88, x0 + 0.30, y + 0.88)

    # ---- Row 2: bi-encoder development ------------------------------------
    y = 2.36
    _label(s, 0.30, y - 0.06, 4.0, "2  Bi-encoder development", pt=STAGE_LABEL_PT, color=C_MODEL_L)
    for x, title, body in (
        (0.30, "Text-representation screening",
         "4 query x 10 CDE recipes,\noff the shelf, Validation Dev"),
        (3.65, "Representation x loss screening", "2 x 2 screen per model,\ntwo seeds"),
        (7.00, "Phase 1 fine-tuning", "in-batch negatives,\nhyperparameter sweep"),
        (10.35, "Phase 2 hard-negative refinement", "mined hard negatives\n-> FT-MPNet selected"),
    ):
        _box(s, x, y + 0.32, 3.05, 1.12, title, body, C_MODEL, C_MODEL_L)
    for x0 in (3.35, 6.70, 10.05):
        _arrow(s, x0, y + 0.88, x0 + 0.30, y + 0.88)

    # ---- Row 3: candidate generation + reranking ---------------------------
    y = 4.00
    _label(s, 0.30, y, 4.2, "3  Candidate generation and reranking",
           pt=STAGE_LABEL_PT, color=C_RANK_L)
    _box(s, 0.30, 4.32, 2.45, 1.00, "FT-MPNet", "semantic retrieval, top 20",
         C_RETR, C_RETR_L)
    _box(s, 0.30, 5.50, 2.45, 1.00, "CDE Match-Fuzzy",
         "keyword retrieval, top 10", C_RETR, C_RETR_L)
    _box(s, 3.05, 4.74, 2.45, 1.34, "Candidate pool",
         "union, deduplicated\nby CDE public identifier\n~30 candidates",
         C_RETR, C_RETR_L)
    _arrow(s, 2.75, 4.82, 3.05, 5.15)
    _arrow(s, 2.75, 6.00, 3.05, 5.67)

    # three PARALLEL feature families computed per pooled candidate
    fx, fw, fh = 6.10, 3.55, 0.90
    fys = (4.10, 5.20, 6.30)
    for (title, body), fy in zip((
        ("Bi-encoder features", "similarity, rank, margins"),
        ("Keyword features", "rule, rank, fuzzy and\ntoken-overlap evidence"),
        ("FT-MedCPT cross-encoder features", "score, rank, within-query z"),
    ), fys):
        _box(s, fx, fy, fw, fh, title, body, C_RANK, C_RANK_L, body_pt=11)
        _arrow(s, 5.50, 5.41, fx, fy + fh / 2)

    _box(s, 10.55, 4.96, 2.85, 0.90, "HGBC reranker",
         "117 features, score descending,\nties broken by CDE public ID",
         C_EVAL, C_EVAL_L, body_pt=11)
    for fy in fys:
        _arrow(s, fx + fw, fy + fh / 2, 10.55, 5.41)

    # ---- Row 4: evaluation --------------------------------------------------
    _label(s, 0.30, 7.42, 5.0, "4  Evaluation", pt=STAGE_LABEL_PT, color=C_EVAL_L)
    _box(s, 3.05, 7.74, 10.35, 0.72,
         "Internal test set and five distribution-shifted holdouts "
         "(CCTG, OID ALT, CDASH, GDC, CIMAC)",
         "", C_EVAL, C_EVAL_L, body_pt=CAPTION_PT)
    _arrow(s, 11.97, 5.86, 11.97, 7.74)

    out = OUT / "figure1_workflow.pptx"
    prs.save(out)
    return out


# ---------------------------------------------------------------------------
# Reranker architecture (Section 4.7)
# ---------------------------------------------------------------------------
def build_architecture() -> Path:
    """Figure 6.

    The candidate pool branches directly into three PARALLEL feature families —
    bi-encoder, keyword and FT-MedCPT cross-encoder — which all feed the HGBC.
    The cross-encoder is one of the three feature producers, not a sequential
    stage upstream of the other two.
    """
    W, H = 12.6, 7.5
    prs, s = _blank_slide(W, H)
    _label(s, 0.3, 0.14, 12.0, "Final reranker architecture", pt=TITLE_PT,
           align=PP_ALIGN.CENTER)

    # query
    _box(s, 4.40, 0.72, 3.80, 0.78, "Source query text",
         "alternate name or question text (+ permissible values)", C_DATA, C_DATA_L)

    # two retrievers
    _box(s, 1.20, 1.94, 4.10, 1.02, "FT-MPNet bi-encoder",
         "cosine search over the production catalog\ntop 20 candidates",
         C_RETR, C_RETR_L)
    _box(s, 7.30, 1.94, 4.10, 1.02, "Keyword retriever",
         "CDE Match-Fuzzy: exact, fuzzy\nand PV rules, top 10 candidates", C_RETR, C_RETR_L)
    _arrow(s, 5.60, 1.50, 3.25, 1.94)
    _arrow(s, 7.00, 1.50, 9.35, 1.94)

    # union / candidate pool
    _box(s, 3.55, 3.36, 5.50, 0.92, "Union and deduplication by CDE public ID",
         "at most ~30 candidates per query", C_RETR, C_RETR_L)
    _arrow(s, 3.25, 2.96, 5.20, 3.36)
    _arrow(s, 9.35, 2.96, 7.40, 3.36)

    # three PARALLEL feature families, computed per candidate over the pool
    fy = 4.78
    _box(s, 0.55, fy, 3.55, 1.02, "Bi-encoder features",
         "FT-MPNet similarity, rank\nand per-query normalisations",
         C_RANK, C_RANK_L, body_pt=11)
    _box(s, 4.45, fy, 3.55, 1.02, "Keyword features",
         "CDE Match-Fuzzy rule, rank, fuzzy,\ntoken-overlap and PV evidence",
         C_RANK, C_RANK_L, body_pt=11)
    _box(s, 8.35, fy, 3.55, 1.02, "FT-MedCPT cross-encoder features",
         "score, rank, margins,\nwithin-query z-score",
         C_RANK, C_RANK_L, body_pt=11)
    # the pool branches directly into all three families
    _arrow(s, 5.20, 4.28, 2.30, fy)
    _arrow(s, 6.30, 4.28, 6.30, fy)
    _arrow(s, 7.40, 4.28, 10.10, fy)

    # HGBC
    _box(s, 2.95, 6.30, 6.70, 0.86, "HGBC reranker  ->  final score and deterministic rank",
         "117 features  ·  score descending, ties broken by CDE public ID",
         C_EVAL, C_EVAL_L, body_pt=11)
    for x in (2.30, 6.30, 10.10):
        _arrow(s, x, fy + 1.02, 6.30, 6.30)

    out = OUT / "figure_reranker_architecture.pptx"
    prs.save(out)
    return out


if __name__ == "__main__":
    print("wrote", build_workflow())
    print("wrote", build_architecture())
