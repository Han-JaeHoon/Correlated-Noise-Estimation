"""Build this week's meeting deck (.pptx -> opens in Keynote).

Mirrors last week's deck flow, updated for the Phase 2 (realistic surface code)
progress. Run: python scripts/build_meeting_deck.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "meeting" / "weekly_2026-06-19.pptx"

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTE = RGBColor(0x5A, 0x6A, 0x80)
ACC = RGBColor(0x2C, 0x4E, 0x86)
GREEN = RGBColor(0x2A, 0x7A, 0x55)
RED = RGBColor(0xC0, 0x39, 0x39)
BG = RGBColor(0xF4, 0xF5, 0xF7)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W = prs.slide_width


def _bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG


def _tb(slide, l, t, w, h):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def title_slide(title, subtitle):
    s = prs.slides.add_slide(BLANK); _bg(s)
    tf = _tb(s, 1, 2.7, 11.3, 1.4)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = title
    r.font.size = Pt(46); r.font.bold = True; r.font.color.rgb = INK
    tf2 = _tb(s, 1, 4.1, 11.3, 0.8)
    p = tf2.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = subtitle
    r.font.size = Pt(22); r.font.color.rgb = MUTE
    return s


def header(slide, title, kicker=None):
    tf = _tb(slide, 0.6, 0.35, 12.1, 1.1)
    if kicker:
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = kicker.upper()
        r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = ACC
        p2 = tf.add_paragraph()
    else:
        p2 = tf.paragraphs[0]
    r = p2.add_run(); r.text = title
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = INK
    # underline rule
    ln = slide.shapes.add_textbox(Inches(0.62), Inches(1.5), Inches(12), Inches(0.05))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACC


def bullets(slide, items, left=0.7, top=1.8, width=7.2, size=18):
    tf = _tb(slide, left, top, width, 5.2)
    for i, it in enumerate(items):
        lvl = 0
        color = INK
        if isinstance(it, tuple):
            it, lvl, *rest = it
            color = rest[0] if rest else INK
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        bullet = "•  " if lvl == 0 else "–  "
        r = p.add_run(); r.text = bullet + it
        r.font.size = Pt(size - 2 * lvl); r.font.color.rgb = color
        p.space_after = Pt(8)


def image(slide, path, l, t, w, caption=None):
    pic = slide.shapes.add_picture(str(ROOT / path), Inches(l), Inches(t), width=Inches(w))
    if caption:
        tf = _tb(slide, l, t + pic.height / 914400 + 0.05, w, 0.4)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = caption
        r.font.size = Pt(11); r.font.italic = True; r.font.color.rgb = MUTE
    return pic


def content(title, kicker, items, **kw):
    s = prs.slides.add_slide(BLANK); _bg(s)
    header(s, title, kicker); bullets(s, items, **kw)
    return s


# ───────────────────────── slides ─────────────────────────
title_slide("Weekly Meeting", "2026.06.19   ·   Jaehun Han")

content("Last meeting — Phase 0 (naive, frozen)", "recap", [
    "d=3 rotated surface code, toy sequential schedule (PennyLane state-vector).",
    "Correlated error fixed to a narrow case: one faulty CNOT (1 of 24), recurring.",
    "Task: identify which CNOT, from syndrome data — two data views, both classified well:",
    ("Sequential  (T, d²−1)  long stream  →  GRU  95.3 % / 91.2 %", 1, GREEN),
    ("Spatial  (N, d, d²−1)  bag-of-shots  →  LogReg  93.9 % (N=300)", 1, GREEN),
    ("Within-distance ambiguity groups (e.g. {6,7}, {14,15,17}) cap per-class accuracy.", 1, MUTE),
])

content("Direction for this phase", "plan", [
    "Implement a realistic, general-d, constant-depth surface code.",
    "Implement compatible MWPM decoding (open-source allowed).",
    "Repeat the Phase 0 classification on the realistic circuit.",
    ("Spatial data: use (N, d·k, d²−1) instead of (N, d, d²−1) — length-d·k shots, k = 1,2,3.", 1),
    ("Decoding per d rounds may be added later; not applied for now.", 1, MUTE),
])

s = content("Progress 1 — General-d surface code", "done", [
    "Constant-depth 4-tick parallel schedule (depth independent of d).",
    "Built from first principles, validated against Stim:",
    ("ancilla set + CNOT schedule  ==  Stim generator (bit-for-bit)", 1, GREEN),
    ("code distance == d ; logical-error-rate matches (d=3/5/7)", 1, GREEN),
    "Backend: Stim (Clifford) — scales where state-vector cannot (d=5 → 49 qubits).",
], width=6.4)
image(s, "viz/preview_d3.png", 7.2, 2.0, 5.6, "d=3 lattice (data · X/Z stabilizers)")

s = content("Progress 2 — MWPM decoding", "done", [
    "MWPM = standard surface-code baseline; via PyMatching 2.",
    "Driven by the circuit's detector error model (DEM) — no extra modeling.",
    ("noiseless → 0 logical errors ; below-threshold LER ↓ as d ↑", 1, GREEN),
    "Works d=3…9. Code → error → syndrome → matching → verdict visualized.",
], width=6.0)
image(s, "viz/decode_examples/decode_d5_success.png", 6.7, 2.3, 6.3,
      "decode pipeline (d=5): error → detection events → MWPM → no logical error")

content("Progress 3 — Realistic correlated-noise model", "in progress", [
    "Move beyond the artificial single-CNOT fault → physically-grounded correlated channels (survey).",
    "Candidate correlated locations = all qubit pairs within distance ≤ 2:",
    ("d=3:  24 data–anc  +  12 data–data  +  8 anc–anc  =  44 pairs  ( + None )", 1, ACC),
    "Channel per mechanism:",
    ("data–anc : stray ZZ during the gate     data–data : always-on ZZ", 1),
    ("anc–anc : readout crosstalk (XX)        background : i.i.d. (= None class)", 1),
    "Task → which pair (or None) is correlated, from (N, d·k, d²−1) syndrome data.",
])

s = prs.slides.add_slide(BLANK); _bg(s)
header(s, "Key finding — fundamental distinguishability", "result")
bullets(s, [
    "Exact test (Phase-0-style TV = 0): two classes are indistinguishable for ANY N iff their correlated error gives the same syndrome signature.",
    ("d=3:  45 classes  →  26 fundamentally distinguishable ; 6 silent (≡ None)", 1, RED),
    ("Invariant to window k (k=1,2,3) and to sample count N.", 1, MUTE),
    "Collisions only among ZZ channels; every anc–anc (readout) class is unique.",
    "Reason: syndrome sees only the net data Pauli (mod stabilizer) — same structure as Phase 0's ambiguity groups.",
], width=6.6, size=16)
image(s, "meeting/fig_distinguishability.png", 7.0, 2.2, 6.0)

content("Implications & open question", "discussion", [
    ("Perfect 44-way classification is impossible — like Phase 0's ambiguity groups.", 0, RED),
    "Honest target = classify the distinguishable equivalence groups (coarsened labels).",
    "Open modeling decision (input wanted):",
    ("the exact silent/collision set depends on the stray-ZZ injection convention", 1),
    ("(where in the round the correlated error is applied) — to be pinned down.", 1, MUTE),
])

content("Next steps", "next", [
    "1.  Fix the correlated-noise injection convention (physical stray-ZZ timing).",
    "2.  Generate the (N, d·k, d²−1) bag dataset per class, no decoding.",
    "3.  Train the bag-of-shots classifier (reuse Phase 0 pipeline); target = distinguishable groups.",
    "4.  Accuracy vs (N, k); compare to the TV=0 ceiling.",
    ("Infrastructure (surface code + MWPM + correlated model) is ready.", 0, GREEN),
])

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
