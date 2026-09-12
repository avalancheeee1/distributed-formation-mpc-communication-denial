"""Generate tight, editable SVG versions of the two method figures.

The MDPI Word pipeline embeds 600-dpi PNGs, but reviewers/authors often want
vector figures for the methods schematic (e5) and the protocol flowchart (e6).

Two fixes make the SVG pleasant to edit in a vector editor:

1. ``bbox_inches="tight"`` crops the viewBox to the drawn content instead of
   the full figure canvas (which carries empty subplot margins).
2. Post-processing strips every ``clip-path="url(#...)"`` reference and its
   ``<clipPath>`` definition.  Matplotlib wraps each artist in a clip path that
   spans the whole axes; after ungrouping, those invisible clip rectangles
   show up as oversized bounding boxes.  Neither method figure clips anything
   (all content is inside the axes), so removing them is lossless.

Text is left as editable glyphs (``svg.fonttype="none"``).
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_figures_v6 as m

OUT = m.OUT


def _strip_clip_paths(svg: str) -> str:
    """Remove clip-path references and clipPath definitions from an SVG string."""
    # attribute references first (before we delete the definitions)
    svg = re.sub(r'\sclip-path="url\(#[^)]*\)"', "", svg)
    # clipPath definitions (may be empty or nested; non-greedy across the block)
    svg = re.sub(r"<clipPath[^>]*>.*?</clipPath>", "", svg, flags=re.DOTALL)
    return svg


def _save_svg(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    text = svg_path.read_text(encoding="utf-8")
    cleaned = _strip_clip_paths(text)
    svg_path.write_text(cleaned, encoding="utf-8")


m._save = _save_svg
m.plot_admm_split(OUT / "e5_admm_split")
m.plot_protocol(OUT / "e6_protocol")

for stem in ("e5_admm_split", "e6_protocol"):
    svg = (OUT / stem).with_suffix(".svg")
    head = svg.read_text(encoding="utf-8").split(">", 1)[0]
    clips = svg.read_text(encoding="utf-8").count("clip-path")
    print(f"wrote {svg.name}: {head[:90]}... clip-path refs remaining: {clips}")
