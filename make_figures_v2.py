"""Regenerate publication figures from an existing ``summary.json``.

This runner reproduces the three manuscript figures (E1 convergence, E2 Pareto,
E3 denial) from the grouped data already emitted by ``experiments_v2.py``,
without re-running the full experiment suite.  It is the standalone entry point
for the redesigned, colorblind-safe figures in
``strict_admm_dmpc/plotting.py``.

Usage::

    python make_figures_v2.py [path/to/summary.json] [output_dir]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strict_admm_dmpc.plotting import render_all


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=root / "artifacts_v2_full" / "tables" / "summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: summary.json's parent directory)",
    )
    args = parser.parse_args()

    summary_path = args.summary
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir = args.output or summary_path.parent
    generated = render_all(summary, output_dir)
    for name, path in generated.items():
        print(f"[{name}] {path.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
