"""Export the E5 fixed-weight ablation as a LaTeX appendix table.

Reads ``artifacts_v2_ablation/tables/summary.json`` (the grouped mean + seed
standard deviation over the 10 deterministic seeds) and emits a booktabs table
contrasting the confidence-decayed and fixed-weight peak formation RMSE across
the full {10,30,60}s duration sweep, for both the nominal (E3) and accelerating
(E4) scenarios.  The result is written to
``artifacts_v2_ablation/tables/ablation_full.tex`` and is ``\\input`` by the
manuscript, so the appendix table traces back to the raw records through the
same pipeline as every other reported number.

No third-party dependencies: only the standard library.
"""

from __future__ import annotations

import json
from math import floor, log10
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "artifacts_v2_ablation" / "tables" / "summary.json"
OUT = ROOT / "artifacts_v2_ablation" / "tables" / "ablation_full.tex"

DURATIONS = (10.0, 30.0, 60.0)

# (scenario label, experiment key, [(prediction label, decayed key, fixed key)])
SCENARIOS = [
    (
        "Nominal",
        "e3",
        [
            ("Constant-velocity", "cv_prediction", "cv_fixed"),
            ("Zero-order hold", "zoh", "zoh_fixed"),
        ],
    ),
    (
        "Accelerating",
        "e4",
        [
            ("Constant-velocity", "cv_prediction", "cv_fixed"),
            ("Zero-order hold", "zoh", "zoh_fixed"),
        ],
    ),
]


def sig(x: float, n: int) -> str:
    """Format ``x`` to ``n`` significant figures, keeping trailing zeros."""
    if x == 0:
        return "0"
    exp = floor(log10(abs(x)))
    decimals = n - 1 - exp
    if decimals <= 0:
        return f"{round(x, -decimals):.0f}"
    return f"{x:.{decimals}f}"


def cell(mean: float, std: float) -> str:
    # 4 significant figures for the mean, 2 for the seed std.
    return f"${sig(mean, 4)} \\pm {sig(std, 2)}$"


def load() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def build(records: list[dict], policy: str, duration: float) -> dict:
    for r in records:
        if r["policy"] == policy and r["duration"] == duration:
            return r
    raise KeyError(f"missing record for {policy} @ {duration}s")


def render() -> str:
    data = load()
    lines: list[str] = []
    lines.append("\\begin{tabular}{lllcc}")
    lines.append("\\toprule")
    lines.append(
        "Scenario & Prediction & $\\tau$ (s) & Decayed peak (m) & Fixed-weight peak (m) \\\\"
    )
    lines.append("\\midrule")

    n_scenarios = len(SCENARIOS)
    for si, (scenario, exp, policies) in enumerate(SCENARIOS):
        records = data[exp]
        for pi, (pred_label, decayed, fixed) in enumerate(policies):
            for di, duration in enumerate(DURATIONS):
                scenario_cell = (
                    f"\\multirow{{6}}{{*}}{{{scenario}}}"
                    if (pi == 0 and di == 0)
                    else ""
                )
                pred_cell = (
                    f"\\multirow{{3}}{{*}}{{{pred_label}}}" if di == 0 else ""
                )
                d = build(records, decayed, duration)
                f = build(records, fixed, duration)
                decayed_cell = cell(
                    d["denial_peak_rmse"], d["denial_peak_rmse_std"]
                )
                fixed_cell = cell(
                    f["denial_peak_rmse"], f["denial_peak_rmse_std"]
                )
                row = (
                    f"{scenario_cell} & {pred_cell} & {int(duration)} & "
                    f"{decayed_cell} & {fixed_cell} \\\\"
                )
                lines.append(row)
        if si < n_scenarios - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    table = render()
    OUT.write_text(table, encoding="utf-8")
    print(f"wrote {OUT}")
    print(table)


if __name__ == "__main__":
    main()
