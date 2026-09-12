"""make_figures_v5.py — journal-grade redesign for MDPI Mathematics.

Design goals (vs v4), following the scipilot / nature-figure skill guidance:

* **Serif typography matched to the manuscript.**  The MDPI body is Palatino
  Linotype, so the figures use the same family with STIX math glyphs (rho,
  xi^k) so labels sit in one visual family instead of clashing with a sans font.
* **Render at final size — no rescaling.**  The canvas width equals the width
  the manuscript embeds each figure at (e1 5.58 in; e2/e3/e4 5.83 in).  This
  keeps every label at its intended point size (7-8 pt) instead of letting the
  Word step shrink it.
* **Type is smaller and the canvas is roomier** — 7-8 pt labels, generous
  x/y margins and panel spacing — fixing text crowding the data and the
  too-tight panels.
* **Refined colourblind-safe palette** (Okabe-Ito hues) with redundant marker
  shapes; the decayed-vs-fixed ablation is carried by fill (hollow vs solid),
  not by new colours.
* **Dot plots, not polylines**, for the categorical denial durations
  (10/30/60 s).  A line joining three categorical points implies a false
  continuous interpolation; points + error bars are the honest encoding.
* **No annotation clutter**: mechanism labels ("saturates", "grows") sit in
  the empty corner of each panel, never pointing an arrow into the data.

Output stems unchanged (e1_convergence, e2_budget, e3_denial, e4_acceleration)
so the manuscript pipeline picks these up without modification.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "artifacts_v2_ablation" / "tables" / "summary.json"
OUT = ROOT / "artifacts_v2_ablation" / "tables"

# --- refined Okabe-Ito palette (colourblind-safe) ----------------------------
IDEAL = "#0072B2"   # blue      — ideal / full communication
LOCAL = "#6B7280"   # grey      — uncoupled local MPC (neutral baseline)
CV = "#009E73"      # green     — constant-velocity prediction
ZOH = "#D55E00"     # vermilion — zero-order hold

HORIZON = {5: "#9EC5E0", 10: "#4C86C3", 20: "#1F4E79"}  # ordered blue ramp

INK = "#111111"
MUTED = "#3A3A3A"
GRID = "#E8EBEE"

MARKER = {"ideal": "o", "local_mpc": "D", "cv_prediction": "^", "zoh": "s"}

_MS = 5.0
_LW = 1.3


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "STIXGeneral",
                           "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": _LW,
            "lines.markersize": _MS,
        }
    )


def _strip(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, colors=MUTED)


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def _err_bounds(y: float, std: float) -> tuple[float, float]:
    """Asymmetric error bounds: on a log axis, keep the lower bar from crossing 0."""
    if y <= 0:
        return 0.0, std
    return min(std, y * 0.98), std


def _dot(
    ax: plt.Axes, x: float, y: float, std: float, color: str, marker: str,
    fill: bool = True, ms: float = _MS,
) -> None:
    lower, upper = _err_bounds(y, std)
    ax.errorbar(
        [x], [y],
        yerr=[[lower], [upper]],
        fmt=marker,
        linestyle="none",
        color=color,
        markerfacecolor=color if fill else "white",
        markeredgecolor=color,
        markeredgewidth=1.1,
        markersize=ms,
        ecolor=color,
        elinewidth=0.8,
        capsize=3.0,
        capthick=0.8,
        zorder=3,
    )


def _dodge(n: int, width: float = 0.7) -> list[float]:
    if n <= 1:
        return [0.0]
    return [width * (i - (n - 1) / 2) / n for i in range(n)]


def _legend(ax: plt.Axes, specs: list[tuple[str, str, bool, str]],
            loc: str, ncol: int = 1, fontsize: float = 7.0) -> None:
    handles = [
        Line2D([], [], color=c, marker=m, linestyle="none",
               markerfacecolor=c if f else "white", markeredgecolor=c,
               markeredgewidth=1.1, markersize=5.0, label=lab)
        for c, m, f, lab in specs
    ]
    ax.legend(handles=handles, frameon=False, loc=loc, ncol=ncol,
              fontsize=fontsize, handlelength=1.0, handletextpad=0.5,
              borderaxespad=0.4)


def _style_duration_axis(ax: plt.Axes, durations: list[float],
                         xpos: np.ndarray) -> None:
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{int(d)} s" for d in durations])
    ax.set_xlim(-0.5, len(durations) - 0.5)
    _strip(ax)


# ---------------------------------------------------------------------------
# E1 — ADMM penalty sweep (ordered horizon ramp, log-log lines)
# ---------------------------------------------------------------------------
def plot_e1(records: list[dict], out: Path) -> None:
    _style()
    by_key: dict[tuple[int, float], list[float]] = {}
    for r in records:
        by_key.setdefault((int(r["horizon"]), float(r["rho"])), []).append(
            float(r["iterations_mean"])
        )
    horizons = sorted({int(r["horizon"]) for r in records})
    rhos = sorted({float(r["rho"]) for r in records})

    fig, ax = plt.subplots(figsize=(5.58, 3.2), constrained_layout=True)
    for h in horizons:
        means = [float(np.mean(by_key[(h, r)])) for r in rhos]
        lows = [float(np.min(by_key[(h, r)])) for r in rhos]
        highs = [float(np.max(by_key[(h, r)])) for r in rhos]
        c = HORIZON[h]
        ax.fill_between(rhos, lows, highs, color=c, alpha=0.15,
                        linewidth=0, zorder=1)
        ax.plot(rhos, means, color=c, linewidth=_LW, marker="o",
                markersize=_MS, markeredgecolor="white",
                markeredgewidth=0.8, zorder=3)
        ax.annotate(f"N = {h}", xy=(rhos[-1], means[-1]), xytext=(6, 0),
                    textcoords="offset points", color=c, fontsize=7.5,
                    va="center")

    # operating point rho = 1 (the middle tested value)
    ax.axvline(1.0, color=MUTED, linewidth=0.7, linestyle=":", zorder=2)
    ax.text(0.985, 0.02, r"operating $\rho$ = 1", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=MUTED)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"ADMM penalty $\rho$")
    ax.set_ylabel("Mean iterations to convergence")
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.set_xlim(rhos[0] * 0.85, rhos[-1] * 1.45)
    _strip(ax)
    _save(fig, out)


# ---------------------------------------------------------------------------
# E2 — iteration budget (tracking error + CPU cost)
# ---------------------------------------------------------------------------
def plot_e2(records: list[dict], out: Path) -> None:
    _style()
    records = sorted(records, key=lambda r: int(r["cap"]))
    caps = [int(r["cap"]) for r in records]
    mean_rmse = [float(r["mean_rmse"]) for r in records]
    terminal = [float(r["terminal_rmse"]) for r in records]
    cpu = [float(r["cpu_seconds"]) for r in records]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.83, 2.6), sharex=True, constrained_layout=True,
    )

    ax_a.plot(caps, mean_rmse, color=IDEAL, linewidth=_LW, marker="o",
              markersize=_MS, markeredgecolor="white", markeredgewidth=0.8,
              label="Mean over 60 steps", zorder=3)
    ax_a.plot(caps, terminal, color=LOCAL, linewidth=_LW, marker="s",
              markersize=_MS, markeredgecolor="white", markeredgewidth=0.8,
              label="Terminal (steady state)", zorder=3)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xticks(caps)
    ax_a.set_xticklabels([f"{c}" for c in caps])
    ax_a.set_xlabel("ADMM iteration cap")
    ax_a.set_ylabel("Tracking error (m)")
    ax_a.legend(frameon=False, loc="center right", fontsize=7, handlelength=1.2)
    _strip(ax_a)
    ax_a.set_title("(a) Tracking error", fontsize=8.5, pad=8)

    ax_b.plot(caps, cpu, color=IDEAL, linewidth=_LW, marker="o", markersize=_MS,
              markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax_b.set_xscale("log")
    ax_b.set_xticks(caps)
    ax_b.set_xticklabels([f"{c}" for c in caps])
    ax_b.set_xlabel("ADMM iteration cap")
    ax_b.set_ylabel("CPU time per closed loop (s)")
    _strip(ax_b)
    ax_b.set_title("(b) Computational cost", fontsize=8.5, pad=8)

    _save(fig, out)


# ---------------------------------------------------------------------------
# E3 / E4 — three-panel denial mechanism figure (dot plots, no polylines)
# ---------------------------------------------------------------------------
def _collect(records: list[dict], policy: str) -> dict[float, dict]:
    return {float(r["duration"]): r for r in records if r["policy"] == policy}


def plot_denial(records: list[dict], out: Path, accelerating: bool) -> None:
    _style()
    durations = sorted({float(r["duration"]) for r in records})
    xpos = np.arange(len(durations))

    pol_a = ["ideal", "local_mpc", "cv_prediction", "zoh"]
    col_a = {"ideal": IDEAL, "local_mpc": LOCAL,
             "cv_prediction": CV, "zoh": ZOH}
    lab_a = {
        "ideal": "Ideal (full comm.)",
        "local_mpc": "Uncoupled local MPC",
        "cv_prediction": "CV (decayed)",
        "zoh": "ZOH (decayed)",
    }

    fig = plt.figure(figsize=(5.83, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # --- (a) confidence-decayed coupling: error saturates --------------------
    off_a = _dodge(len(pol_a), width=0.7)
    for k, p in enumerate(pol_a):
        m = _collect(records, p)
        for xi, d in enumerate(durations):
            if d not in m:
                continue
            y = float(m[d]["denial_peak_rmse"])
            e = float(m[d].get("denial_peak_rmse_std", 0.0))
            _dot(ax_a, xi + off_a[k], y, e, col_a[p], MARKER[p], fill=True)

    ax_a.set_yscale("log")
    ax_a.set_ylabel("Peak formation error (m)")
    ax_a.set_title("(a) Confidence-decayed coupling", fontsize=8.5, pad=8)
    _legend(ax_a, [(col_a[p], MARKER[p], True, lab_a[p]) for p in pol_a],
            loc="upper left")
    _style_duration_axis(ax_a, durations, xpos)
    ax_a.text(0.97, 0.96, "saturates", transform=ax_a.transAxes,
              ha="right", va="top", fontsize=7.5, color=ZOH, style="italic")

    # --- (b) fixed-weight ablation: growth returns ---------------------------
    pol_b = ["cv_prediction", "cv_fixed", "zoh", "zoh_fixed"]
    fill_b = {"cv_prediction": True, "cv_fixed": False,
              "zoh": True, "zoh_fixed": False}
    col_b = {"cv_prediction": CV, "cv_fixed": CV, "zoh": ZOH, "zoh_fixed": ZOH}
    marker_b = {"cv_prediction": "^", "cv_fixed": "^",
                "zoh": "s", "zoh_fixed": "s"}
    off_b = _dodge(4, width=0.7)
    for k, p in enumerate(pol_b):
        m = _collect(records, p)
        for xi, d in enumerate(durations):
            y = float(m[d]["denial_peak_rmse"])
            e = float(m[d].get("denial_peak_rmse_std", 0.0))
            _dot(ax_b, xi + off_b[k], y, e, col_b[p], marker_b[p], fill=fill_b[p])

    ax_b.set_yscale("log")
    ax_b.set_title("(b) Fixed-weight ablation", fontsize=8.5, pad=8)
    _legend(ax_b, [
        (CV, "^", True, "CV (decayed)"),
        (CV, "^", False, "CV (fixed)"),
        (ZOH, "s", True, "ZOH (decayed)"),
        (ZOH, "s", False, "ZOH (fixed)"),
    ], loc="upper left")
    _style_duration_axis(ax_b, durations, xpos)
    ax_b.text(0.97, 0.96, "grows", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=7.5, color=ZOH, style="italic")

    # --- (c) prediction error keeps growing ----------------------------------
    pol_c = ["cv_prediction", "zoh"]
    off_c = _dodge(2, width=0.45)
    for k, p in enumerate(pol_c):
        m = _collect(records, p)
        for xi, d in enumerate(durations):
            y = float(m[d]["prediction_error_peak"])
            e = float(m[d].get("prediction_error_peak_std", 0.0))
            _dot(ax_c, xi + off_c[k], y, e, col_a[p], MARKER[p], fill=True)

    ax_c.set_yscale("log")
    ax_c.set_ylabel(r"Peak prediction error $\|\xi^{k}\|$ (m)")
    ax_c.set_xlabel("Denial duration (s)")
    ax_c.set_title("(c) Prediction error (unbounded)", fontsize=8.5, pad=8)
    _legend(ax_c, [
        (CV, "^", True, "Constant-velocity"),
        (ZOH, "s", True, "Zero-order hold"),
    ], loc="upper left")
    _style_duration_axis(ax_c, durations, xpos)

    _save(fig, out)


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("e1"):
        plot_e1(summary["e1"], OUT / "e1_convergence")
    if summary.get("e2"):
        plot_e2(summary["e2"], OUT / "e2_budget")
    if summary.get("e3"):
        plot_denial(summary["e3"], OUT / "e3_denial", accelerating=False)
    if summary.get("e4"):
        plot_denial(summary["e4"], OUT / "e4_acceleration", accelerating=True)
    print("regenerated:",
          "e1_convergence", "e2_budget", "e3_denial", "e4_acceleration")


if __name__ == "__main__":
    main()
