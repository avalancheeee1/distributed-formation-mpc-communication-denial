"""make_figures_v6.py — journal-grade figures for MDPI Mathematics (2nd pass).

Changes over v5, in response to editorial and author feedback:

* **e3 (denial) — text occlusion fixed.**  The per-panel legends and the
  floating "saturates"/"grows" corner tags (which sat in the top-right where the
  high-error fixed-weight points live) are gone.  One shared figure-level legend
  sits outside the axes, and the mechanism is folded into each panel title, so
  no label ever overlaps a data point.
* **e4 (acceleration) — new expression.**  The old e4 was a structural clone of
  e3 and read as redundant.  It is now a two-panel "stress test": (a) the
  decayed-vs-fixed peak-error curves under acceleration, and (b) a growth-factor
  bar chart (60 s / 10 s) that makes the fixed-weight explosion under
  acceleration (36x for CV, 10x for ZOH) immediately visible against the ~1.3-3x
  that confidence decay keeps.  Filled = decayed, hollow = fixed.
* **e0 (equivalence) — new figure.**  The paper's title claim
  ("Centralized-Equivalence Gating") had no dedicated figure.  A small-multiples
  heatmap of the centralized-vs-distributed objective gap across topology,
  horizon and penalty shows the gate holds uniformly, plus a margin panel that
  places the objective and decision gaps against the 2e-5 acceptance threshold.

Output stems are e1_convergence, e2_budget, e3_denial, e4_acceleration (same as
before, so the manuscript picks them up) plus a new e0_equivalence.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "artifacts_v2_ablation" / "tables" / "summary.json"
RAW = ROOT / "artifacts_v2_ablation" / "raw" / "results.json"
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

# --- typography: uniform letter sizes across figures -------------------------
# Figures 1 and 2 (the edge/agent schematic and the denial flowchart) are
# lettered at 6.3-8.5 pt on a 5.83 in canvas.  Every matplotlib figure here is
# embedded 1:1 at its native figsize width on that same canvas, so pt sizes are
# directly comparable and Figures 1/2 set the standard the other figures must
# match.  Rather than flattening to a single size, the rest of the figures are
# pulled onto the two tiers that Figures 1/2 themselves use -- body 7.5 pt (the
# `_box` / `_diamond` default and the `min f_i` label) and emphasis 8.5 pt (the
# "Agent i" / "Edge e=(i,j)" headings).  Nothing is lettered below 7.5 pt any
# more, where the old spread reached down to 6.0 pt.
FS_EMPHASIS = 8.5   # panel titles, axis labels
FS_BODY = 7.5       # tick labels, legends, annotations, in-plot numerals
FS_SMALL = 7.5      # floor: nothing in a figure may be smaller than this


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "STIXGeneral",
                           "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": FS_BODY,
            "axes.labelsize": FS_EMPHASIS,
            "axes.titlesize": FS_EMPHASIS,
            "xtick.labelsize": FS_BODY,
            "ytick.labelsize": FS_BODY,
            "legend.fontsize": FS_BODY,
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


def _handle(color: str, marker: str, fill: bool, label: str) -> Line2D:
    return Line2D(
        [], [], color=color, marker=marker, linestyle="none",
        markerfacecolor=color if fill else "white", markeredgecolor=color,
        markeredgewidth=1.1, markersize=5.0, label=label,
    )


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
                    textcoords="offset points", color=c, fontsize=FS_BODY,
                    va="center")

    ax.axvline(1.0, color=MUTED, linewidth=0.7, linestyle=":", zorder=2)
    ax.text(0.985, 0.02, r"operating $\rho$ = 1", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_BODY, color=MUTED)

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
    ax_a.legend(frameon=False, loc="center right", fontsize=FS_BODY, handlelength=1.2)
    _strip(ax_a)
    ax_a.set_title("(a) Tracking error", fontsize=FS_EMPHASIS, pad=8)

    ax_b.plot(caps, cpu, color=IDEAL, linewidth=_LW, marker="o", markersize=_MS,
              markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax_b.set_xscale("log")
    ax_b.set_xticks(caps)
    ax_b.set_xticklabels([f"{c}" for c in caps])
    ax_b.set_xlabel("ADMM iteration cap")
    ax_b.set_ylabel("CPU time per closed loop (s)")
    _strip(ax_b)
    ax_b.set_title("(b) Computational cost", fontsize=FS_EMPHASIS, pad=8)

    _save(fig, out)


# ---------------------------------------------------------------------------
# E3 — three-panel denial mechanism (single shared legend, no occlusion)
# ---------------------------------------------------------------------------
def _collect(records: list[dict], policy: str) -> dict[float, dict]:
    return {float(r["duration"]): r for r in records if r["policy"] == policy}


def plot_e3(records: list[dict], out: Path) -> None:
    _style()
    durations = sorted({float(r["duration"]) for r in records})
    xpos = np.arange(len(durations))

    pol_a = ["ideal", "local_mpc", "cv_prediction", "zoh"]
    col_a = {"ideal": IDEAL, "local_mpc": LOCAL,
             "cv_prediction": CV, "zoh": ZOH}

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
    ax_a.set_title("(a) Decayed coupling — error saturates", fontsize=FS_EMPHASIS, pad=8)
    _style_duration_axis(ax_a, durations, xpos)

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
    ax_b.set_title("(b) Fixed-weight ablation — error grows", fontsize=FS_EMPHASIS, pad=8)
    _style_duration_axis(ax_b, durations, xpos)

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
    ax_c.set_title(r"(c) Prediction error $\|\xi^{k}\|$ keeps growing",
                   fontsize=FS_EMPHASIS, pad=8)
    _style_duration_axis(ax_c, durations, xpos)

    # --- one shared legend outside the axes (no per-panel occlusion) ---------
    handles = [
        _handle(IDEAL, "o", True, "Ideal (full comm.)"),
        _handle(LOCAL, "D", True, "Uncoupled local MPC"),
        _handle(CV, "^", True, "CV (decayed)"),
        _handle(CV, "^", False, "CV (fixed)"),
        _handle(ZOH, "s", True, "ZOH (decayed)"),
        _handle(ZOH, "s", False, "ZOH (fixed)"),
    ]
    # Six entries on one row no longer fit once the lettering is at the
    # Figures-1/2 body size: at 7.5 pt the sixth ("ZOH (fixed)") ran past the
    # right edge of the 5.83 in canvas and was clipped.  Two rows of three.
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, fontsize=FS_BODY, handletextpad=0.4,
               columnspacing=1.4, borderaxespad=0.3)

    _save(fig, out)


# ---------------------------------------------------------------------------
# E4 — acceleration stress test (decayed-vs-fixed curves + growth-factor bars)
# ---------------------------------------------------------------------------
def _growth(records: list[dict], policy: str, d1: float = 10.0,
            d2: float = 60.0) -> float:
    m = _collect(records, policy)
    return float(m[d2]["denial_peak_rmse"]) / float(m[d1]["denial_peak_rmse"])


def plot_e4(records: list[dict], out: Path) -> None:
    _style()
    durations = sorted({float(r["duration"]) for r in records})
    xpos = np.arange(len(durations))

    pol = ["cv_prediction", "cv_fixed", "zoh", "zoh_fixed"]
    fill = {"cv_prediction": True, "cv_fixed": False, "zoh": True, "zoh_fixed": False}
    col = {"cv_prediction": CV, "cv_fixed": CV, "zoh": ZOH, "zoh_fixed": ZOH}
    marker = {"cv_prediction": "^", "cv_fixed": "^", "zoh": "s", "zoh_fixed": "s"}
    lab = {"cv_prediction": "CV (decayed)", "cv_fixed": "CV (fixed)",
           "zoh": "ZOH (decayed)", "zoh_fixed": "ZOH (fixed)"}

    fig = plt.figure(figsize=(5.83, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # --- (a) decayed saturates, fixed explodes under acceleration ------------
    off = _dodge(4, width=0.7)
    for k, p in enumerate(pol):
        m = _collect(records, p)
        for xi, d in enumerate(durations):
            y = float(m[d]["denial_peak_rmse"])
            e = float(m[d].get("denial_peak_rmse_std", 0.0))
            _dot(ax_a, xi + off[k], y, e, col[p], marker[p], fill=fill[p])

    ax_a.set_yscale("log")
    ax_a.set_ylabel("Peak formation error (m)")
    ax_a.set_title("(a) Peak error under an accelerating reference",
                   fontsize=FS_EMPHASIS, pad=8)
    _style_duration_axis(ax_a, durations, xpos)

    # --- (b) growth factor 60 s / 10 s ----------------------------------------
    # One line per group, not two.  As two lines the labels were already
    # abutting at the old 7 pt; at the Figures-1/2 body size the descender of
    # "(nominal)" collided with the cap-height of the next "CV", so the block of
    # four groups read as six separate labels.  Same words as the caption
    # ("the nominal and accelerating scenarios"), one line each.
    groups = ["CV (nominal)", "CV (accelerating)",
              "ZOH (nominal)", "ZOH (accelerating)"]
    # decayed vs fixed, per scenario, per predictor (nominal from e3, accel here)
    nominal = json.loads(SUMMARY.read_text(encoding="utf-8"))["e3"]
    accel = records  # this is e4
    growth = [
        (_growth(nominal, "cv_prediction"), _growth(nominal, "cv_fixed")),
        (_growth(accel, "cv_prediction"), _growth(accel, "cv_fixed")),
        (_growth(nominal, "zoh"), _growth(nominal, "zoh_fixed")),
        (_growth(accel, "zoh"), _growth(accel, "zoh_fixed")),
    ]
    y = np.arange(len(groups))[::-1]
    h = 0.34
    for i, (g_dec, g_fix) in enumerate(growth):
        ax_b.barh(y[i] + h / 2, g_dec, height=h, color=CV if i < 2 else ZOH,
                  edgecolor="white", linewidth=0.6, zorder=3)
        ax_b.barh(y[i] - h / 2, g_fix, height=h,
                  color="white", edgecolor=CV if i < 2 else ZOH,
                  linewidth=1.2, zorder=3)
        ax_b.text(g_dec, y[i] + h / 2, f" {g_dec:.1f}$\\times$", va="center",
                  ha="left", fontsize=FS_BODY, color=INK)
        ax_b.text(g_fix, y[i] - h / 2, f" {g_fix:.0f}$\\times$", va="center",
                  ha="left", fontsize=FS_BODY, color=INK)

    ax_b.set_xscale("log")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(groups, fontsize=FS_BODY)
    ax_b.set_xlabel("Growth in peak error, 10 s to 60 s ($\\times$)")
    ax_b.set_title("(b) Fixed weight diverges, decay stays bounded", fontsize=FS_EMPHASIS, pad=8)
    ax_b.axvline(1.0, color=MUTED, linewidth=0.7, linestyle=":", zorder=2)
    ax_b.set_axisbelow(True)
    ax_b.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax_b.tick_params(length=2.5, colors=MUTED)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_xlim(0.8, 90)

    # --- shared legend --------------------------------------------------------
    handles = [_handle(CV, "^", True, "CV (decayed)"), _handle(CV, "^", False, "CV (fixed)"),
               _handle(ZOH, "s", True, "ZOH (decayed)"), _handle(ZOH, "s", False, "ZOH (fixed)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4,
               frameon=False, fontsize=FS_BODY, handletextpad=0.4,
               columnspacing=0.9, borderaxespad=0.3)

    _save(fig, out)


# ---------------------------------------------------------------------------
# E0 — centralized-equivalence gate (heatmap small multiples + margin panel)
# ---------------------------------------------------------------------------
def plot_equivalence(out: Path) -> None:
    _style()
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    e1 = raw["e1"]
    e0 = raw["e0"]
    topologies = ["chain", "ring", "connected_random"]
    horizons = [5, 10, 20]
    rhos = [0.1, 1.0, 10.0]

    # max objective gap per (topology, horizon, rho)
    gaps = np.empty((len(topologies), len(horizons), len(rhos)))
    for ti, t in enumerate(topologies):
        for hi, h in enumerate(horizons):
            for ri, r in enumerate(rhos):
                vals = [x["objective_gap"] for x in e1
                        if x["topology"] == t and int(x["horizon"]) == h
                        and float(x["rho"]) == r]
                gaps[ti, hi, ri] = max(vals)

    fig = plt.figure(figsize=(5.83, 3.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.72])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_m = fig.add_subplot(gs[1, :])

    vmin, vmax = 1e-14, 1e-10
    cmap = plt.get_cmap("viridis").copy()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    for ti, t in enumerate(topologies):
        ax = axs[ti]
        im = ax.imshow(gaps[ti], cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(len(rhos)))
        ax.set_xticklabels([f"{r:g}" for r in rhos], fontsize=FS_BODY)
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels([f"N={h}" for h in horizons], fontsize=FS_BODY)
        ax.set_title({"chain": "Chain", "ring": "Ring",
                      "connected_random": "Connected random"}[t],
                     fontsize=FS_BODY, pad=4)
        for hi in range(len(horizons)):
            for ri in range(len(rhos)):
                v = gaps[ti, hi, ri]
                ax.text(ri, hi, f"{v:.0e}", ha="center", va="center",
                        fontsize=FS_BODY,
                        color="white" if v > 3e-12 else "#1a1a1a")
        ax.tick_params(length=1.5)
    axs[0].set_ylabel("Horizon")
    axs[1].set_xlabel("ADMM penalty $\\rho$")

    cbar = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=FS_BODY, length=1.5)
    cbar.set_label("Objective gap (m)", fontsize=FS_BODY)

    # --- margin panel: e0 correctness gate vs tolerance ----------------------
    og = np.array([x["objective_gap"] for x in e0])
    dg = np.array([x["decision_gap"] for x in e0])
    tol = 2e-5
    # Row order matters: `og` is drawn at y=0 and `dg` at y=1, so the tick
    # labels must list the objective gap first.  They were previously reversed,
    # which put the blue objective-gap points on a row reading "Decision gap"
    # (the blue and orange colours were unexplained -- and an unswapped colour
    # key would have contradicted the axis).
    cats = ["Objective gap\n$J(x)-J^\\star$", "Decision gap\n$\\|x-x^\\star\\|$"]
    # Distinct marker shapes as well as distinct hues, so the key stays readable
    # in greyscale and does not rely on colour alone.
    ax_m.scatter(og, np.full_like(og, 0), color=IDEAL, marker="o", s=14,
                 alpha=0.8, edgecolor="white", linewidth=0.4, zorder=3,
                 label="Objective gap $J(x)-J^\\star$")
    ax_m.scatter(dg, np.full_like(dg, 1), color=ZOH, marker="s", s=14,
                 alpha=0.8, edgecolor="white", linewidth=0.4, zorder=3,
                 label="Decision gap $\\|x-x^\\star\\|$")
    ax_m.axvline(tol, color=MUTED, linestyle="--", linewidth=0.9, zorder=2)
    ax_m.text(tol, 1.42, "gate $2\\times10^{-5}$", ha="center", fontsize=FS_BODY,
              color=MUTED)
    ax_m.set_xscale("log")
    ax_m.set_yticks([0, 1])
    ax_m.set_yticklabels(cats, fontsize=FS_BODY)
    ax_m.set_xlim(1e-16, 1e-3)
    ax_m.set_ylim(-0.5, 1.7)
    ax_m.set_xlabel("Gap (log scale)")
    ax_m.set_title("Correctness gate: 30 cases pass by 2–9 orders of magnitude",
                   fontsize=FS_BODY, pad=4)
    # Explicit colour key.  Placed in the empty upper-left
    # corner: the two data rows sit at y=0 and y=1 and the gate annotation sits
    # at x=2e-5, so nothing is occluded.
    key_handles = [
        Line2D([], [], color=IDEAL, marker="o", linestyle="none", markersize=4.0,
               markeredgecolor="white", markeredgewidth=0.4,
               label="Objective gap $J(x)-J^\\star$"),
        Line2D([], [], color=ZOH, marker="s", linestyle="none", markersize=4.0,
               markeredgecolor="white", markeredgewidth=0.4,
               label="Decision gap $\\|x-x^\\star\\|$"),
    ]
    ax_m.legend(handles=key_handles, loc="upper left", frameon=False,
                fontsize=FS_BODY, handletextpad=0.5, borderaxespad=0.3,
                labelspacing=0.35)
    ax_m.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax_m.set_axisbelow(True)
    ax_m.tick_params(length=2.0, colors=MUTED)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)

    _save(fig, out)


def _box(ax, x, y, w, h, text, *, fc, ec, fs=7.5, lw=1.1, bold=False,
         align="center") -> None:
    """Draw a rounded text box centred at (x, y)."""
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3,
            color=INK, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color=INK, lw=1.0, style="-|>", ms=10,
           rad=0.0, ls="-") -> None:
    a = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, zorder=3,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls,
    )
    ax.add_patch(a)


def _diamond(ax, x, y, w, h, text, *, fc, ec, fs=7.5) -> None:
    d = Polygon(
        [[x, y + h / 2], [x + w / 2, y], [x, y - h / 2], [x - w / 2, y]],
        closed=True, linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(d)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            color=INK, linespacing=1.3)


# ---------------------------------------------------------------------------
# Method figure 1 — edge-splitting ADMM schematic (one edge, two agents)
# ---------------------------------------------------------------------------
def plot_admm_split(out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.83, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    agent_fc = (0.0, 0.447, 0.698, 0.10)     # IDEAL blue, light
    agent_ec = IDEAL
    edge_fc = (0.0, 0.620, 0.451, 0.12)      # CV green, light
    edge_ec = CV
    dual_c = ZOH

    # --- agent boxes --------------------------------------------------------
    for x, name in ((1.7, "Agent $i$"), (8.3, "Agent $j$")):
        _box(ax, x, 3.35, 2.6, 2.15, "", fc=agent_fc, ec=agent_ec, lw=1.2)
        ax.text(x, 4.05, name, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=INK)
        ax.text(x, 3.55, r"$\min\ f_i(x_i,u_i)$", ha="center", va="center",
                fontsize=7.5, color=INK)
        ax.text(x, 3.15, r"s.t. dynamics, $|u_i|\leq\bar{u}$", ha="center",
                va="center", fontsize=6.8, color=MUTED)
        # highlighted output sub-box
        op = FancyBboxPatch((x - 1.05, 2.35), 2.1, 0.62,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            linewidth=0.9, edgecolor=dual_c, facecolor="white",
                            zorder=3)
        ax.add_patch(op)
        ax.text(x, 2.66, r"$s_{e,%s}=C x_{%s}-d_{%s}$"
                % ("i" if x < 5 else "j", "i" if x < 5 else "j",
                   "i" if x < 5 else "j"),
                ha="center", va="center", fontsize=7.2, color=INK)

    # --- edge box -----------------------------------------------------------
    ex, ey = 5.0, 3.35
    _box(ax, ex, ey, 3.0, 2.15, "", fc=edge_fc, ec=edge_ec, lw=1.2)
    ax.text(ex, 4.05, "Edge $e=(i,j)$", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=INK)
    # two copy sub-boxes
    for xoff, lab in ((-0.72, "$z_{e,i}$"), (0.72, "$z_{e,j}$")):
        sb = FancyBboxPatch((ex + xoff - 0.55, ey - 0.15), 1.1, 0.62,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            linewidth=0.9, edgecolor=edge_ec, facecolor="white",
                            zorder=3)
        ax.add_patch(sb)
        ax.text(ex + xoff, ey + 0.16, lab, ha="center", va="center",
                fontsize=7.6, color=INK)
    ax.text(ex, ey - 0.95, r"couple\ $\gamma\,w_e\,\|z_{e,i}-z_{e,j}\|_W^2$",
            ha="center", va="center", fontsize=7.2, color=edge_ec)
    # coupling spring between copies
    _arrow(ax, ex - 0.17, ey + 0.16, ex + 0.17, ey + 0.16, color=edge_ec,
           lw=1.2, style="<|-|>", ms=9)

    # --- consensus arrows (s -> z), with scaled dual ------------------------
    for sgn in (-1, 1):
        xa = 3.0 if sgn < 0 else 7.0
        xz = ex + sgn * 0.72
        _arrow(ax, xa, 2.66, xz - sgn * 0.03, 3.19, color=dual_c, lw=1.2)
        ax.text((xa + xz) / 2 + sgn * 0.02, 3.62, "consensus", ha="center",
                va="bottom", fontsize=6.3, color=MUTED)
        ax.text((xa + xz) / 2 + sgn * 0.02, 2.28, r"$\eta_{e,%s}$"
                % ("i" if sgn < 0 else "j"), ha="center", va="top",
                fontsize=7.0, color=dual_c)

    # bottom annotation
    ax.text(5.0, 0.35,
            "each endpoint output is duplicated at the edge; the scaled dual "
            r"$\eta_e$ enforces $s_{e,\cdot}=z_{e,\cdot}$, and only $z_e$ is "
            "coupled",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Method figure 2 — confidence-decay denial protocol flowchart
# ---------------------------------------------------------------------------
def plot_protocol(out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.83, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.4)
    ax.axis("off")

    box_fc = (0.0, 0.447, 0.698, 0.08)
    box_ec = IDEAL
    no_fc = (0.835, 0.369, 0.0, 0.10)     # ZOH orange, light
    no_ec = ZOH

    # start
    _box(ax, 5.0, 10.5, 3.6, 0.95, "MPC step $k$:\ntry to receive neighbor "
         "messages", fc=box_fc, ec=box_ec, fs=7.2)
    # decision
    _diamond(ax, 5.0, 8.8, 2.3, 1.35, "communication\nactive?", fc="white",
             ec=INK, fs=7.0)
    _arrow(ax, 5.0, 10.02, 5.0, 9.48)

    # yes branch (left)
    _box(ax, 2.3, 7.6, 2.7, 0.95, "exact edge update\n(eq.~5)", fc=box_fc,
         ec=box_ec, fs=7.2)
    _arrow(ax, 4.35, 8.35, 2.6, 8.05, color=CV, lw=1.2)
    ax.text(3.2, 8.52, "yes", ha="center", va="bottom", fontsize=6.5,
            color=CV, style="italic")

    # no branch (right)
    _box(ax, 7.7, 7.6, 3.2, 0.95, "predict remote output\n"
         r"$\hat{s}$ (ZOH / CV)", fc=no_fc, ec=no_ec, fs=7.2)
    _arrow(ax, 5.65, 8.35, 7.4, 8.05, color=no_ec, lw=1.2)
    ax.text(6.6, 8.52, "no", ha="center", va="bottom", fontsize=6.5,
            color=no_ec, style="italic")
    _box(ax, 7.7, 6.2, 3.2, 0.95, "compute age $\\tau$ of\nlast message",
         fc=no_fc, ec=no_ec, fs=7.2)
    _arrow(ax, 7.7, 7.12, 7.7, 6.68, color=no_ec, lw=1.1)
    _box(ax, 7.7, 4.8, 3.2, 0.95, "decay weight\n"
         r"$w\,e^{-\tau/T_{\mathrm{w}}}$", fc=no_fc, ec=no_ec, fs=7.2)
    _arrow(ax, 7.7, 5.72, 7.7, 5.28, color=no_ec, lw=1.1)
    _box(ax, 7.7, 3.4, 3.2, 0.95, "inexact (half) edge\nupdate", fc=no_fc,
         ec=no_ec, fs=7.2)
    _arrow(ax, 7.7, 4.32, 7.7, 3.88, color=no_ec, lw=1.1)

    # merge
    _box(ax, 5.0, 1.9, 4.4, 1.05, "solve local QP $+$ dual update\n"
         "apply first control $u_0$, shift horizon", fc="white", ec=INK,
         fs=7.2)
    _arrow(ax, 2.3, 7.12, 3.6, 2.42, color=CV, lw=1.1, rad=-0.12)
    _arrow(ax, 7.7, 2.92, 6.6, 2.42, color=no_ec, lw=1.1, rad=0.12)
    # loop back
    _arrow(ax, 2.8, 1.9, 0.4, 1.9, color=INK, lw=1.0, rad=0.0)
    _arrow(ax, 0.4, 1.9, 0.4, 10.5, color=INK, lw=1.0)
    _arrow(ax, 0.4, 10.5, 3.2, 10.5, color=INK, lw=1.0, ms=10)
    ax.text(0.28, 6.2, "repeat", ha="center", va="center", fontsize=6.3,
            color=MUTED, rotation=90)

    _save(fig, out)


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("e1"):
        plot_e1(summary["e1"], OUT / "e1_convergence")
    if summary.get("e2"):
        plot_e2(summary["e2"], OUT / "e2_budget")
    if summary.get("e3"):
        plot_e3(summary["e3"], OUT / "e3_denial")
    if summary.get("e4"):
        plot_e4(summary["e4"], OUT / "e4_acceleration")
    plot_equivalence(OUT / "e0_equivalence")
    plot_admm_split(OUT / "e5_admm_split")
    plot_protocol(OUT / "e6_protocol")
    print("regenerated:",
          "e1_convergence", "e2_budget", "e3_denial", "e4_acceleration",
          "e0_equivalence", "e5_admm_split", "e6_protocol")


if __name__ == "__main__":
    main()
