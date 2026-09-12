"""Enhanced publication figures for the v2 manuscript (v3 redesign).

The v3 figures tighten the visual argument of the paper's core mechanism:

* ``e1_convergence``  — penalty sweep (unchanged skeleton, clearer operating band).
* ``e2_budget``       — iteration-cap sweep (unchanged skeleton, stronger
  "orders of magnitude" bracket).
* ``e3_denial`` / ``e4_acceleration`` — **three-panel redesign** that makes the
  saturation signature visible: (a) formation error under confidence-decayed
  coupling (saturates), (b) fixed-weight ablation (growth returns), (c) the
  prediction error that *keeps growing* throughout.  Panels (a)+(c) juxtaposed
  are the paper's central claim — unbounded disturbance, bounded formation error.

The palette is the Okabe--Ito colorblind-safe set; markers carry shape as a
redundant channel so no comparison relies on colour alone.  Same output stems as
the v2 figures so the manuscript pipeline picks the new images up unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "artifacts_v2_ablation" / "tables" / "summary.json"
OUT = ROOT / "artifacts_v2_ablation" / "tables"

# --- Okabe--Ito colourblind-safe palette -------------------------------------
BLUE = "#0072B2"       # ideal / full communication
ORANGE = "#E69F00"     # uncoupled local MPC
GREEN = "#009E73"      # constant-velocity prediction
VERMILLION = "#D55E00"  # zero-order hold
SKY = "#56B4E9"        # (spare)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e3e3e0"
AXIS = "#c8c8c4"

_LW = 2.0
_MS = 6.5


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def _strip(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def _errorbar(ax, xs, ys, errs, color, marker, linestyle="-", label=None):
    ax.errorbar(
        xs,
        ys,
        yerr=errs,
        color=color,
        linestyle=linestyle,
        linewidth=_LW,
        marker=marker,
        markersize=_MS,
        markeredgecolor="white",
        markeredgewidth=1.1,
        capsize=3,
        capthick=1.1,
        elinewidth=1.1,
        label=label,
        zorder=3,
    )


# ---------------------------------------------------------------------------
# E1 — penalty sweep
# ---------------------------------------------------------------------------
def plot_e1(records: list[dict], out: Path) -> None:
    _apply_style()
    by_key: dict[tuple[int, float], list[float]] = {}
    for r in records:
        by_key.setdefault((int(r["horizon"]), float(r["rho"])), []).append(
            float(r["iterations_mean"])
        )
    horizons = sorted({int(r["horizon"]) for r in records})
    rhos = sorted({float(r["rho"]) for r in records})
    colors = {5: BLUE, 10: ORANGE, 20: GREEN}

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for h in horizons:
        means = [float(np.mean(by_key[(h, r)])) for r in rhos]
        lows = [float(np.min(by_key[(h, r)])) for r in rhos]
        highs = [float(np.max(by_key[(h, r)])) for r in rhos]
        c = colors[h]
        ax.fill_between(rhos, lows, highs, color=c, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(
            rhos, means, color=c, linewidth=_LW, marker="o", markersize=_MS,
            markeredgecolor="white", markeredgewidth=1.1, zorder=3,
        )
        ax.annotate(
            f"N = {h}", xy=(rhos[-1], means[-1]), xytext=(8, 0),
            textcoords="offset points", color=c, fontsize=10,
            fontweight="bold", va="center",
        )

    # operating regime (rho = 1)
    ax.axvspan(0.62, 1.55, color=INK, alpha=0.06, linewidth=0, zorder=0)
    ax.annotate(
        "operating $\\rho$",
        xy=(1.0, 30), xytext=(0.35, 14), color=INK, fontsize=9,
        ha="center",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ADMM penalty $\\rho$")
    ax.set_ylabel("Mean iterations to convergence")
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.set_xlim(rhos[0] * 0.9, rhos[-1] * 1.6)
    _strip(ax)
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# E2 — iteration budget
# ---------------------------------------------------------------------------
def plot_e2(records: list[dict], out: Path) -> None:
    _apply_style()
    records = sorted(records, key=lambda r: int(r["cap"]))
    caps = [int(r["cap"]) for r in records]
    cpu = [float(r["cpu_seconds"]) for r in records]
    mean_rmse = [float(r["mean_rmse"]) for r in records]
    terminal = [float(r["terminal_rmse"]) for r in records]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 3.1), sharex=True)

    ax_a.plot(caps, mean_rmse, color=BLUE, linewidth=_LW, marker="o",
              markersize=_MS, markeredgecolor="white", markeredgewidth=1.1,
              label="Mean over 60 steps", zorder=3)
    ax_a.plot(caps, terminal, color=ORANGE, linewidth=_LW, marker="s",
              markersize=_MS, markeredgecolor="white", markeredgewidth=1.1,
              label="Terminal (steady state)", zorder=3)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xticks(caps)
    ax_a.set_xticklabels([f"{c}" for c in caps])
    ax_a.set_xlabel("ADMM iteration cap")
    ax_a.set_ylabel("Tracking error (m)")
    ax_a.legend(frameon=False, loc="center right", handlelength=1.6)
    _strip(ax_a)
    ax_a.set_title("(a) Tracking error", fontsize=10, pad=6)
    i20 = caps.index(20)
    ax_a.annotate(
        "≈4 orders of magnitude",
        xy=(20, terminal[i20]), xytext=(1, 2e-5),
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
        color=INK, fontsize=8.5, va="center", ha="left",
    )

    ax_b.plot(caps, cpu, color=BLUE, linewidth=_LW, marker="o", markersize=_MS,
              markeredgecolor="white", markeredgewidth=1.1, zorder=3)
    ax_b.set_xscale("log")
    ax_b.set_xticks(caps)
    ax_b.set_xticklabels([f"{c}" for c in caps])
    ax_b.set_xlabel("ADMM iteration cap")
    ax_b.set_ylabel("CPU time per closed loop (s)")
    _strip(ax_b)
    ax_b.set_title("(b) Computational cost", fontsize=10, pad=6)

    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# E3 / E4 — three-panel denial mechanism figure
# ---------------------------------------------------------------------------
def _collect(records: list[dict], policy: str) -> dict[float, dict]:
    sub = [r for r in records if r["policy"] == policy]
    return {float(r["duration"]): r for r in sub}


def plot_denial(records: list[dict], out: Path, accelerating: bool) -> None:
    _apply_style()
    durations = sorted({float(r["duration"]) for r in records})

    def series(policy, field):
        m = _collect(records, policy)
        xs = [d for d in durations if d in m]
        ys = [float(m[d][field]) for d in xs]
        errs = [float(m[d][field + "_std"]) if field + "_std" in m[d] else 0.0
                for d in xs]
        return xs, ys, errs

    fig = plt.figure(figsize=(6.9, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # --- (a) confidence-decayed coupling: formation error saturates ----------
    _errorbar(ax_a, *series("ideal", "denial_peak_rmse"), BLUE, "o",
              label="Ideal (full comm.)")
    _errorbar(ax_a, *series("local_mpc", "denial_peak_rmse"), ORANGE, "s",
              label="Uncoupled local MPC")
    _errorbar(ax_a, *series("cv_prediction", "denial_peak_rmse"), GREEN, "^",
              label="CV (decayed)")
    _errorbar(ax_a, *series("zoh", "denial_peak_rmse"), VERMILLION, "D",
              label="ZOH (decayed)")

    zoh = _collect(records, "zoh")
    zxs = [d for d in durations if d in zoh]
    zys = [float(zoh[d]["denial_peak_rmse"]) for d in zxs]
    # saturation plateau: 30 s and 60 s coincide
    ax_a.axhspan(zys[-1] * 0.75, zys[-1] * 1.25, xmin=0.42, xmax=1.0,
                 color=VERMILLION, alpha=0.08, linewidth=0, zorder=0)
    ax_a.annotate(
        "saturates", xy=(zxs[-1], zys[-1]),
        xytext=(zxs[-1] - 4.0, zys[-1] * 2.6),
        arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=1.0),
        color=VERMILLION, fontsize=8.5, va="center", ha="right",
    )

    ax_a.set_yscale("log")
    ax_a.set_ylabel("Peak formation error (m)")
    ax_a.set_title("(a) Confidence-decayed coupling", fontsize=9.5, pad=6)
    ax_a.legend(frameon=False, loc="upper left", handlelength=1.5,
                borderaxespad=0.4, fontsize=7.6)
    _style_duration_axis(ax_a, durations)

    # --- (b) fixed-weight ablation: growth returns ---------------------------
    _errorbar(ax_b, *series("cv_prediction", "denial_peak_rmse"), GREEN, "^",
              label="CV (decayed)")
    _errorbar(ax_b, *series("cv_fixed", "denial_peak_rmse"), GREEN, "^",
              linestyle="--", label="CV (fixed weight)")
    _errorbar(ax_b, *series("zoh", "denial_peak_rmse"), VERMILLION, "D",
              label="ZOH (decayed)")
    _errorbar(ax_b, *series("zoh_fixed", "denial_peak_rmse"), VERMILLION, "D",
              linestyle="--", label="ZOH (fixed weight)")

    zf = _collect(records, "zoh_fixed")
    zfxs = [d for d in durations if d in zf]
    zfys = [float(zf[d]["denial_peak_rmse"]) for d in zfxs]
    ax_b.annotate(
        "grows", xy=(zfxs[-1], zfys[-1]),
        xytext=(zfxs[-1] - 6.0, zfys[-1] * 0.5),
        arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=1.0),
        color=VERMILLION, fontsize=8.5, va="center", ha="right",
    )

    ax_b.set_yscale("log")
    ax_b.set_title("(b) Fixed-weight ablation", fontsize=9.5, pad=6)
    ax_b.legend(frameon=False, loc="upper left", handlelength=1.5,
                borderaxespad=0.4, fontsize=7.6)
    _style_duration_axis(ax_b, durations)

    # --- (c) prediction error keeps growing ----------------------------------
    _errorbar(ax_c, *series("cv_prediction", "prediction_error_peak"), GREEN, "^",
              label="Constant-velocity")
    _errorbar(ax_c, *series("zoh", "prediction_error_peak"), VERMILLION, "D",
              label="Zero-order hold")

    # growth ratio annotation (ZOH)
    zpe = [float(zoh[d]["prediction_error_peak"]) for d in zxs]
    if accelerating:
        note = ("prediction error grows "
                f"({zpe[0]:.1f} → {zpe[-1]:.0f} m)\n"
                "while formation error (a) saturates")
    else:
        note = (f"prediction error grows ≈{zpe[-1] / zpe[0]:.0f}× "
                f"({zpe[0]:.1f} → {zpe[-1]:.0f} m)\n"
                "while formation error (a) saturates")
    ax_c.text(
        0.02, 0.72, note, transform=ax_c.transAxes, fontsize=8.2, color=INK,
        va="top", ha="left",
    )

    ax_c.set_yscale("log")
    ax_c.set_ylabel("Peak prediction error $\\|\\xi^{k}\\|$ (m)")
    ax_c.set_title("(c) Prediction error (unbounded)", fontsize=9.5, pad=6)
    ax_c.legend(frameon=False, loc="lower right", handlelength=1.5,
                fontsize=8.0)
    _style_duration_axis(ax_c, durations)
    ax_c.set_xlabel("Denial duration (s)")

    _save(fig, out)


def _style_duration_axis(ax: plt.Axes, durations: list[float]) -> None:
    ax.set_xticks(durations)
    ax.set_xticklabels([f"{int(d):g}" for d in durations])
    ax.set_xlim(durations[0] - 1.5, durations[-1] + 6.0)
    _strip(ax)


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
