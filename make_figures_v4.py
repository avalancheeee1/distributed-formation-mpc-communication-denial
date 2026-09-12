"""make_figures_v4.py — high-end editorial redesign of the publication figures.

Design goals (vs v3):

* **Dot plots instead of connecting polylines** for the categorical denial
  durations (10/30/60 s).  Three points joined by a line imply a false
  continuous interpolation; points + error bars are the honest encoding, and
  it is the classic "Cleveland dot plot" look of high-end methods figures.
* **Deeper, muted Okabe--Ito hues** — the same hue family as v3 (so the
  semantic mapping of ideal/local/CV/ZOH is preserved), pulled down in
  saturation for an editorial, not "MATLAB default", feel.  Marker shape stays
  a redundant channel so no comparison relies on colour alone.
* **Hollow markers for the fixed-weight ablation** — filled = confidence-decayed,
  hollow = fixed weight — so the ablation is one visual channel (fill), not a
  new colour.  This is the clearest possible encoding of "the decay is doing
  the work".
* **Sequential single-hue palette for the ordered horizon sweep (e1)** — an
  ordered quantity (N = 5/10/20) gets a light-to-dark ramp, not three
  unrelated hues.
* Reduced chartjunk: no top/right spines, faint gridlines, tightened spacing.

Same output stems as v2/v3 so the manuscript pipeline picks these up unchanged.
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

# --- refined editorial palette (deepened Okabe--Ito hues) --------------------
IDEAL = "#00527E"   # deep blue      — ideal / full communication
LOCAL = "#B87400"   # deep amber     — uncoupled local MPC
CV = "#00715B"      # deep green     — constant-velocity prediction
ZOH = "#A34700"     # deep rust      — zero-order hold

INK = "#1F2733"
MUTED = "#5A6572"
GRID = "#E7EAED"
AXIS = "#C4C9D0"

# sequential blue for the ordered horizon sweep (e1)
HORIZON = {5: "#8FB2D1", 10: "#3E7CB1", 20: "#1F4E79"}

# redundant marker channel
MARKER = {"ideal": "o", "local_mpc": "D", "cv_prediction": "^", "zoh": "s"}

_MS = 6.5
_LW = 1.9


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "axes.edgecolor": AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
        }
    )


def _strip(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, colors=MUTED)


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def _dodge(n: int, width: float = 0.72) -> list[float]:
    """Offsets centred on 0 so n dots fan out around each categorical x."""
    if n <= 1:
        return [0.0]
    return [width * (i - (n - 1) / 2) / n for i in range(n)]


def _dot(
    ax: plt.Axes,
    x: float,
    y: float,
    err: float,
    color: str,
    marker: str,
    fill: bool = True,
    ms: float = _MS,
) -> None:
    """A single point + vertical error bar, no connecting line."""
    ax.errorbar(
        [x],
        [y],
        yerr=[err],
        fmt=marker,
        linestyle="none",
        color=color,
        markerfacecolor=color if fill else "white",
        markeredgecolor=color,
        markeredgewidth=1.3,
        markersize=ms,
        ecolor=color,
        elinewidth=1.1,
        capsize=3.5,
        capthick=1.1,
        zorder=3,
    )


def _proxy(
    ax: plt.Axes,
    specs: list[tuple[str, str, bool, str]],
    loc: str,
    ncol: int = 1,
    fontsize: float = 8.0,
    **kw,
) -> None:
    """Legend from (color, marker, fill, label) tuples."""
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="none",
            markerfacecolor=color if fill else "white",
            markeredgecolor=color,
            markeredgewidth=1.3,
            markersize=6,
            label=label,
        )
        for color, marker, fill, label in specs
    ]
    ax.legend(handles=handles, frameon=False, loc=loc, ncol=ncol,
              fontsize=fontsize, handlelength=1.2, borderaxespad=0.4, **kw)


# ---------------------------------------------------------------------------
# E1 — penalty sweep (ordered horizon ramp)
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

    fig, ax = plt.subplots(figsize=(5.6, 3.5), constrained_layout=True)
    for h in horizons:
        means = [float(np.mean(by_key[(h, r)])) for r in rhos]
        lows = [float(np.min(by_key[(h, r)])) for r in rhos]
        highs = [float(np.max(by_key[(h, r)])) for r in rhos]
        c = HORIZON[h]
        ax.fill_between(rhos, lows, highs, color=c, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(
            rhos, means, color=c, linewidth=_LW, marker="o", markersize=_MS,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3,
        )
        ax.annotate(
            f"N = {h}", xy=(rhos[-1], means[-1]), xytext=(7, 0),
            textcoords="offset points", color=c, fontsize=10,
            fontweight="bold", va="center",
        )

    # operating regime (rho = 1)
    ax.axvspan(0.62, 1.55, color=INK, alpha=0.05, linewidth=0, zorder=0)
    ax.annotate(
        "operating $\\rho$",
        xy=(1.0, 30), xytext=(0.35, 14), color=MUTED, fontsize=9, ha="center",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ADMM penalty $\\rho$")
    ax.set_ylabel("Mean iterations to convergence")
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.set_xlim(rhos[0] * 0.9, rhos[-1] * 1.6)
    _strip(ax)
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

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.9, 3.1), sharex=True, constrained_layout=True
    )

    ax_a.plot(caps, mean_rmse, color=IDEAL, linewidth=_LW, marker="o",
              markersize=_MS, markeredgecolor="white", markeredgewidth=1.0,
              label="Mean over 60 steps", zorder=3)
    ax_a.plot(caps, terminal, color=LOCAL, linewidth=_LW, marker="s",
              markersize=_MS, markeredgecolor="white", markeredgewidth=1.0,
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

    ax_b.plot(caps, cpu, color=IDEAL, linewidth=_LW, marker="o", markersize=_MS,
              markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    ax_b.set_xscale("log")
    ax_b.set_xticks(caps)
    ax_b.set_xticklabels([f"{c}" for c in caps])
    ax_b.set_xlabel("ADMM iteration cap")
    ax_b.set_ylabel("CPU time per closed loop (s)")
    _strip(ax_b)
    ax_b.set_title("(b) Computational cost", fontsize=10, pad=6)

    _save(fig, out)


# ---------------------------------------------------------------------------
# E3 / E4 — three-panel denial mechanism figure (dot plots, no polylines)
# ---------------------------------------------------------------------------
def _collect(records: list[dict], policy: str) -> dict[float, dict]:
    return {float(r["duration"]): r for r in records if r["policy"] == policy}


def _style_duration_axis(ax: plt.Axes, durations: list[float], xpos: np.ndarray) -> None:
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{int(d)} s" for d in durations])
    ax.set_xlim(-0.5, len(durations) - 0.5)
    _strip(ax)


def plot_denial(records: list[dict], out: Path, accelerating: bool) -> None:
    _apply_style()
    durations = sorted({float(r["duration"]) for r in records})
    xpos = np.arange(len(durations))

    pol_a = ["ideal", "local_mpc", "cv_prediction", "zoh"]
    col_a = {"ideal": IDEAL, "local_mpc": LOCAL, "cv_prediction": CV, "zoh": ZOH}
    lab_a = {
        "ideal": "Ideal (full comm.)",
        "local_mpc": "Uncoupled local MPC",
        "cv_prediction": "CV (decayed)",
        "zoh": "ZOH (decayed)",
    }

    fig = plt.figure(figsize=(6.9, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.5, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # --- (a) confidence-decayed coupling: formation error saturates ----------
    off_a = _dodge(len(pol_a), width=0.72)
    for k, p in enumerate(pol_a):
        m = _collect(records, p)
        for xi, d in enumerate(durations):
            if d not in m:
                continue
            y = float(m[d]["denial_peak_rmse"])
            e = float(m[d].get("denial_peak_rmse_std", 0.0))
            _dot(ax_a, xi + off_a[k], y, e, col_a[p], MARKER[p], fill=True)

    zoh = _collect(records, "zoh")
    zys = [float(zoh[d]["denial_peak_rmse"]) for d in durations]
    # saturation plateau: 30 s and 60 s coincide -> shade the right half
    ax_a.axhspan(zys[-1] * 0.72, zys[-1] * 1.28, xmin=0.44, xmax=1.0,
                 color=ZOH, alpha=0.07, linewidth=0, zorder=0)
    ax_a.annotate(
        "saturates",
        xy=(xpos[-1] + off_a[3], zys[-1]),
        xytext=(xpos[-1] + off_a[3] - 0.12, zys[-1] * 2.6),
        arrowprops=dict(arrowstyle="->", color=ZOH, lw=0.9),
        color=ZOH, fontsize=8.5, va="center", ha="right",
    )

    ax_a.set_yscale("log")
    ax_a.set_ylabel("Peak formation error (m)")
    ax_a.set_title("(a) Confidence-decayed coupling", fontsize=9.5, pad=6)
    _proxy(ax_a, [(col_a[p], MARKER[p], True, lab_a[p]) for p in pol_a],
           loc="upper left", fontsize=7.6)
    _style_duration_axis(ax_a, durations, xpos)

    # --- (b) fixed-weight ablation: growth returns ---------------------------
    pol_b = ["cv_prediction", "cv_fixed", "zoh", "zoh_fixed"]
    fill_b = {
        "cv_prediction": True, "cv_fixed": False, "zoh": True, "zoh_fixed": False,
    }
    col_b = {"cv_prediction": CV, "cv_fixed": CV, "zoh": ZOH, "zoh_fixed": ZOH}
    marker_b = {"cv_prediction": "^", "cv_fixed": "^", "zoh": "s", "zoh_fixed": "s"}
    off_b = _dodge(4, width=0.72)
    for k, p in enumerate(pol_b):
        m = _collect(records, p)
        for xi, d in enumerate(durations):
            y = float(m[d]["denial_peak_rmse"])
            e = float(m[d].get("denial_peak_rmse_std", 0.0))
            _dot(ax_b, xi + off_b[k], y, e, col_b[p], marker_b[p], fill=fill_b[p])

    zf = _collect(records, "zoh_fixed")
    zfys = [float(zf[d]["denial_peak_rmse"]) for d in durations]
    ax_b.annotate(
        "grows",
        xy=(xpos[-1] + off_b[3], zfys[-1]),
        xytext=(xpos[-1] + off_b[3] - 0.1, zfys[-1] * 0.45),
        arrowprops=dict(arrowstyle="->", color=ZOH, lw=0.9),
        color=ZOH, fontsize=8.5, va="center", ha="right",
    )

    ax_b.set_yscale("log")
    ax_b.set_title("(b) Fixed-weight ablation", fontsize=9.5, pad=6)
    _proxy(ax_b, [
        (CV, "^", True, "CV (decayed)"),
        (CV, "^", False, "CV (fixed)"),
        (ZOH, "s", True, "ZOH (decayed)"),
        (ZOH, "s", False, "ZOH (fixed)"),
    ], loc="upper left", fontsize=7.6)
    _style_duration_axis(ax_b, durations, xpos)

    # --- (c) prediction error keeps growing ----------------------------------
    pol_c = ["cv_prediction", "zoh"]
    off_c = _dodge(2, width=0.5)
    for k, p in enumerate(pol_c):
        m = _collect(records, p)
        for xi, d in enumerate(durations):
            y = float(m[d]["prediction_error_peak"])
            e = float(m[d].get("prediction_error_peak_std", 0.0))
            _dot(ax_c, xi + off_c[k], y, e, col_a[p], MARKER[p], fill=True)

    zpe = [float(zoh[d]["prediction_error_peak"]) for d in durations]
    if accelerating:
        note = (
            "prediction error grows "
            f"({zpe[0]:.1f} → {zpe[-1]:.0f} m)\n"
            "while formation error (a) saturates"
        )
    else:
        note = (
            f"prediction error grows ≈{zpe[-1] / zpe[0]:.0f}× "
            f"({zpe[0]:.1f} → {zpe[-1]:.0f} m)\n"
            "while formation error (a) saturates"
        )
    ax_c.text(0.02, 0.74, note, transform=ax_c.transAxes, fontsize=8.2,
              color=INK, va="top", ha="left")

    ax_c.set_yscale("log")
    ax_c.set_ylabel("Peak prediction error $\\|\\xi^{k}\\|$ (m)")
    ax_c.set_title("(c) Prediction error (unbounded)", fontsize=9.5, pad=6)
    _proxy(ax_c, [
        (CV, "^", True, "Constant-velocity"),
        (ZOH, "s", True, "Zero-order hold"),
    ], loc="lower right", fontsize=8.0)
    _style_duration_axis(ax_c, durations, xpos)
    ax_c.set_xlabel("Denial duration (s)")

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
