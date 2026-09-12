"""Publication-quality figure generation for the v2 manuscript.

Figures follow a colorblind-safe categorical palette (Okabe--Ito, the
Nature-Methods standard) and a set of fixed mark specs: thin 2 px lines,
>=8 px markers with a surface ring, recessive hairline gridlines, no top or
right spines, and human-readable direct labels / legends.  Each function
consumes the grouped record lists already written to ``summary.json`` and
emits both vector PDF and high-resolution PNG into ``output_dir``.

The palette was validated with the dataviz ``validate_palette.js`` harness:
``#0072B2,#E69F00,#009E73,#D55E00`` passes the lightness band, chroma floor,
CVD-separation (worst adjacent deutan dE = 11.0) and normal-vision floor
(worst adjacent 24.2) gates on a light surface.  The orange step sits just
under 3:1 contrast, so it is always paired with a direct label or legend --
never relied on as color-alone.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Okabe--Ito colorblind-safe categorical palette -------------------------
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"

# Chart chrome & ink
INK = "#1a1a1a"          # primary text / axis titles
MUTED = "#6b6b6b"        # tick labels
GRID = "#e3e3e0"         # hairline gridline
AXIS = "#c8c8c4"         # baseline / axis rule

_LINE_WIDTH = 2.0
_MARKER_SIZE = 7.0


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def _strip_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, stem: Path) -> Path:
    fig.savefig(stem.with_suffix(".pdf"))
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return png


def _markers() -> list[str]:
    return ["o", "s", "^", "D"]


def plot_e1_convergence(
    records: list[dict],
    output_path: Path,
) -> Path:
    """Mean strict-ADMM iterations vs penalty, per horizon, with a topology band.

    ``records`` is the grouped E1 list: one dict per (topology, horizon, rho)
    carrying ``iterations_mean`` and ``iterations_std``.  The three topologies
    are collapsed to a mean line per horizon; the shaded band spans the min--max
    across topologies and thus visualises topology insensitivity directly.
    """
    _apply_style()
    by_key: dict[tuple[int, float], list[float]] = defaultdict(list)
    for record in records:
        key = (int(record["horizon"]), float(record["rho"]))
        by_key[key].append(float(record["iterations_mean"]))

    horizons = sorted({int(r["horizon"]) for r in records})
    rhos = sorted({float(r["rho"]) for r in records})
    colors = {"5": BLUE, "10": ORANGE, "20": GREEN}

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for horizon in horizons:
        means = [float(np.mean(by_key[(horizon, rho)])) for rho in rhos]
        lows = [float(np.min(by_key[(horizon, rho)])) for rho in rhos]
        highs = [float(np.max(by_key[(horizon, rho)])) for rho in rhos]
        color = colors[str(horizon)]
        ax.fill_between(
            rhos,
            lows,
            highs,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            rhos,
            means,
            color=color,
            linewidth=_LINE_WIDTH,
            marker="o",
            markersize=_MARKER_SIZE,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            f"N = {horizon}",
            xy=(rhos[-1], means[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=10,
            fontweight="bold",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"ADMM penalty $\rho$")
    ax.set_ylabel("Mean iterations to convergence")
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{rho:g}" for rho in rhos])
    ax.set_xlim(rhos[0] * 0.9, rhos[-1] * 1.6)
    # Operating penalty used in the closed-loop runs (rho = 1): the shallow
    # minimum in the mean iteration count across horizons.
    ax.axvspan(0.7, 1.4, color=INK, alpha=0.05, linewidth=0, zorder=0)
    _strip_axes(ax)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_e2_budget(
    records: list[dict],
    output_path: Path,
) -> Path:
    """Effect of the ADMM iteration cap on closed-loop tracking and cost.

    ``records`` is the grouped E2 list keyed by ``cap`` carrying ``mean_rmse``,
    ``terminal_rmse`` and ``cpu_seconds``.  Two shared-x panels separate the two
    stories the data supports: (a) the *mean* tracking error is dominated by the
    startup transient and is nearly cap-independent, whereas the *terminal*
    (steady-state) error improves by roughly four orders of magnitude and
    saturates around cap 20; (b) the per-closed-loop CPU time grows with the cap.
    """
    _apply_style()
    records = sorted(records, key=lambda r: int(r["cap"]))
    caps = [int(r["cap"]) for r in records]
    cpu = [float(r["cpu_seconds"]) for r in records]
    mean_rmse = [float(r["mean_rmse"]) for r in records]
    terminal = [float(r["terminal_rmse"]) for r in records]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.9, 3.1), sharex=True
    )

    # Panel (a): mean vs terminal tracking error.
    ax_a.plot(
        caps,
        mean_rmse,
        color=BLUE,
        linewidth=_LINE_WIDTH,
        marker="o",
        markersize=_MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Mean over 60 steps",
        zorder=3,
    )
    ax_a.plot(
        caps,
        terminal,
        color=ORANGE,
        linewidth=_LINE_WIDTH,
        marker="s",
        markersize=_MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Terminal (steady state)",
        zorder=3,
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xticks(caps)
    ax_a.set_xticklabels([f"{cap}" for cap in caps])
    ax_a.set_xlabel("ADMM iteration cap")
    ax_a.set_ylabel("Tracking error (m)")
    ax_a.legend(frameon=False, loc="center right", handlelength=1.6)
    _strip_axes(ax_a)
    ax_a.set_title("(a) Tracking error", fontsize=10, pad=6)
    if 20 in caps:
        ax_a.annotate(
            "> 3 orders of magnitude",
            xy=(20, terminal[caps.index(20)]),
            xytext=(3, 2e-3),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
            color=INK,
            fontsize=9,
            va="center",
            ha="left",
        )

    # Panel (b): computational cost.
    ax_b.plot(
        caps,
        cpu,
        color=BLUE,
        linewidth=_LINE_WIDTH,
        marker="o",
        markersize=_MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )
    ax_b.set_xscale("log")
    ax_b.set_xticks(caps)
    ax_b.set_xticklabels([f"{cap}" for cap in caps])
    ax_b.set_xlabel("ADMM iteration cap")
    ax_b.set_ylabel("CPU time per closed loop (s)")
    _strip_axes(ax_b)
    ax_b.set_title("(b) Computational cost", fontsize=10, pad=6)

    fig.tight_layout()
    return _save(fig, output_path)


def plot_e3_denial(
    records: list[dict],
    output_path: Path,
) -> Path:
    """Peak formation error during denial as a two-panel visual argument.

    Panel (a) carries the primary ordering under confidence-decayed coupling:
    ideal / uncoupled local MPC / constant-velocity / zero-order hold, with a
    logarithmic error axis so the near-ideal policies stay distinguishable from
    ZOH (~3 orders larger).  Panel (b) isolates the causal mechanism: it
    overlays the decayed curve (solid) with the fixed-weight counterfactual
    (dashed) for CV and ZOH only, showing that removing the decay converts ZOH's
    saturation back into growth.  Shared log axis; one policy keeps one color
    and one marker across both panels.
    """
    _apply_style()
    policy_colors = {
        "ideal": BLUE,
        "local_mpc": ORANGE,
        "cv_prediction": GREEN,
        "cv_fixed": GREEN,
        "zoh": VERMILLION,
        "zoh_fixed": VERMILLION,
    }
    policy_markers = {
        "ideal": "o",
        "local_mpc": "s",
        "cv_prediction": "^",
        "cv_fixed": "^",
        "zoh": "D",
        "zoh_fixed": "D",
    }
    policy_linestyles = {
        "ideal": "-",
        "local_mpc": "-",
        "cv_prediction": "-",
        "cv_fixed": "--",
        "zoh": "-",
        "zoh_fixed": "--",
    }
    panel_a_labels = {
        "ideal": "Ideal (full comm.)",
        "local_mpc": "Uncoupled local MPC",
        "cv_prediction": "Constant-velocity",
        "zoh": "Zero-order hold",
    }
    panel_b_labels = {
        "cv_prediction": "CV (decayed)",
        "cv_fixed": "CV (fixed weight)",
        "zoh": "ZOH (decayed)",
        "zoh_fixed": "ZOH (fixed weight)",
    }
    durations = sorted({float(r["duration"]) for r in records})

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)

    def _draw(ax: plt.Axes, policies: list[str], labels: dict[str, str]) -> None:
        for policy in policies:
            subset = sorted(
                [r for r in records if r["policy"] == policy],
                key=lambda r: float(r["duration"]),
            )
            if not subset:
                continue
            xs = [float(r["duration"]) for r in subset]
            ys = [float(r["denial_peak_rmse"]) for r in subset]
            errs = [float(r["denial_peak_rmse_std"]) for r in subset]
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                color=policy_colors[policy],
                linestyle=policy_linestyles[policy],
                linewidth=_LINE_WIDTH,
                marker=policy_markers[policy],
                markersize=_MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=1.2,
                capsize=3,
                capthick=1.2,
                elinewidth=1.2,
                label=labels[policy],
                zorder=3,
            )

    _draw(ax_a, ["ideal", "local_mpc", "cv_prediction", "zoh"], panel_a_labels)
    _draw(
        ax_b,
        ["cv_prediction", "cv_fixed", "zoh", "zoh_fixed"],
        panel_b_labels,
    )

    for ax in (ax_a, ax_b):
        ax.set_yscale("log")
        ax.set_xlabel("Denial duration (s)")
        ax.set_xticks(durations)
        ax.set_xticklabels([f"{int(d):g}" for d in durations])
        ax.set_xlim(durations[0] - 1.5, durations[-1] + 6.0)
        _strip_axes(ax)

    # Saturation fingerprint: the decayed zero-order-hold curve plateaus between
    # 30 s and 60 s in both the nominal and accelerating scenarios.
    zoh = sorted(
        [r for r in records if r["policy"] == "zoh"],
        key=lambda r: float(r["duration"]),
    )
    if len(zoh) >= 3:
        ax_a.annotate(
            "saturates",
            xy=(float(zoh[-1]["duration"]), float(zoh[-1]["denial_peak_rmse"])),
            xytext=(
                float(zoh[-1]["duration"]) - 6.0,
                float(zoh[-1]["denial_peak_rmse"]) * 3.0,
            ),
            arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=1.0),
            color=VERMILLION,
            fontsize=9,
            va="center",
            ha="right",
        )

    ax_a.set_ylabel("Peak formation error during denial (m)")
    ax_a.set_title("(a) Confidence-decayed coupling", fontsize=10, pad=6)
    ax_b.set_title("(b) Fixed-weight ablation", fontsize=10, pad=6)
    ax_a.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.4)
    ax_b.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.4)

    fig.tight_layout()
    return _save(fig, output_path)


def render_all(summary: dict, output_dir: Path) -> dict[str, Path]:
    """Render every available figure from a loaded summary document."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}
    if summary.get("e1"):
        generated["e1_figure"] = plot_e1_convergence(
            summary["e1"], output_dir / "e1_convergence"
        )
    if summary.get("e2"):
        generated["e2_figure"] = plot_e2_budget(
            summary["e2"], output_dir / "e2_budget"
        )
    if summary.get("e3"):
        generated["e3_figure"] = plot_e3_denial(
            summary["e3"], output_dir / "e3_denial"
        )
    if summary.get("e4"):
        generated["e4_figure"] = plot_e3_denial(
            summary["e4"], output_dir / "e4_acceleration"
        )
    return generated
