"""Substantive-robustness experiments that close the three review gaps.

E6  -- comparative distributed baseline: plain Lagrangian dual decomposition
       vs edge-splitting ADMM (iteration-to-accuracy + step-size sensitivity).
E7  -- confidence time-constant sweep: saturation of Theorem 3 across
       T_w in {6, 12, 24} s, with the prediction error tracked alongside.
E8  -- numerical Lipschitz constant of Theorem 3, finite-differenced over
       the edge-weight scale.

Results are written to ``artifacts_v3_extra/`` (raw JSON + a human summary),
which is kept separate from the canonical ``artifacts_v2_ablation/`` evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from strict_admm_dmpc.dual_decomposition import solve_dual_decomposition
from strict_admm_dmpc.experiments_extra import (
    run_e6_dual_decomposition,
    run_e6_convergence_trace,
    run_e7_tw_sweep,
    run_e8_lipschitz,
)

import make_figures_v6 as m

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts_v3_extra"
RAW = OUT / "raw"
TBL = OUT / "tables"
for d in (RAW, TBL):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# E6 -- dual decomposition: diminishing sweep (iteration-to-accuracy)
# ---------------------------------------------------------------------------
DIMINISHING_ALPHAS = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
CONSTANT_ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
E6_SEEDS = range(3)
E6_TOPO = ("chain", "ring", "connected_random")


def _dd_safe(scenario, step_size, schedule, max_iterations, central):
    """Run dual decomposition, returning None on divergence / numerical failure."""
    try:
        result = solve_dual_decomposition(
            scenario,
            step_size=step_size,
            schedule=schedule,
            max_iterations=max_iterations,
            central_objective=central,
        )
        final_gap = result.history[-1].objective_gap
        if not np.isfinite(final_gap) or final_gap > 1.0:
            return None
        return result
    except Exception:
        return None


def _run_e6_constant():
    """Constant-step dual decomposition: document the divergence threshold.

    A plain (non-diminishing) subgradient step has no convergence guarantee;
    it diverges once the step exceeds the reciprocal of the dual gradient's
    Lipschitz constant.  We record the final objective gap per step size so the
    manuscript can state the failure honestly rather than cherry-pick ADMM.
    """
    from strict_admm_dmpc.centralized_qp import solve_centralized_qp
    from strict_admm_dmpc.experiments import build_scenario

    records = []
    for topology in E6_TOPO:
        for seed in E6_SEEDS:
            scenario = build_scenario(seed, topology, horizon=10, rho=1.0)
            central = solve_centralized_qp(scenario).objective
            for alpha in CONSTANT_ALPHAS:
                result = _dd_safe(scenario, alpha, "constant", 2000, central)
                records.append(
                    {
                        "topology": topology,
                        "seed": seed,
                        "alpha": alpha,
                        "diverged": result is None,
                        "final_gap": (
                            float(result.history[-1].objective_gap)
                            if result is not None
                            else float("inf")
                        ),
                    }
                )
    return records


def main() -> None:
    m._style()

    # --- E6 diminishing -----------------------------------------------------
    e6 = run_e6_dual_decomposition(
        seeds=E6_SEEDS,
        topologies=E6_TOPO,
        horizon=10,
        rho=1.0,
        step_sizes=DIMINISHING_ALPHAS,
        max_iterations=3000,
        gap_threshold=1e-4,
    )
    (RAW / "e6_dual_decomposition.json").write_text(
        json.dumps(e6, indent=2), encoding="utf-8"
    )

    # --- E6 constant (divergence) ------------------------------------------
    e6_const = _run_e6_constant()
    (RAW / "e6_dual_constant.json").write_text(
        json.dumps(e6_const, indent=2), encoding="utf-8"
    )

    # --- E6 convergence trace (figure backing) -----------------------------
    e6_trace = run_e6_convergence_trace(
        seed=0,
        topology="chain",
        horizon=10,
        rho=1.0,
        step_sizes=(0.3, 1.0, 10.0),
        max_iterations=5000,
    )
    (RAW / "e6_dual_trace.json").write_text(
        json.dumps(e6_trace, indent=2), encoding="utf-8"
    )

    # --- E7 T_w sweep -------------------------------------------------------
    e7 = run_e7_tw_sweep(
        seeds=range(10), durations=(10.0, 30.0, 60.0), tws=(6.0, 12.0, 24.0), cap=5
    )
    (RAW / "e7_tw_sweep.json").write_text(
        json.dumps(e7, indent=2), encoding="utf-8"
    )

    # --- E8 Lipschitz -------------------------------------------------------
    e8 = run_e8_lipschitz(
        seeds=range(10), topologies=E6_TOPO, horizon=10,
        scales=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    (RAW / "e8_lipschitz.json").write_text(
        json.dumps(e8, indent=2), encoding="utf-8"
    )

    # --- human-readable summary --------------------------------------------
    summary = _summarize(e6, e6_const, e6_trace, e7, e8)
    (TBL / "summary_extra.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # --- figures ------------------------------------------------------------
    plot_dual_decomposition(e6, e6_const, e6_trace, TBL / "e7_dual_decomposition")
    plot_tw_sweep(e7, TBL / "e8_tw_sweep")
    print("wrote figures under", TBL)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("inf")


def _summarize(e6, e6_const, e6_trace, e7, e8):
    # ADMM iterations-to-gap (fair same-metric reference)
    admm_to_gap = _mean([r["admm_iterations_to_gap"] for r in e6])
    # DD iterations-to-gap per alpha
    dd_by_alpha = {}
    for alpha in DIMINISHING_ALPHAS:
        key = str(alpha)
        vals = [r[f"dd_{key}_iterations_to_gap"] for r in e6]
        dd_by_alpha[alpha] = _mean([v for v in vals if v > 0])
    best_alpha = min(dd_by_alpha, key=lambda a: dd_by_alpha[a])
    worst_alpha = max(dd_by_alpha, key=lambda a: dd_by_alpha[a])

    # constant-schedule divergence: min alpha that diverges across seeds
    const_div = {}
    for r in e6_const:
        const_div.setdefault(r["alpha"], []).append(r["diverged"])
    const_fraction = {
        a: _mean(vs) for a, vs in const_div.items()
    }

    # Lipschitz lower bound: max over scalar path, mean over scenarios
    lipschitz = _mean([r["lipschitz_lower_bound"] for r in e8])
    lipschitz_std = float(
        np.std([r["lipschitz_lower_bound"] for r in e8])
    )

    # T_w saturation: for each T_w, ratio of 60s peak to 30s peak (ZOH)
    tw_sat = {}
    tw_level = {}
    for tw in (6.0, 12.0, 24.0):
        p30 = _mean(
            [r["denial_peak_rmse"] for r in e7
             if r["tw"] == tw and r["duration"] == 30.0]
        )
        p60 = _mean(
            [r["denial_peak_rmse"] for r in e7
             if r["tw"] == tw and r["duration"] == 60.0]
        )
        pe30 = _mean(
            [r["prediction_error_peak"] for r in e7
             if r["tw"] == tw and r["duration"] == 30.0]
        )
        pe60 = _mean(
            [r["prediction_error_peak"] for r in e7
             if r["tw"] == tw and r["duration"] == 60.0]
        )
        tw_sat[tw] = p60 / p30 if p30 > 0 else float("inf")
        tw_level[tw] = p60
        # prediction error growth factor 60/30
        tw_sat[f"{tw}_pred_err_ratio"] = pe60 / pe30 if pe30 > 0 else float("inf")

    return {
        "e6": {
            "admm_iterations_to_gap": admm_to_gap,
            "dd_iterations_to_gap_by_alpha": dd_by_alpha,
            "best_alpha": best_alpha,
            "best_iterations": dd_by_alpha[best_alpha],
            "worst_alpha": worst_alpha,
            "worst_iterations": dd_by_alpha[worst_alpha],
        },
        "e6_constant": {
            "diverged_fraction_by_alpha": const_fraction,
        },
        "e7_tw": {
            "saturation_ratio_60_over_30": {str(k): v for k, v in tw_sat.items()},
            "peak60_by_tw": {str(k): v for k, v in tw_level.items()},
        },
        "e8": {
            "lipschitz_lower_bound": lipschitz,
            "lipschitz_std": lipschitz_std,
        },
    }


# ---------------------------------------------------------------------------
# Figure: dual-decomposition comparison
# ---------------------------------------------------------------------------
def plot_dual_decomposition(e6, e6_const, e6_trace, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.83, 2.6), constrained_layout=True
    )

    # --- panel (a): iterations-to-gap vs step size (U-shape) ---------------
    alphas = sorted(DIMINISHING_ALPHAS)
    y = [
        _mean([r[f"dd_{alpha}_iterations_to_gap"] for r in e6
               if r[f"dd_{alpha}_iterations_to_gap"] > 0])
        for alpha in alphas
    ]
    ax_a.plot(alphas, y, marker="o", markersize=5, linewidth=1.3,
              color=m.ZOH, label="dual decomposition\n(diminishing step)")
    admm_mean = _mean([r["admm_iterations_to_gap"] for r in e6])
    ax_a.axhline(admm_mean, color=m.IDEAL, linewidth=1.3, linestyle="--",
                 label=f"ADMM (mean {admm_mean:.0f})")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"step size $\alpha$")
    ax_a.set_ylabel(r"iterations to $10^{-4}$ gap")
    ax_a.set_title("(a) step-size sensitivity", fontsize=8.5)
    m._strip(ax_a)
    ax_a.legend(fontsize=6.5, frameon=False, loc="upper left")

    # --- panel (b): convergence traces -------------------------------------
    admm_trace = e6_trace["admm"]
    ax_b.plot(
        [r["iteration"] for r in admm_trace],
        [r["objective_gap"] for r in admm_trace],
        color=m.IDEAL, linewidth=1.3, label="ADMM",
    )
    for alpha, color in ((0.3, "#9EC5E0"), (1.0, m.ZOH), (10.0, "#D55E00")):
        dd = e6_trace["dual_decomposition"].get(str(alpha), [])
        if dd:
            ax_b.plot(
                [r["iteration"] for r in dd],
                [r["objective_gap"] for r in dd],
                color=color, linewidth=1.3, linestyle="-",
                label=rf"dual $\alpha={alpha:g}$",
            )
    ax_b.axhline(1e-4, color=m.MUTED, linewidth=0.8, linestyle=":")
    ax_b.text(1.0, 1.6e-4, r"$10^{-4}$", fontsize=6.5, color=m.MUTED)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("iteration")
    ax_b.set_ylabel("relative objective gap")
    ax_b.set_title("(b) convergence trace", fontsize=8.5)
    m._strip(ax_b)
    ax_b.legend(fontsize=6.5, frameon=False, loc="lower left")

    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: T_w sweep (formation error saturates while prediction error grows)
# ---------------------------------------------------------------------------
def plot_tw_sweep(e7, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.83, 2.6), constrained_layout=True
    )
    tws = (6.0, 12.0, 24.0)
    durations = (10.0, 30.0, 60.0)
    colors = {6.0: "#9EC5E0", 12.0: m.IDEAL, 24.0: "#1F4E79"}

    for tw in tws:
        peaks = [
            _mean([r["denial_peak_rmse"] for r in e7
                   if r["tw"] == tw and r["duration"] == d])
            for d in durations
        ]
        ax_a.plot(durations, peaks, marker="o", markersize=5, linewidth=1.3,
                  color=colors[tw], label=rf"$T_\mathrm{{w}}={tw:g}$ s")
        perr = [
            _mean([r["prediction_error_peak"] for r in e7
                   if r["tw"] == tw and r["duration"] == d])
            for d in durations
        ]
        ax_b.plot(durations, perr, marker="s", markersize=5, linewidth=1.3,
                  color=colors[tw], label=rf"$T_\mathrm{{w}}={tw:g}$ s")

    ax_a.set_yscale("log")
    ax_a.set_xlabel("denial duration (s)")
    ax_a.set_ylabel("denial-window peak RMSE (m)")
    ax_a.set_title("(a) formation error saturates", fontsize=8.5)
    m._strip(ax_a)
    ax_a.legend(fontsize=6.5, frameon=False)

    ax_b.set_yscale("log")
    ax_b.set_xlabel("denial duration (s)")
    ax_b.set_ylabel(r"prediction error peak $\|\xi^k\|$ (m)")
    ax_b.set_title("(b) prediction error keeps growing", fontsize=8.5)
    m._strip(ax_b)
    ax_b.legend(fontsize=6.5, frameon=False)

    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
