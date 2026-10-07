"""Scaling and noise-robustness experiments that close the two review gaps.

E9  -- swarm-size scaling: correctness gate + iteration/runtime cost on
       4 / 6 / 8 agents across chain / ring / connected_random topologies.
E10 -- measurement noise: observed-position noise sweep (sigma in {0, 0.05,
       0.1, 0.2} m), verifying the Theorem-3 saturation signature survives
       localization noise.

Results are written alongside the other substantive-robustness evidence in
``artifacts_v3_extra/`` (raw JSON + a human summary), separate from the
canonical ``artifacts_v2_ablation/`` evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from strict_admm_dmpc.experiments_scaling import run_e9_scaling, run_e10_noise

import make_figures_v6 as m

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts_v3_extra"
RAW = OUT / "raw"
TBL = OUT / "tables"
for d in (RAW, TBL):
    d.mkdir(parents=True, exist_ok=True)

E9_TOPO = ("chain", "ring", "connected_random")
E9_AGENTS = (4, 6, 8)
SIGMAS = (0.0, 0.05, 0.1, 0.2)
DURATIONS = (30.0, 60.0)

# distinct colour per topology (ordered ramp, reuse the horizon ramp)
TOPO_COLORS = {"chain": m.IDEAL, "ring": m.ZOH, "connected_random": "#009E73"}
# ordered ramp per sigma
SIGMA_COLORS = {0.0: "#9EC5E0", 0.05: m.IDEAL, 0.1: m.ZOH, 0.2: "#1F4E79"}


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("inf")


def _std(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.std(xs)) if xs else float("inf")


def main() -> None:
    m._style()

    # --- E9 scaling ---------------------------------------------------------
    e9 = run_e9_scaling(
        seeds=range(5),
        n_agents_list=E9_AGENTS,
        topologies=E9_TOPO,
        horizon=10,
        rho=1.0,
    )
    (RAW / "e9_scaling.json").write_text(
        json.dumps(e9, indent=2), encoding="utf-8"
    )

    # --- E10 noise ----------------------------------------------------------
    e10 = run_e10_noise(
        seeds=range(10),
        sigmas=SIGMAS,
        durations=DURATIONS,
        cap=5,
    )
    (RAW / "e10_noise.json").write_text(
        json.dumps(e10, indent=2), encoding="utf-8"
    )

    # --- human-readable summary --------------------------------------------
    summary = _summarize(e9, e10)
    (TBL / "summary_scaling_noise.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # --- figures ------------------------------------------------------------
    plot_scaling(e9, TBL / "e9_scaling")
    plot_noise(e10, TBL / "e10_noise")
    print("wrote figures under", TBL)


def _summarize(e9, e10):
    # E9: all-pass gate + iteration/runtime growth 4 -> 8 agents
    passed = _mean([r["passed"] for r in e9])
    iters_by_n = {}
    runtime_by_n = {}
    for n in E9_AGENTS:
        iters_by_n[n] = _mean([r["iterations"] for r in e9 if r["n_agents"] == n])
        runtime_by_n[n] = _mean(
            [r["runtime_seconds"] for r in e9 if r["n_agents"] == n]
        )
    # E10: saturation ratio (60s / 30s peak) per sigma, and prediction-error
    # growth ratio over the same window
    sat_by_sigma = {}
    growth_by_sigma = {}
    for sigma in SIGMAS:
        p30 = _mean(
            [r["denial_peak_rmse"] for r in e10
             if r["sigma"] == sigma and r["duration"] == 30.0]
        )
        p60 = _mean(
            [r["denial_peak_rmse"] for r in e10
             if r["sigma"] == sigma and r["duration"] == 60.0]
        )
        pe30 = _mean(
            [r["prediction_error_peak"] for r in e10
             if r["sigma"] == sigma and r["duration"] == 30.0]
        )
        pe60 = _mean(
            [r["prediction_error_peak"] for r in e10
             if r["sigma"] == sigma and r["duration"] == 60.0]
        )
        sat_by_sigma[str(sigma)] = p60 / p30 if p30 > 0 else float("inf")
        growth_by_sigma[str(sigma)] = pe60 / pe30 if pe30 > 0 else float("inf")

    return {
        "e9_scaling": {
            "gate_pass_fraction": passed,
            "iterations_by_n_agents": {str(k): v for k, v in iters_by_n.items()},
            "runtime_seconds_by_n_agents": {
                str(k): v for k, v in runtime_by_n.items()
            },
        },
        "e10_noise": {
            "saturation_ratio_60_over_30": {
                str(k): v for k, v in sat_by_sigma.items()
            },
            "prediction_error_growth_60_over_30": {
                str(k): v for k, v in growth_by_sigma.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# Figure: swarm-size scaling
# ---------------------------------------------------------------------------
def plot_scaling(e9, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.83, 2.6), constrained_layout=True
    )

    for topology in E9_TOPO:
        iters = [
            _mean([r["iterations"] for r in e9
                   if r["n_agents"] == n and r["topology"] == topology])
            for n in E9_AGENTS
        ]
        iters_std = [
            _std([r["iterations"] for r in e9
                  if r["n_agents"] == n and r["topology"] == topology])
            for n in E9_AGENTS
        ]
        ax_a.errorbar(
            E9_AGENTS, iters, yerr=iters_std, marker="o", markersize=5,
            linewidth=1.3, color=TOPO_COLORS[topology], capsize=2,
            label=topology.replace("_", " "),
        )
        runtime = [
            _mean([r["runtime_seconds"] for r in e9
                   if r["n_agents"] == n and r["topology"] == topology])
            for n in E9_AGENTS
        ]
        ax_b.errorbar(
            E9_AGENTS, runtime, yerr=0.0, marker="s", markersize=5,
            linewidth=1.3, color=TOPO_COLORS[topology], capsize=2,
            label=topology.replace("_", " "),
        )

    ax_a.set_xticks(E9_AGENTS)
    ax_a.set_xlabel("number of agents")
    ax_a.set_ylabel("ADMM iterations")
    ax_a.set_title("(a) iteration cost", fontsize=m.FS_EMPHASIS)
    m._strip(ax_a)
    ax_a.legend(fontsize=m.FS_BODY, frameon=False)

    ax_b.set_xticks(E9_AGENTS)
    ax_b.set_xlabel("number of agents")
    ax_b.set_ylabel("runtime (s)")
    ax_b.set_title("(b) wall-clock cost", fontsize=m.FS_EMPHASIS)
    m._strip(ax_b)
    ax_b.legend(fontsize=m.FS_BODY, frameon=False)

    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: measurement noise
# ---------------------------------------------------------------------------
def plot_noise(e10, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.83, 2.6), constrained_layout=True
    )

    for sigma in SIGMAS:
        peaks = [
            _mean([r["denial_peak_rmse"] for r in e10
                   if r["sigma"] == sigma and r["duration"] == d])
            for d in DURATIONS
        ]
        ax_a.plot(
            DURATIONS, peaks, marker="o", markersize=5, linewidth=1.3,
            color=SIGMA_COLORS[sigma],
            label=rf"$\sigma={sigma:g}$ m",
        )
        perr = [
            _mean([r["prediction_error_peak"] for r in e10
                   if r["sigma"] == sigma and r["duration"] == d])
            for d in DURATIONS
        ]
        ax_b.plot(
            DURATIONS, perr, marker="s", markersize=5, linewidth=1.3,
            color=SIGMA_COLORS[sigma],
            label=rf"$\sigma={sigma:g}$ m",
        )

    ax_a.set_xlabel("denial duration (s)")
    ax_a.set_ylabel("denial-window peak RMSE (m)")
    ax_a.set_title("(a) formation error saturates", fontsize=m.FS_EMPHASIS)
    m._strip(ax_a)
    ax_a.legend(fontsize=m.FS_BODY, frameon=False)

    ax_b.set_xlabel("denial duration (s)")
    ax_b.set_ylabel(r"prediction error peak $\|\xi^k\|$ (m)")
    ax_b.set_title("(b) prediction error keeps growing", fontsize=m.FS_EMPHASIS)
    m._strip(ax_b)
    ax_b.legend(fontsize=m.FS_BODY, frameon=False)

    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
