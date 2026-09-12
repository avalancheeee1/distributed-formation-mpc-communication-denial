from __future__ import annotations

import json
import pytest

from strict_admm_dmpc.experiments import build_scenario, run_e0_correctness
from strict_admm_dmpc.reporting import summarize_experiments


def test_scenario_builder_is_deterministic_and_seeded() -> None:
    first = build_scenario(seed=4, topology="chain", horizon=5, rho=1.0)
    repeated = build_scenario(seed=4, topology="chain", horizon=5, rho=1.0)
    changed = build_scenario(seed=5, topology="chain", horizon=5, rho=1.0)

    assert first.seed == repeated.seed
    assert (first.initial_states == repeated.initial_states).all()
    assert not (first.initial_states == changed.initial_states).all()


def test_e0_correctness_gate_passes_small_profile() -> None:
    records = run_e0_correctness(seeds=(0, 1))

    assert len(records) == 6
    assert all(record["passed"] for record in records)
    assert max(record["objective_gap"] for record in records) <= 2e-5


def test_reporting_writes_machine_and_human_readable_outputs(tmp_path) -> None:
    raw = {
        "e0": [
            {
                "passed": True,
                "objective_gap": 1e-8,
                "decision_gap": 2e-8,
            }
        ],
        "e1": [
            {
                "topology": "chain",
                "horizon": 5,
                "rho": 1.0,
                "seed": 0,
                "iterations": 12,
                "objective_gap": 1e-7,
                "primal_residual": 1e-6,
                "dual_residual": 1e-6,
                "runtime_seconds": 0.01,
                "cpu_seconds": 0.009,
            }
        ],
        "e2": [
            {
                "cap": 1,
                "mean_rmse": 0.1,
                "peak_rmse": 0.2,
                "terminal_rmse": 0.05,
                "control_energy": 1.2,
                "runtime_seconds": 0.03,
                "cpu_seconds": 0.02,
                "communicated_scalars": 100,
            }
        ],
        "e3": [
            {
                "duration": 10.0,
                "policy": "zoh",
                "denial_peak_rmse": 0.8,
                "reconnection_peak_rmse": 0.7,
                "mean_rmse": 0.3,
                "denial_mean_rmse": 0.4,
                "control_energy": 1.5,
                "runtime_seconds": 0.2,
                "cpu_seconds": 0.1,
                "communicated_scalars": 0,
            },
            {
                "duration": 10.0,
                "policy": "cv_prediction",
                "denial_peak_rmse": 0.02,
                "reconnection_peak_rmse": 0.01,
                "mean_rmse": 0.01,
                "denial_mean_rmse": 0.01,
                "control_energy": 1.1,
                "runtime_seconds": 0.3,
                "cpu_seconds": 0.15,
                "communicated_scalars": 0,
            },
        ],
    }
    raw_path = tmp_path / "results.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    outputs = summarize_experiments(raw_path, tmp_path)

    assert outputs["summary_json"].exists()
    assert outputs["summary_markdown"].exists()
    assert outputs["e1_figure"].exists()
    assert outputs["e2_figure"].exists()
    assert outputs["e3_figure"].exists()
    assert outputs["results_values"].exists()
    assert "E1" in outputs["summary_markdown"].read_text(encoding="utf-8")


@pytest.mark.parametrize("e0", [[], [{"passed": False, "objective_gap": 1.0}]])
def test_reporting_rejects_missing_or_failed_correctness_gate(tmp_path, e0) -> None:
    raw_path = tmp_path / "results.json"
    raw_path.write_text(
        json.dumps({"e0": e0, "e1": [], "e2": [], "e3": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="E0 correctness gate"):
        summarize_experiments(raw_path, tmp_path)
