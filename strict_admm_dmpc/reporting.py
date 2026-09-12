"""Generate auditable summaries and figures from raw v2 experiment data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strict_admm_dmpc.plotting import (
    plot_e1_convergence,
    plot_e2_budget,
    plot_e3_denial,
)


def _markdown_table(records: list[dict[str, object]]) -> str:
    """Render simple records without pandas' optional ``tabulate`` dependency."""
    if not records:
        return ""
    columns = list(records[0])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(record[column]) for column in columns) + " |"
        for record in records
    ]
    return "\n".join([header, separator, *rows])


def summarize_experiments(raw_path: Path, output_dir: Path) -> dict[str, Path]:
    """Write JSON/Markdown summaries and any available experiment figures."""
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    generated_figures: dict[str, Path] = {}
    values = {
        "EZeroCases": "NA",
        "EZeroObjectiveGap": "NA",
        "EZeroDecisionGap": "NA",
        "EOneObjectiveGap": "NA",
        "EOneRhoOneIterations": "NA",
        "ETwoCapOneRmse": "NA",
        "ETwoCapFiveRmse": "NA",
        "ETwoCapHundredRmse": "NA",
        "EThreeCvSixtyPeak": "NA",
        "EThreeLocalSixtyPeak": "NA",
        "EThreeZohSixtyPeak": "NA",
        "EThreeCvFixedSixtyPeak": "NA",
        "EThreeZohFixedSixtyPeak": "NA",
        "EThreeCvPredErrorTen": "NA",
        "EThreeCvPredErrorThirty": "NA",
        "EThreeCvPredErrorSixty": "NA",
        "EThreeZohPredErrorTen": "NA",
        "EThreeZohPredErrorThirty": "NA",
        "EThreeZohPredErrorSixty": "NA",
        "EFourCvSixtyPeak": "NA",
        "EFourZohSixtyPeak": "NA",
        "EFourCvFixedSixtyPeak": "NA",
        "EFourZohFixedSixtyPeak": "NA",
        "EFourCvPredErrorTen": "NA",
        "EFourCvPredErrorThirty": "NA",
        "EFourCvPredErrorSixty": "NA",
        "EFourZohPredErrorTen": "NA",
        "EFourZohPredErrorThirty": "NA",
        "EFourZohPredErrorSixty": "NA",
    }
    markdown = ["# Strict ADMM-DMPC v2 experiment summary", ""]

    e0 = raw.get("e0")
    if not isinstance(e0, list) or not e0 or any(
        not isinstance(record, dict) or record.get("passed") is not True
        for record in e0
    ):
        raise ValueError("E0 correctness gate is missing or failed")
    try:
        objective_gaps = np.asarray(
            [record["objective_gap"] for record in e0], dtype=float
        )
        decision_gaps = np.asarray(
            [record["decision_gap"] for record in e0], dtype=float
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("E0 correctness gate has invalid records") from exc
    if not np.all(np.isfinite(objective_gaps)) or not np.all(
        np.isfinite(decision_gaps)
    ):
        raise ValueError("E0 correctness gate has non-finite gaps")
    if np.max(objective_gaps) > 2e-5 or np.max(decision_gaps) > 2e-5:
        raise ValueError("E0 correctness gate exceeds the declared tolerance")
    e0_summary = {
        "passed": len(e0),
        "total": len(e0),
        "objective_gap_max": float(np.max(objective_gaps)),
        "decision_gap_max": float(np.max(decision_gaps)),
    }
    summary["e0"] = e0_summary
    values.update(
        {
            "EZeroCases": str(len(e0)),
            "EZeroObjectiveGap": f"{float(np.max(objective_gaps)):.3g}",
            "EZeroDecisionGap": f"{float(np.max(decision_gaps)):.3g}",
        }
    )
    markdown.extend(
        ["## E0 correctness gate", "", _markdown_table([e0_summary]), ""]
    )

    e1 = pd.DataFrame(raw.get("e1", []))
    if not e1.empty:
        if "cpu_seconds" not in e1:
            e1["cpu_seconds"] = e1["runtime_seconds"]
        grouped = (
            e1.groupby(["topology", "horizon", "rho"], as_index=False)
            .agg(
                iterations_mean=("iterations", "mean"),
                iterations_std=("iterations", "std"),
                objective_gap_max=("objective_gap", "max"),
                runtime_mean=("runtime_seconds", "mean"),
                cpu_seconds_mean=("cpu_seconds", "mean"),
            )
            .fillna(0.0)
            .to_dict(orient="records")
        )
        summary["e1"] = grouped
        rho_one = e1[np.isclose(e1["rho"], 1.0)]
        values["EOneObjectiveGap"] = f"{float(e1['objective_gap'].max()):.3g}"
        values["EOneRhoOneIterations"] = f"{float(rho_one['iterations'].mean()):.2f}"
        markdown.extend(["## E1 convergence", "", _markdown_table(grouped), ""])
        generated_figures["e1_figure"] = plot_e1_convergence(
            grouped, output_dir / "e1_convergence"
        )

    e2 = pd.DataFrame(raw.get("e2", []))
    if not e2.empty:
        if "cpu_seconds" not in e2:
            e2["cpu_seconds"] = e2["runtime_seconds"]
        grouped_e2 = (
            e2.groupby("cap", as_index=False)
            .agg(
                mean_rmse=("mean_rmse", "mean"),
                mean_rmse_std=("mean_rmse", "std"),
                peak_rmse=("peak_rmse", "mean"),
                terminal_rmse=("terminal_rmse", "mean"),
                control_energy=("control_energy", "mean"),
                runtime_seconds=("runtime_seconds", "mean"),
                cpu_seconds=("cpu_seconds", "mean"),
                communicated_scalars=("communicated_scalars", "mean"),
            )
            .fillna(0.0)
            .to_dict(orient="records")
        )
        summary["e2"] = grouped_e2
        for cap, macro in (
            (1, "ETwoCapOneRmse"),
            (5, "ETwoCapFiveRmse"),
            (100, "ETwoCapHundredRmse"),
        ):
            matching = pd.DataFrame(grouped_e2)
            matching = matching[matching["cap"] == cap]
            if not matching.empty:
                values[macro] = f"{float(matching.iloc[0]['mean_rmse']):.4f}"
        markdown.extend(["## E2 iteration budget", "", _markdown_table(grouped_e2), ""])
        generated_figures["e2_figure"] = plot_e2_budget(
            grouped_e2, output_dir / "e2_budget"
        )

    e3 = pd.DataFrame(raw.get("e3", []))
    if not e3.empty:
        if "cpu_seconds" not in e3:
            e3["cpu_seconds"] = e3["runtime_seconds"]
        if "prediction_error_peak" not in e3:
            e3["prediction_error_peak"] = 0.0
        grouped_e3 = (
            e3.groupby(["duration", "policy"], as_index=False)
            .agg(
                denial_peak_rmse=("denial_peak_rmse", "mean"),
                denial_peak_rmse_std=("denial_peak_rmse", "std"),
                reconnection_peak_rmse=("reconnection_peak_rmse", "mean"),
                mean_rmse=("mean_rmse", "mean"),
                denial_mean_rmse=("denial_mean_rmse", "mean"),
                prediction_error_peak=("prediction_error_peak", "mean"),
                prediction_error_peak_std=("prediction_error_peak", "std"),
                control_energy=("control_energy", "mean"),
                runtime_seconds=("runtime_seconds", "mean"),
                cpu_seconds=("cpu_seconds", "mean"),
                communicated_scalars=("communicated_scalars", "mean"),
            )
            .fillna(0.0)
            .to_dict(orient="records")
        )
        summary["e3"] = grouped_e3
        for policy, macro in (
            ("cv_prediction", "EThreeCvSixtyPeak"),
            ("local_mpc", "EThreeLocalSixtyPeak"),
            ("zoh", "EThreeZohSixtyPeak"),
            ("cv_fixed", "EThreeCvFixedSixtyPeak"),
            ("zoh_fixed", "EThreeZohFixedSixtyPeak"),
        ):
            matching = pd.DataFrame(grouped_e3)
            matching = matching[
                (matching["policy"] == policy) & (matching["duration"] == 60.0)
            ]
            if not matching.empty:
                values[macro] = (
                    f"{float(matching.iloc[0]['denial_peak_rmse']):.4f}"
                )
        for policy, prefix in (
            ("cv_prediction", "EThreeCvPredError"),
            ("zoh", "EThreeZohPredError"),
        ):
            for duration, suffix in (
                (10.0, "Ten"),
                (30.0, "Thirty"),
                (60.0, "Sixty"),
            ):
                matching = pd.DataFrame(grouped_e3)
                matching = matching[
                    (matching["policy"] == policy)
                    & (matching["duration"] == duration)
                ]
                if not matching.empty:
                    values[prefix + suffix] = (
                        f"{float(matching.iloc[0]['prediction_error_peak']):.4g}"
                    )
        markdown.extend(["## E3 communication denial", "", _markdown_table(grouped_e3), ""])
        generated_figures["e3_figure"] = plot_e3_denial(
            grouped_e3, output_dir / "e3_denial"
        )

    e4 = pd.DataFrame(raw.get("e4", []))
    if not e4.empty:
        if "cpu_seconds" not in e4:
            e4["cpu_seconds"] = e4["runtime_seconds"]
        if "prediction_error_peak" not in e4:
            e4["prediction_error_peak"] = 0.0
        grouped_e4 = (
            e4.groupby(["duration", "policy"], as_index=False)
            .agg(
                denial_peak_rmse=("denial_peak_rmse", "mean"),
                denial_peak_rmse_std=("denial_peak_rmse", "std"),
                reconnection_peak_rmse=("reconnection_peak_rmse", "mean"),
                mean_rmse=("mean_rmse", "mean"),
                denial_mean_rmse=("denial_mean_rmse", "mean"),
                prediction_error_peak=("prediction_error_peak", "mean"),
                prediction_error_peak_std=("prediction_error_peak", "std"),
                control_energy=("control_energy", "mean"),
                runtime_seconds=("runtime_seconds", "mean"),
                cpu_seconds=("cpu_seconds", "mean"),
                communicated_scalars=("communicated_scalars", "mean"),
            )
            .fillna(0.0)
            .to_dict(orient="records")
        )
        summary["e4"] = grouped_e4
        for policy, macro in (
            ("cv_prediction", "EFourCvSixtyPeak"),
            ("zoh", "EFourZohSixtyPeak"),
            ("cv_fixed", "EFourCvFixedSixtyPeak"),
            ("zoh_fixed", "EFourZohFixedSixtyPeak"),
        ):
            matching = pd.DataFrame(grouped_e4)
            matching = matching[
                (matching["policy"] == policy) & (matching["duration"] == 60.0)
            ]
            if not matching.empty:
                values[macro] = f"{float(matching.iloc[0]['denial_peak_rmse']):.4f}"
        for policy, prefix in (
            ("cv_prediction", "EFourCvPredError"),
            ("zoh", "EFourZohPredError"),
        ):
            for duration, suffix in (
                (10.0, "Ten"),
                (30.0, "Thirty"),
                (60.0, "Sixty"),
            ):
                matching = pd.DataFrame(grouped_e4)
                matching = matching[
                    (matching["policy"] == policy)
                    & (matching["duration"] == duration)
                ]
                if not matching.empty:
                    values[prefix + suffix] = (
                        f"{float(matching.iloc[0]['prediction_error_peak']):.4g}"
                    )
        markdown.extend(
            ["## E4 accelerating-reference denial", "", _markdown_table(grouped_e4), ""]
        )
        generated_figures["e4_figure"] = plot_e3_denial(
            grouped_e4, output_dir / "e4_acceleration"
        )

    summary_json = output_dir / "summary.json"
    summary_markdown = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_markdown.write_text("\n".join(markdown), encoding="utf-8")
    results_values = output_dir / "results_values.tex"
    results_values.write_text(
        "\n".join(
            f"\\newcommand{{\\{name}}}{{{value}}}"
            for name, value in values.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "summary_json": summary_json,
        "summary_markdown": summary_markdown,
        "results_values": results_values,
        **generated_figures,
    }
