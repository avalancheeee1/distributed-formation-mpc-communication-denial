"""Run the fixed-weight (no confidence decay) ablation and merge it into the
canonical full-run data, then regenerate all summaries/figures.

This is the counterfactual for Theorem 3: with the edge weight held fixed
(``confidence_decay=False``) stale predictions keep their full coupling
strength, so the formation error is expected to *grow* with outage duration
instead of saturating.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from strict_admm_dmpc.experiments import run_e5_fixed_weight_ablation
from strict_admm_dmpc.reporting import summarize_experiments

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts_v2_full" / "raw" / "results.json"
OUTPUT = ROOT / "artifacts_v2_ablation"


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    started = perf_counter()
    raw_dir = OUTPUT / "raw"
    tables_dir = OUTPUT / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("[E5] fixed-weight ablation (CV + ZOH, no confidence decay)", flush=True)
    ablation = run_e5_fixed_weight_ablation(seeds=range(10), durations=(10.0, 30.0, 60.0), cap=5)

    checkpoint = raw_dir / "ablation.json"
    _write_json_atomic(checkpoint, ablation)
    print(f"[E5] done in {perf_counter() - started:.1f}s; ablation saved to {checkpoint}", flush=True)

    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    base["e3"] = [*base["e3"], *ablation["e3_fixed"]]
    base["e4"] = [*base["e4"], *ablation["e4_fixed"]]
    base["profile"] = "full+ablation"

    merged_path = raw_dir / "results.json"
    _write_json_atomic(merged_path, base)
    print(f"[merge] e3={len(base['e3'])} records, e4={len(base['e4'])} records", flush=True)

    outputs = summarize_experiments(merged_path, tables_dir)
    print("[summary] regenerated:", flush=True)
    for name, path in outputs.items():
        print(f"  {name}: {path.relative_to(OUTPUT)}", flush=True)

    # Causal check: fixed weight should GROW with duration where decayed saturates.
    print("\n60s peak formation error (decayed vs fixed):", flush=True)
    for key, policies in (
        ("e3", ("cv_prediction", "cv_fixed", "zoh", "zoh_fixed")),
        ("e4", ("cv_prediction", "cv_fixed", "zoh", "zoh_fixed")),
    ):
        recs = base[key]
        for policy in policies:
            vals = [
                r["denial_peak_rmse"]
                for r in recs
                if r["policy"] == policy and r["duration"] == 60.0
            ]
            if vals:
                mean = sum(vals) / len(vals)
                print(f"  {key} {policy:14s} 60s peak = {mean:.6f}  (n={len(vals)})", flush=True)

    print(f"\n[total] {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
