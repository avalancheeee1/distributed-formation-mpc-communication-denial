"""Command-line runner for strict ADMM-DMPC v2 experiments."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

from strict_admm_dmpc.experiments import (
    run_e0_correctness,
    run_e1_convergence,
    run_e2_budget,
    run_e3_denial,
    run_e4_acceleration,
)
from strict_admm_dmpc.reporting import summarize_experiments


ROOT = Path(__file__).resolve().parent


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    sources = sorted((ROOT / "strict_admm_dmpc").glob("*.py")) + [
        ROOT / "experiments_v2.py",
        ROOT / "requirements.txt",
        ROOT / "requirements-lock.txt",
    ]
    for path in sources:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts_v2")
    args = parser.parse_args()
    seeds = range(3) if args.profile == "pilot" else range(10)
    output = args.output
    raw_dir = output / "raw"
    tables_dir = output / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    checkpoint = raw_dir / "partial_results.json"
    print("[E0] centralized equivalence gate", flush=True)
    e0 = run_e0_correctness(seeds)
    if not all(record["passed"] for record in e0):
        raise SystemExit("E0 correctness gate failed")
    partial: dict[str, object] = {"profile": args.profile, "e0": e0}
    _write_json_atomic(checkpoint, partial)
    print("[E1] convergence sweep", flush=True)
    e1 = run_e1_convergence(seeds)
    partial["e1"] = e1
    _write_json_atomic(checkpoint, partial)
    print("[E2] closed-loop iteration budget", flush=True)
    e2 = run_e2_budget(seeds)
    partial["e2"] = e2
    _write_json_atomic(checkpoint, partial)
    print("[E3] communication-denial policies", flush=True)
    e3 = run_e3_denial(seeds)
    partial["e3"] = e3
    _write_json_atomic(checkpoint, partial)
    print("[E4] accelerating-reference denial policies", flush=True)
    e4 = run_e4_acceleration(seeds)
    raw = {
        "profile": args.profile,
        "e0": e0,
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
    }
    raw_path = raw_dir / "results.json"
    _write_json_atomic(raw_path, raw)
    outputs = summarize_experiments(raw_path, tables_dir)
    manifest = {
        "profile": args.profile,
        "seeds": list(seeds),
        "python": sys.version,
        "platform": platform.platform(),
        "direct_dependency_versions": {
            package: version(package)
            for package in ("numpy", "scipy", "osqp", "matplotlib", "pandas")
        },
        "source_sha256": _source_sha256(),
        "raw_sha256": _sha256(raw_path),
        "outputs": {name: str(path.relative_to(output)) for name, path in outputs.items()},
        "artifacts_sha256": {
            str(path.relative_to(output)): _sha256(path)
            for path in (raw_path, *outputs.values())
        },
        "git_hash": None,
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
