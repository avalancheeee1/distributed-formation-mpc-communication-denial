"""Regenerate the SHA-256 manifest for the frozen canonical data directory.

The manuscript's canonical data lives in ``artifacts_v2_ablation/``: the full
E0--E4 run merged with the E5 fixed-weight ablation, plus every processed table
and figure.  ``experiments_v2.py`` writes a manifest for its own output
directory, but ``run_e5_ablation.py`` merges E5 into that data without re-emitting
one, so the canonical directory is left without the hash record the manuscript
promises ("source/artifact SHA-256 manifests").

This script closes that gap *without re-running any experiment*.  It hashes the
frozen raw records and processed artifacts together with the current source, so a
reader can (i) confirm the committed files are byte-identical to what the paper
cites, and (ii) pin the exact source set those files were produced from.

It is read-only over the data directory and only writes
``artifacts_v2_ablation/manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts_v2_ablation"

# The reproducibility surface for the full + ablation pipeline.  Order is
# significant (it mirrors ``experiments_v2.py``: a path prefix followed by the
# file bytes is fed into the digest).
SOURCE_FILES = sorted((ROOT / "strict_admm_dmpc").glob("*.py")) + [
    ROOT / "experiments_v2.py",
    ROOT / "run_e5_ablation.py",
    ROOT / "export_ablation_table.py",
    ROOT / "requirements.txt",
    ROOT / "requirements-lock.txt",
]

RAW_FILES = [
    OUTPUT / "raw" / "results.json",
    OUTPUT / "raw" / "ablation.json",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_FILES:
        if not path.exists():
            raise FileNotFoundError(f"source file missing: {path}")
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "osqp", "matplotlib", "pandas"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def main() -> None:
    tables_dir = OUTPUT / "tables"
    artifacts = [*RAW_FILES, *sorted(tables_dir.glob("*"))]

    manifest = {
        "profile": "full+ablation",
        "seeds": list(range(10)),
        "python": sys.version,
        "platform": platform.platform(),
        "direct_dependency_versions": _dependency_versions(),
        "source_sha256": _source_sha256(),
        "source_files": [str(p.relative_to(ROOT)) for p in SOURCE_FILES],
        "raw_sha256": {
            str(p.relative_to(OUTPUT)): _sha256(p) for p in RAW_FILES
        },
        "artifacts_sha256": {
            str(p.relative_to(OUTPUT)): _sha256(p) for p in artifacts
        },
        "git_hash": None,
    }

    out = OUTPUT / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
