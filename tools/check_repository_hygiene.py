#!/usr/bin/env python3
"""Reject generated, oversized, dataset and model artifacts in Git."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".parquet",
    ".onnx",
    ".ckpt",
    ".safetensors",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
}
ALLOWED_LARGE_FILES: dict[Path, int] = {}
DEFAULT_MAX_FILE_BYTES = 2_000_000
# The integrated MPS/statevector evidence corpus is part of the reproducibility
# contract and is referenced by benchmark-contract tests and public reports.
# Keep a bounded repository-wide budget while retaining the stricter per-file
# and forbidden-artifact checks above.
DEFAULT_MAX_TOTAL_BYTES = 110_000_000
CANONICAL_RESULT_DIRECTORIES = {"comparison", "local", "scalability", "smoke"}
RESULT_CONTAINER_DIRECTORIES = {"legacy"}
LEGACY_RESULT_DIRECTORIES = {
    "adapt_vqe_tn_20260803",
    "backend_phase_diagram_20260730",
    "distributed_selector_calibrations",
    "distributed_tn_patent",
    "gradient_backend_phase_diagram_20260730",
    "mps_capacity_16xa800_20260806",
    "statevector_figure_book",
    "statevector_mlsys_current",
    "statevector_submission_figures",
    "tn_compare_tcng_20260731",
    "tn_complex_scaling_20260802",
    "tn_cost_aware_scaling_20260730",
    "tn_cotengra_gap",
    "tn_gradient_capacity_scaling_20260730",
    "tn_hard_gate_20260731",
    "tn_high_width_scaling_20260730",
    "tn_high_width_working_set_scaling_20260730",
    "tn_kiloqubit_ladder_20260730",
    "tn_mps_crossover_20260730",
    "tn_persistent_cache_20260730",
    "tn_persistent_cache_v2_20260730",
    "tn_readme_scaling_20260731",
    "tn_shared_dag_scaling_20260730",
    "tn_sparse_capacity_20260730",
    "tn_working_set_calibration_20260731",
    "vqe_backend_switch",
}
LEGACY_ARTIFACT_DIRECTORIES = {
    "quafu_context_batch_01",
    "quafu_context_batch_02",
    "quafu_context_batch_03",
    "quafu_predictive_batch_01",
    "quafu_predictive_batch_02",
    "quafu_scale_batch_01",
    "quafu_vqe",
    "quafu_vqe_10240",
}
ARTIFACT_CONTAINER_DIRECTORIES = {"development", "legacy"}


def tracked_files(root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=root,
        capture_output=True,
    )
    if completed.returncode != 0:
        git_dir = root
        while git_dir != git_dir.parent and not (git_dir / ".git").exists():
            git_dir = git_dir.parent
        metadata = git_dir / ".git"
        if not metadata.exists() or (
            metadata.is_dir() and not (metadata / "HEAD").exists()
        ):
            return ()
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    output = completed.stdout
    return tuple(root / item.decode() for item in output.split(b"\0") if item)


def violations(
    root: Path, *, max_file_bytes: int, max_total_bytes: int
) -> tuple[str, ...]:
    errors = []
    total = 0
    for path in tracked_files(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        size = path.stat().st_size
        total += size
        if FORBIDDEN_PARTS.intersection(relative.parts):
            errors.append(f"generated cache tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/data artifact tracked: {relative}")
        allowed_size = ALLOWED_LARGE_FILES.get(relative, max_file_bytes)
        if size > allowed_size:
            errors.append(f"tracked file exceeds {max_file_bytes} bytes: {relative}")
        if path.read_bytes()[:43] == b"version https://git-lfs.github.com/spec/v1":
            errors.append(f"Git-LFS pointer is not allowed in source: {relative}")
    if total > max_total_bytes:
        errors.append(f"tracked tree is {total} bytes; limit is {max_total_bytes}")
    return tuple(errors)


def layout_violations(root: Path) -> tuple[str, ...]:
    errors = [
        f"capability contract must live under contracts/: {path.name}"
        for path in sorted(root.glob("*-contract.toml"))
    ]
    result_directories = {
        path.name
        for path in (root / "benchmarks" / "results").iterdir()
        if path.is_dir()
    }
    for name in sorted(
        result_directories - CANONICAL_RESULT_DIRECTORIES - RESULT_CONTAINER_DIRECTORIES
    ):
        errors.append(f"new top-level benchmark result family is forbidden: {name}")
    legacy_root = root / "benchmarks" / "results" / "legacy"
    legacy_directories = {path.name for path in legacy_root.iterdir() if path.is_dir()}
    for name in sorted(legacy_directories - LEGACY_RESULT_DIRECTORIES):
        errors.append(f"new legacy benchmark result family is forbidden: {name}")
    artifact_directories = {
        path.name for path in (root / "artifacts").iterdir() if path.is_dir()
    }
    for name in sorted(artifact_directories - ARTIFACT_CONTAINER_DIRECTORIES):
        errors.append(f"new top-level artifact family is forbidden: {name}")
    legacy_artifact_root = root / "artifacts" / "legacy"
    legacy_artifact_directories = {
        path.name for path in legacy_artifact_root.iterdir() if path.is_dir()
    }
    for name in sorted(legacy_artifact_directories - LEGACY_ARTIFACT_DIRECTORIES):
        errors.append(f"new legacy artifact batch is forbidden: {name}")
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = violations(
        root,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
    ) + layout_violations(root)
    if errors:
        print("\n".join(errors))
        return 1
    print("repository hygiene passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
