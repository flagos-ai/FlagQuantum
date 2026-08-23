"""Run a resumable single-node TN memory calibration matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-sizes", nargs="+", type=int, required=True)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("complex64", "complex128"),
        required=True,
    )
    parser.add_argument(
        "--budgets-gib",
        nargs="+",
        type=float,
        default=(8.0, 16.0, 32.0),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-run-timeout", type=int, default=180)
    parser.add_argument("--calibration-dir", type=Path)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for dtype in arguments.dtypes:
        for world_size in arguments.world_sizes:
            for budget in arguments.budgets_gib:
                budget_name = f"{budget:g}".replace(".", "p")
                output = arguments.output_dir / (
                    f"a800_{dtype}_grid4x9_c4_ws{world_size}"
                    f"_budget{budget_name}"
                    f"{'_calibrated' if arguments.calibration_dir else ''}.json"
                )
                if output.is_file():
                    try:
                        existing = json.loads(output.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        existing = {}
                    if existing.get("status") == "measured_success":
                        print(
                            json.dumps(
                                {"event": "skip_completed", "output": str(output)}
                            ),
                            flush=True,
                        )
                        continue
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={world_size}",
                    "benchmarks/tn_sparse_capacity_case.py",
                    "--qubits",
                    "36",
                    "--grid-rows",
                    "4",
                    "--grid-cols",
                    "9",
                    "--depth",
                    "4",
                    "--targets",
                    "1",
                    "--warmup",
                    "0",
                    "--iterations",
                    "1",
                    "--dtype",
                    dtype,
                    "--max-working-set-gib",
                    str(budget),
                    "--working-set-safety-factor",
                    "4",
                    "--output",
                    str(output),
                ]
                if arguments.calibration_dir is not None:
                    command.extend(
                        (
                            "--memory-calibration",
                            str(
                                arguments.calibration_dir
                                / f"calibration_a800_{dtype}_ws{world_size}.json"
                            ),
                        )
                    )
                print(
                    json.dumps(
                        {
                            "event": "start",
                            "world_size": world_size,
                            "dtype": dtype,
                            "budget_gib": budget,
                            "output": str(output),
                        }
                    ),
                    flush=True,
                )
                started = time.monotonic()
                try:
                    completed = subprocess.run(
                        command,
                        check=False,
                        timeout=arguments.per_run_timeout,
                    )
                    returncode = completed.returncode
                except subprocess.TimeoutExpired:
                    returncode = 124
                print(
                    json.dumps(
                        {
                            "event": "finish",
                            "world_size": world_size,
                            "dtype": dtype,
                            "budget_gib": budget,
                            "returncode": returncode,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
                if returncode:
                    failures.append(
                        {
                            "world_size": world_size,
                            "dtype": dtype,
                            "budget_gib": budget,
                            "returncode": returncode,
                        }
                    )
    if failures:
        raise SystemExit(
            "TN calibration matrix failures: " + json.dumps(failures)
        )


if __name__ == "__main__":
    main()
