"""Run the retained serial phase order against two clean installed checkouts."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--before", required=True, type=Path)
parser.add_argument("--after", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
environment = dict(
    os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", JAX_PLATFORMS="cpu"
)
for trial, order in enumerate(
    (("before", "after"), ("after", "before"), ("before", "after"))
):
    for phase in order:
        checkout = getattr(args, phase).resolve()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=checkout,
            timeout=20,
        )
        if dirty:
            raise SystemExit(f"Tracked source is dirty: {phase}")
        destination = (args.output / f"trial-{trial}-{phase}").resolve()
        print(f"trial={trial} phase={phase} start", flush=True)
        with destination.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.module_execution_matrix",
                    "--output",
                    str(destination),
                    "--repeats",
                    "5",
                    "--iterations",
                    "30",
                    "--timeout",
                    "120",
                ],
                cwd=checkout,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=True,
            )
        print(f"trial={trial} phase={phase} completed", flush=True)
