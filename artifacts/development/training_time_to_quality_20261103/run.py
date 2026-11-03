"""Repeat the frozen shape matrix with paired cold quality and shape probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent / "torch_training_cpu_20261103"
ROUTES = ("default_eager", "forced_candidate", "tensorcircuit_whole_jax")
OPTIMIZERS = ("SGD", "Adam")


def freeze_targets():
    """Fix targets from the already published eager campaign."""
    history = {}
    for suffix in ("summary", "workers"):
        with tarfile.open(BASE / f"callback-{suffix}.tar.gz") as archive:
            for member in archive.getmembers():
                if member.name.endswith("-default_eager.json"):
                    stream = archive.extractfile(member)
                    assert stream is not None
                    history[Path(member.name).name] = json.load(stream)["losses"]
    targets = {}
    for case in range(6):
        for optimizer in OPTIMIZERS:
            samples = [
                history[f"trial{trial}-case{case}-{optimizer}-default_eager.json"]
                for trial in range(3)
            ]
            start = statistics.median(losses[0] for losses in samples)
            end = statistics.median(losses[-1] for losses in samples)
            targets[f"{case}-{optimizer}"] = start + 0.9 * (end - start)
    return targets


def run_worker(args):
    sys.path.insert(0, str(BASE))
    if args.route == "tensorcircuit_whole_jax":
        from compare_whole_jax import worker

        worker(args.case, args.optimizer, args.output, measure_invalidation=True)
    else:
        from reproduce import worker

        worker(
            args.case,
            args.optimizer,
            args.route,
            args.output,
            False,
            measure_invalidation=True,
        )


def run_case(case, optimizer, route, path, cache):
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "TORCHINDUCTOR_CACHE_DIR": cache,
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with tempfile.TemporaryFile(mode="w+t") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--worker",
                "--case",
                str(case),
                "--optimizer",
                optimizer,
                "--route",
                route,
                "--output",
                str(path),
            ],
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.seek(0)
        diagnostics = log.read()[-5000:]
    row = json.loads(path.read_text()) if path.exists() else {}
    if timed_out or process.returncode:
        row["status"] = "timeout" if timed_out else "process_failed"
    if row.get("status") != "passed":
        row["log_tail"] = diagnostics
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", type=int, choices=range(6))
    parser.add_argument("--optimizer", choices=OPTIMIZERS)
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return 0
    sys.path.insert(0, str(ROOT))
    from tools.sanitize_public_evidence import sanitize_text

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets = freeze_targets()
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "targets": targets,
                "target_definition": "90% of historical eager first-to-final training loss improvement",
                "seed": 20261102,
                "trials": args.trials,
                "steps": 60,
                "timeout_seconds": 120,
                "producer_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path(__file__),
                        BASE / "reproduce.py",
                        BASE / "compare_whole_jax.py",
                        BASE / "callback-summary.tar.gz",
                        BASE / "callback-workers.tar.gz",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    for trial in range(args.trials):
        routes = ROUTES[trial % len(ROUTES) :] + ROUTES[: trial % len(ROUTES)]
        for case in range(6):
            for optimizer in OPTIMIZERS:
                for route in routes:
                    name = f"trial{trial}-case{case}-{optimizer}-{route}"
                    with tempfile.TemporaryDirectory() as directory:
                        row = run_case(
                            case,
                            optimizer,
                            route,
                            Path(directory) / "worker.json",
                            directory,
                        )
                    threshold = targets[f"{case}-{optimizer}"]
                    updates = row.get("updates", [])
                    hit = next(
                        (
                            i
                            for i, update in enumerate(updates, 1)
                            if update["loss"] <= threshold
                        ),
                        None,
                    )
                    cold = row.get("import_seconds", 0) + row.get("setup_seconds", 0)
                    row.update(
                        trial=trial,
                        case=case,
                        optimizer=optimizer,
                        route=route,
                        target=threshold,
                        target_step=hit,
                        cold_target_seconds=(
                            None
                            if hit is None
                            else cold + updates[hit - 1]["elapsed_seconds"]
                        ),
                    )
                    (output / f"{name}.json").write_text(
                        sanitize_text(json.dumps(row, indent=2)) + "\n"
                    )
                    print(name, row.get("status"), hit, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
