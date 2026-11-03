"""Compare CPU disjoint-region backward with the exact baseline function."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import runpy
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = "flagquantum/simulation/statevector/local.py"
BASELINE = "0fe2e2cea541273bc371c98482a087b4cdef0313"
PROTOCOL = HERE.parent / "training_time_to_quality_20261103"
WORKER = HERE.parent / "torch_training_cpu_20261103" / "reproduce.py"


def install_baseline():
    from flagquantum.simulation.statevector import local

    original = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{SOURCE}"], cwd=ROOT, text=True, timeout=15
    )
    function = next(
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_disjoint_dense_step"
    )
    code = ast.Module(body=[function], type_ignores=[])
    exec(compile(code, SOURCE, "exec"), vars(local))


def worker(case, optimizer, route, path):
    imported = time.perf_counter()
    from flagquantum.simulation.statevector import local

    assert callable(local._apply_disjoint_dense_step)
    preloaded_seconds = time.perf_counter() - imported
    if route == "baseline":
        install_baseline()
    started = time.perf_counter()
    train = runpy.run_path(str(WORKER))["worker"]
    preloaded_seconds += time.perf_counter() - started
    train(case, optimizer, "default_eager", path, False)
    record = json.loads(path.read_text())
    record["import_seconds"] += preloaded_seconds
    path.write_text(json.dumps(record))


def compare(rows):
    paired = {(r["trial"], r["case"], r["optimizer"], r["route"]): r for r in rows}
    assert len(rows) == len(paired) == 72
    maximum = {"loss": 0.0, "parameters": 0.0, "gradients": 0.0}
    for trial in range(3):
        for case in range(6):
            for optimizer in ("SGD", "Adam"):
                a, b = (
                    paired[trial, case, optimizer, route]
                    for route in ("baseline", "candidate")
                )
                assert a["status"] == b["status"] == "passed"
                assert (
                    a["target_step"] == b["target_step"]
                    and a["target_step"] is not None
                )
                assert (
                    max(
                        a["recovery_parameter_max_error"],
                        b["recovery_parameter_max_error"],
                    )
                    <= 1e-10
                )
                for left, right in zip(a["updates"], b["updates"], strict=True):
                    assert left["ir_hash"] and right["ir_hash"]
                    maximum["loss"] = max(
                        maximum["loss"], abs(left["loss"] - right["loss"])
                    )
                    for key in ("parameters", "gradients"):
                        maximum[key] = max(
                            maximum[key],
                            *(
                                abs(x - y)
                                for x, y in zip(left[key], right[key], strict=True)
                            ),
                        )
    assert max(maximum.values()) <= 1e-10
    print("Maximum paired errors", maximum)
    for case in range(6):
        for optimizer in ("SGD", "Adam"):
            groups = [
                [paired[t, case, optimizer, r] for t in range(3)]
                for r in ("baseline", "candidate")
            ]
            values = [
                [
                    round(statistics.median(row[key] for row in group) * 1000, 3)
                    for group in groups
                ]
                for key in ("cold_target_seconds", "steady_median_seconds")
            ]
            print(case, optimizer, "target, steady in ms", values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.worker:
        case, optimizer, route = args.worker
        worker(int(case), optimizer, route, args.output)
        return
    if args.verify:
        with gzip.open(args.output, "rt") as stream:
            compare(json.load(stream)["rows"])
        return
    sys.path.insert(0, str(ROOT))
    from tools.sanitize_public_evidence import sanitize_text

    with gzip.open(PROTOCOL / "receipts.json.gz", "rt") as stream:
        thresholds = json.load(stream)["protocol"]["targets"]
    rows = []
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "worker.json"
        for trial in range(3):
            routes = (
                ("baseline", "candidate")
                if trial % 2 == 0
                else ("candidate", "baseline")
            )
            for case in range(6):
                for optimizer in ("SGD", "Adam"):
                    for route in routes:
                        path.unlink(missing_ok=True)
                        try:
                            process = subprocess.run(
                                [
                                    sys.executable,
                                    __file__,
                                    "--worker",
                                    str(case),
                                    optimizer,
                                    route,
                                    "--output",
                                    str(path),
                                ],
                                cwd=ROOT,
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=60,
                            )
                            row = (
                                json.loads(path.read_text())
                                if process.returncode == 0
                                else {
                                    "status": "failed",
                                    "log_tail": process.stderr[-4000:],
                                }
                            )
                        except subprocess.TimeoutExpired:
                            row = {"status": "timeout"}
                        threshold = thresholds[f"{case}-{optimizer}"]
                        updates = row.get("updates", [])
                        step = next(
                            (
                                i
                                for i, update in enumerate(updates, 1)
                                if update["loss"] <= threshold
                            ),
                            None,
                        )
                        row.update(
                            case=case,
                            optimizer=optimizer,
                            trial=trial,
                            route=route,
                            target=threshold,
                            target_step=step,
                            cold_target_seconds=(
                                None
                                if step is None
                                else row["import_seconds"]
                                + row["setup_seconds"]
                                + updates[step - 1]["elapsed_seconds"]
                            ),
                        )
                        rows.append(row)
                        print(
                            trial,
                            case,
                            optimizer,
                            route,
                            row["status"],
                            step,
                            flush=True,
                        )
                        packet = {
                            "baseline": BASELINE,
                            "candidate_sha256": hashlib.sha256(
                                (ROOT / SOURCE).read_bytes()
                            ).hexdigest(),
                            "harness_sha256": hashlib.sha256(
                                Path(__file__).read_bytes()
                            ).hexdigest(),
                            "rows": rows,
                        }
                        args.output.write_bytes(
                            gzip.compress(
                                sanitize_text(
                                    json.dumps(packet, separators=(",", ":"))
                                ).encode(),
                                mtime=0,
                            )
                        )
    compare(rows)


if __name__ == "__main__":
    main()
