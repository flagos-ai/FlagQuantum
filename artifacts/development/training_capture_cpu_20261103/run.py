"""Measure conditional FX regeneration on the frozen public training cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent / "torch_training_cpu_20261103"


def worker(case, route, path):
    import torch.fx

    from flagquantum.compiler import module_builder

    sys.path.insert(0, str(BASE))
    from reproduce import worker as train

    captures = []
    eliminate = torch.fx.Graph.eliminate_dead_code
    capture = module_builder.capture_module_builder
    active = None

    def observed_eliminate(graph, *args, **kwargs):
        changed = eliminate(graph, *args, **kwargs)
        if active is not None and sys._getframe(1).f_code is capture.__code__:
            active.append(changed)
            if route == "unconditional":
                return True
        return changed

    def observed_capture(*args, **kwargs):
        nonlocal active
        active = []
        start = time.perf_counter()
        try:
            return capture(*args, **kwargs)
        finally:
            captures.append(
                {
                    "seconds": time.perf_counter() - start,
                    "passes_changed": active,
                }
            )
            active = None

    torch.fx.Graph.eliminate_dead_code = observed_eliminate
    module_builder.capture_module_builder = observed_capture
    train(case, "Adam", "default_eager", path, False, measure_invalidation=True)
    record = json.loads(path.read_text())
    record["captures"] = captures
    record["comparison_route"] = route
    path.write_text(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", nargs=2)
    args = parser.parse_args()
    if args.worker:
        case, route = args.worker
        worker(int(case), route, args.output)
        return
    sys.path.insert(0, str(ROOT))
    from tools.sanitize_public_evidence import sanitize_text

    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        for trial in range(5):
            for case in (0, 3, 5):
                routes = ("unconditional", "conditional")
                for route in routes if trial % 2 == 0 else reversed(routes):
                    path = Path(temporary) / "receipt.json"
                    path.unlink(missing_ok=True)
                    try:
                        run = subprocess.run(
                            [
                                sys.executable,
                                __file__,
                                "--worker",
                                str(case),
                                route,
                                "--output",
                                str(path),
                            ],
                            cwd=ROOT,
                            env={
                                **os.environ,
                                "PYTHONPATH": str(ROOT),
                                "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1",
                            },
                            timeout=60,
                            capture_output=True,
                            text=True,
                        )
                        row = (
                            json.loads(path.read_text())
                            if run.returncode == 0
                            else {"status": "failed", "log_tail": run.stderr[-4000:]}
                        )
                    except subprocess.TimeoutExpired:
                        row = {"status": "timeout"}
                    row.update(trial=trial, case=case, comparison_route=route)
                    if row["status"] == "passed":
                        updates = row.pop("updates")
                        row["trajectory_sha256"] = hashlib.sha256(
                            json.dumps(
                                [
                                    (
                                        update["loss"],
                                        update["parameters"],
                                        update["gradients"],
                                        update["ir_hash"],
                                    )
                                    for update in updates
                                ],
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()
                    rows.append(row)
                    args.output.write_text(
                        sanitize_text(
                            json.dumps(
                                {
                                    "compiler_sha256": hashlib.sha256(
                                        (
                                            ROOT
                                            / "flagquantum/compiler/module_builder.py"
                                        ).read_bytes()
                                    ).hexdigest(),
                                    "harness_sha256": hashlib.sha256(
                                        Path(__file__).read_bytes()
                                    ).hexdigest(),
                                    "rows": rows,
                                },
                                indent=2,
                            )
                        )
                        + "\n"
                    )
                    print(trial, case, route, row["status"], flush=True)


if __name__ == "__main__":
    main()
