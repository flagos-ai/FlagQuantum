"""Retain fresh-process fq.train candidates, including cold losses and refusals.

This diagnostic forces a private, unpromoted candidate to falsify it. It does not
represent the default route or satisfy #936's whole-training comparison alone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

CASES = (
    (3, 2, 1, "chain"),
    (3, 4, 8, "ring"),
    (5, 2, 8, "ring"),
    (5, 4, 64, "chain"),
    (7, 2, 64, "ring"),
    (7, 4, 1, "chain"),
)
ROOT = Path(__file__).resolve().parents[3]


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, timeout=15
    ).strip()


def worker(case: int, kind: str, route: str, output: Path) -> None:
    imported = time.perf_counter()
    import torch

    import flagquantum as fq
    from flagquantum.simulation.native_cpu.adjoint import native_cpu_adjoint_available

    imports = time.perf_counter() - imported
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    training = importlib.import_module("flagquantum.runtime.training")
    training._MIN_COMPILED_STEPS = 0 if route == "forced_candidate" else None
    width, depth, batch, layout = CASES[case]
    generator = torch.Generator().manual_seed(20261102)
    initial = torch.randn(2 * width, generator=generator, dtype=torch.float64) * 0.2
    features = torch.randn(batch, width, generator=generator, dtype=torch.float64) * 0.3
    target = torch.linspace(-0.2, 0.4, batch * width, dtype=torch.float64).reshape(
        batch, width
    )
    weights = torch.linspace(-0.7, 1.3, batch * width, dtype=torch.float64).reshape(
        batch, width
    )

    def build(p: Any, x: Any) -> Any:
        circuit = fq.Circuit(width, bsz=x.shape[0], dtype=torch.complex128)
        for layer in range(depth):
            for wire in range(width):
                circuit.ry(
                    wire,
                    p[wire].square()
                    + p[width + wire] * x[:, wire].sin()
                    + layer * 0.01,
                )
                circuit.rz(wire, p[width + wire] * 0.3 + p[wire].cos())
            for wire in range(width - 1):
                circuit.cx(wire, wire + 1)
            if layout == "ring":
                circuit.cx(width - 1, 0)
        return circuit

    def model() -> Any:
        return fq.Module(
            build,
            2 * width,
            dtype=torch.float64,
            init=initial,
            policy=fq.RuntimePolicy(observable_qubits=tuple(range(width))),
        )

    def objective(value: Any) -> Any:
        return ((value - target) * weights).square().mean()

    def inputs(index: int) -> Any:
        return features + index * 0.0005

    record: dict[str, Any] = {
        "schema": "flagquantum.torch_training_candidate.v1",
        "case": case,
        "shape": CASES[case],
        "optimizer": kind,
        "route": route,
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain")),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "native_cpu_adjoint_available": native_cpu_adjoint_available(),
        "effective_derivative": "Torch backprop or torch.func VJP, not whole-circuit native adjoint",
        "seed": 20261102,
        "dtype": "complex128/float64",
        "device": "cpu",
        "steps": 60,
        "checkpoint_step": 30,
        "import_seconds": imports,
        "torchinductor_cache": "isolated empty directory for each process",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "status": "running",
        "updates": [],
    }
    output.write_text(json.dumps(record))
    with tempfile.TemporaryDirectory(prefix="fq936-checkpoint-") as temporary:
        current = model()
        optimizer = getattr(torch.optim, kind)(current.parameters(), lr=0.01)
        record["setup_seconds"] = time.perf_counter() - imported - imports
        previous = time.perf_counter()
        intervals: list[float] = []
        checkpoint_seconds = 0.0

        def callback(step: int, loss: float, execution: Any) -> None:
            nonlocal previous, checkpoint_seconds
            now = time.perf_counter()
            intervals.append(now - previous)
            record["updates"].append(
                {
                    "loss": loss,
                    "parameters": current.parameters_tensor.detach().tolist(),
                    "gradients": current.parameters_tensor.grad.detach().tolist(),
                    "ir_hash": execution.provenance["ir_hash"],
                }
            )
            if step == 30:
                started = time.perf_counter()
                current.save_checkpoint(
                    Path(temporary) / "step30.pt",
                    optimizer=optimizer,
                    step=30,
                    seed=20261102,
                )
                checkpoint_seconds = time.perf_counter() - started
                # Preserve partial receipts if a later compilation times out.
                output.write_text(json.dumps(record))
            previous = now

        try:
            started = time.perf_counter()
            result = fq.train(
                current,
                optimizer=optimizer,
                objective=objective,
                inputs=inputs,
                steps=60,
                callback=callback,
            )
            record.update(
                training_seconds=time.perf_counter() - started,
                first_update_seconds=intervals[0],
                per_update_seconds=intervals,
                steady_median_seconds=statistics.median(intervals[10:]),
                checkpoint_seconds=checkpoint_seconds,
                diagnostics=dict(result.last_execution.runtime).get(
                    "training_compilation"
                ),
                final_parameters=current.parameters_tensor.detach().tolist(),
                losses=result.losses,
                final_optimizer={
                    str(key): {
                        name: (
                            value.detach().tolist()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for name, value in values.items()
                    }
                    for key, values in optimizer.state_dict()["state"].items()
                },
            )
            restored = model()
            restored_optimizer = getattr(torch.optim, kind)(
                restored.parameters(), lr=0.01
            )
            started = time.perf_counter()
            restored(inputs(29))
            restored.load_checkpoint(
                Path(temporary) / "step30.pt", optimizer=restored_optimizer
            )
            replay = fq.train(
                restored,
                optimizer=restored_optimizer,
                objective=objective,
                inputs=lambda index: inputs(index + 30),
                steps=30,
            )
            record["recovery_seconds"] = time.perf_counter() - started
            torch.testing.assert_close(
                restored.parameters_tensor,
                current.parameters_tensor,
                atol=1e-10,
                rtol=1e-10,
            )
            torch.testing.assert_close(
                replay.losses, result.losses[30:], atol=1e-10, rtol=1e-10
            )
            torch.testing.assert_close(
                replay.last_execution.state,
                result.last_execution.state,
                atol=1e-10,
                rtol=1e-10,
            )
            torch.testing.assert_close(
                restored_optimizer.state_dict(),
                optimizer.state_dict(),
                atol=1e-10,
                rtol=1e-10,
            )
            assert (
                replay.last_execution.provenance["ir_hash"]
                == result.last_execution.provenance["ir_hash"]
            )
            record.update(
                status="passed",
                recovery_parameter_max_error=float(
                    (restored.parameters_tensor - current.parameters_tensor)
                    .abs()
                    .max()
                    .detach()
                ),
            )
        except Exception:
            record.update(status="failed", error=traceback.format_exc())
        finally:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            record["peak_worker_rss_bytes"] = (
                rss if sys.platform == "darwin" else rss * 1024
            )
            if record["peak_worker_rss_bytes"] > 2 * 1024**3:
                record.update(status="memory_budget_exceeded")
            output.write_text(json.dumps(record, indent=2) + "\n")


def campaign(output: Path, trials: int) -> int:
    if git("status", "--porcelain"):
        raise RuntimeError("measure a committed clean source tree")
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for trial in range(trials):
        routes = ("default_eager", "forced_candidate")
        if trial % 2:
            routes = tuple(reversed(routes))
        for case in range(len(CASES)):
            for kind in ("SGD", "Adam"):
                for route in routes:
                    name = f"trial{trial}-case{case}-{kind}-{route}"
                    receipt, log = output / f"{name}.json", output / f"{name}.txt"
                    with (
                        tempfile.TemporaryDirectory(prefix="fq936-inductor-") as cache,
                        log.open("w") as stream,
                    ):
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                str(case),
                                "--optimizer",
                                kind,
                                "--route",
                                route,
                                "--output",
                                str(receipt),
                            ],
                            cwd=ROOT,
                            start_new_session=True,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            env={
                                **os.environ,
                                "PYTHONPATH": str(ROOT),
                                "TORCHINDUCTOR_CACHE_DIR": cache,
                                "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1",
                            },
                        )
                        timeout = False
                        try:
                            process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            timeout = True
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=5)
                    record = json.loads(receipt.read_text()) if receipt.exists() else {}
                    record.update(
                        trial=trial,
                        case=case,
                        optimizer=kind,
                        route=route,
                        exit_code=process.returncode,
                    )
                    if timeout or process.returncode:
                        record.update(status="timeout" if timeout else "process_failed")
                    receipt.write_text(json.dumps(record, indent=2) + "\n")
                    records.append(record)
                    print(
                        name,
                        record.get("status"),
                        record.get("training_seconds"),
                        flush=True,
                    )
                    (output / "campaign.json").write_text(
                        json.dumps(
                            {
                                "source_commit": git("rev-parse", "HEAD"),
                                "trials": trials,
                                "timeout_seconds": 120,
                                "rss_budget_bytes": 2 * 1024**3,
                                "records": records,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
    return int(any(record.get("status") != "passed" for record in records))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=range(6))
    parser.add_argument("--optimizer", choices=("SGD", "Adam"))
    parser.add_argument("--route", choices=("default_eager", "forced_candidate"))
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.optimizer, args.route, args.output)
        return 0
    return campaign(args.output, args.trials)


if __name__ == "__main__":
    raise SystemExit(main())
