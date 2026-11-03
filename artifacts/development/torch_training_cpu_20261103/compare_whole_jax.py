"""Exact held-out whole-TensorCircuit/JAX training; a separate interface category."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import signal
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


def worker(case: int, kind: str, output: Path) -> None:
    imported = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tensorcircuit as tc
    import torch
    from jax.example_libraries import optimizers

    imports = time.perf_counter() - imported
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configuration: Any = jax.config
    configuration.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    width, depth, batch, layout = CASES[case]
    generator = torch.Generator().manual_seed(20261102)
    initial = torch.randn(2 * width, generator=generator, dtype=torch.float64) * 0.2
    features = torch.randn(batch, width, generator=generator, dtype=torch.float64) * 0.3
    target = jnp.asarray(
        torch.linspace(-0.2, 0.4, batch * width, dtype=torch.float64)
        .reshape(batch, width)
        .numpy()
    )
    weights = jnp.asarray(
        torch.linspace(-0.7, 1.3, batch * width, dtype=torch.float64)
        .reshape(batch, width)
        .numpy()
    )
    init, update, get = optimizers.sgd(0.01) if kind == "SGD" else optimizers.adam(0.01)
    state = init(jnp.asarray(initial.numpy()))
    base = jnp.asarray(features.numpy())

    def quantum(p: Any, x: Any) -> Any:
        circuit = tc.Circuit(width)
        for layer in range(depth):
            for wire in range(width):
                circuit.ry(
                    wire,
                    theta=p[wire] ** 2
                    + p[width + wire] * jnp.sin(x[wire])
                    + layer * 0.01,
                )
                circuit.rz(wire, theta=p[width + wire] * 0.3 + jnp.cos(p[wire]))
            for wire in range(width - 1):
                circuit.cnot(wire, wire + 1)
            if layout == "ring":
                circuit.cnot(width - 1, 0)
        return jnp.stack(
            [
                jnp.real(circuit.expectation((tc.gates.z(), [wire])))
                for wire in range(width)
            ]
        )

    def loss(p: Any, x: Any) -> Any:
        return jnp.mean(
            ((jax.vmap(lambda row: quantum(p, row))(x) - target) * weights) ** 2
        )

    @jax.jit
    def step(index: Any, current: Any, x: Any) -> Any:
        value, gradient = jax.value_and_grad(loss)(get(current), x)
        return value, gradient, update(index, gradient, current)

    record: dict[str, Any] = {
        "schema": "flagquantum.whole_jax_training_control.v1",
        "case": case,
        "shape": CASES[case],
        "optimizer": kind,
        "route": "tensorcircuit_whole_jax",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True, timeout=15
            ).strip()
        ),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "jax": jax.__version__,
        "tensorcircuit": tc.__version__,
        "dtype": "complex128/float64",
        "device": "cpu",
        "interface": "whole-JAX objective/AD/optimizer; not a Torch interface",
        "seed": 20261102,
        "steps": 60,
        "checkpoint_step": 30,
        "import_seconds": imports,
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "updates": [],
        "status": "running",
    }
    output.write_text(json.dumps(record))
    record["setup_seconds"] = time.perf_counter() - imported - imports
    intervals = []
    checkpoint = None
    try:
        started = time.perf_counter()
        previous = started
        for index in range(60):
            value, gradient, state = step(index, state, base + index * 0.0005)
            get(state).block_until_ready()
            now = time.perf_counter()
            intervals.append(now - previous)
            record["updates"].append(
                {
                    "loss": float(value),
                    "parameters": np.asarray(get(state)).tolist(),
                    "gradients": np.asarray(gradient).tolist(),
                }
            )
            if index == 29:
                checkpoint_start = time.perf_counter()
                checkpoint = pickle.dumps(state)
                record["checkpoint_seconds"] = time.perf_counter() - checkpoint_start
                record["checkpoint_sha256"] = hashlib.sha256(checkpoint).hexdigest()
                output.write_text(json.dumps(record))
            previous = now
        record.update(
            training_seconds=time.perf_counter() - started,
            first_update_seconds=intervals[0],
            per_update_seconds=intervals,
            steady_median_seconds=float(np.median(intervals[10:])),
            final_parameters=np.asarray(get(state)).tolist(),
            losses=[x["loss"] for x in record["updates"]],
        )
        assert checkpoint is not None
        replay = pickle.loads(checkpoint)
        started = time.perf_counter()
        for index in range(30, 60):
            _, _, replay = step(index, replay, base + index * 0.0005)
        get(replay).block_until_ready()
        record["recovery_seconds"] = time.perf_counter() - started
        error = float(np.max(np.abs(np.asarray(get(replay)) - np.asarray(get(state)))))
        record["recovery_parameter_max_error"] = error
        if error > 1e-10:
            raise AssertionError("checkpoint replay differs")
        record["status"] = "passed"
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["peak_worker_rss_bytes"] = (
            rss if sys.platform == "darwin" else rss * 1024
        )
        if record["peak_worker_rss_bytes"] > 2 * 1024**3:
            record["status"] = "memory_budget_exceeded"
        output.write_text(json.dumps(record, indent=2) + "\n")


def campaign(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for trial in range(3):
        for case in range(6):
            for kind in ("SGD", "Adam"):
                name = f"trial{trial}-case{case}-{kind}"
                receipt = output / f"{name}.json"
                with (
                    (output / f"{name}.txt").open("w") as stream,
                    tempfile.TemporaryDirectory(prefix="fq936-jax-cache-") as cache,
                ):
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            str(case),
                            "--optimizer",
                            kind,
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
                            "JAX_PLATFORMS": "cpu",
                            "JAX_ENABLE_COMPILATION_CACHE": "false",
                            "JAX_COMPILATION_CACHE_DIR": cache,
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                        },
                    )
                    timed_out = False
                    try:
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                record = json.loads(receipt.read_text()) if receipt.exists() else {}
                record.update(
                    trial=trial, case=case, optimizer=kind, exit_code=process.returncode
                )
                if timed_out or process.returncode:
                    record["status"] = "timeout" if timed_out else "process_failed"
                receipt.write_text(json.dumps(record, indent=2) + "\n")
                records.append(record)
                (output / "campaign.json").write_text(
                    json.dumps(
                        {
                            "records": records,
                            "timeout_seconds": 120,
                            "rss_budget_bytes": 2 * 1024**3,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    name, record["status"], record.get("training_seconds"), flush=True
                )
    return int(any(record["status"] != "passed" for record in records))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=range(6))
    parser.add_argument("--optimizer", choices=("SGD", "Adam"))
    args = parser.parse_args()
    if args.worker is None:
        return campaign(args.output)
    worker(args.worker, args.optimizer, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
