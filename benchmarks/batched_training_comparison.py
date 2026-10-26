"""Matched public PyTorch training and whole-JAX comparator on exact CPU circuits.

Run each shape in a fresh process. JSON retains every candidate, including losses
and failures; whole-JAX timings are a different interface category.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def worker(width: int, batch: int, depth: int, rotation: int = 0) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tensorcircuit as tc
    import torch

    import flagquantum as fq

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configuration: Any = jax.config
    configuration.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    initial = torch.linspace(-0.2, 0.3, 2 * width * depth, dtype=torch.float64)
    features = torch.linspace(-0.4, 0.5, batch * width, dtype=torch.float64).reshape(
        batch, width
    )

    def build(parameters: Any, inputs: Any) -> Any:
        circuit = fq.Circuit(width, bsz=inputs.shape[0], dtype=torch.complex128)
        for layer in range(depth):
            for wire in range(width):
                offset = 2 * (layer * width + wire)
                circuit.ry(wire, parameters[offset] + inputs[:, wire])
                circuit.rz(wire, parameters[offset + 1])
            for wire in range(width - 1):
                circuit.cx(wire, wire + 1)
        return circuit

    def tc_value(parameters: Any, inputs: Any, observable: str) -> Any:
        circuit = tc.Circuit(width)
        for layer in range(depth):
            for wire in range(width):
                offset = 2 * (layer * width + wire)
                circuit.ry(wire, theta=parameters[offset] + inputs[wire])
                circuit.rz(wire, theta=parameters[offset + 1])
            for wire in range(width - 1):
                circuit.cnot(wire, wire + 1)
        if observable == "state":
            indices = jnp.arange(2**width)
            weights = sum(
                1 - 2 * ((indices >> (width - 1 - wire)) & 1) for wire in range(width)
            )
            return jnp.sum(jnp.abs(circuit.state()) ** 2 * weights) / width
        identity = jnp.eye(2, dtype=jnp.complex128)
        z = jnp.diag(jnp.array([1.0, -1.0])).astype(jnp.complex128)
        zero = jnp.zeros_like(identity)
        nodes: list[Any] = []
        for wire in range(width):
            matrix = (
                jnp.stack((z, identity))
                if wire == 0
                else (
                    jnp.stack((identity, z))
                    if wire == width - 1
                    else jnp.stack(
                        (jnp.stack((identity, zero)), jnp.stack((z, identity)))
                    )
                )
            )
            node = tc.gates.Gate(matrix)
            if wire:
                nodes[-1][0 if wire == 1 else 1] ^ node[0]
            nodes.append(node)
        mpo = tc.quantum.QuOperator(
            [node[-2] for node in nodes], [node[-1] for node in nodes]
        )
        return tc.templates.measurements.mpo_expectation(circuit, mpo) / width

    records: list[dict[str, Any]] = []
    oracle_value: Any = None
    oracle_gradient: Any = None
    routes: tuple[str, ...] = (
        "fq_default",
        "fq_jax_statevector",
        "fq_jax_tensor_network",
        "tc_state_numpy",
        "tc_state_dlpack",
        "tc_mpo_numpy",
        "tc_mpo_dlpack",
        "tc_state_whole_jax",
        "tc_mpo_whole_jax",
    )
    routes = routes[rotation:] + routes[:rotation]
    for route in routes:
        started = time.perf_counter()
        record: dict[str, Any] = {"route": route}
        try:
            parameters = initial.clone().requires_grad_()
            if route.startswith("fq_"):
                options = (
                    fq.ExecutionOptions()
                    if route == "fq_default"
                    else fq.ExecutionOptions(
                        backend="jax", mode=route.removeprefix("fq_jax_")
                    )
                )
                model = fq.Module(
                    build,
                    initial.numel(),
                    dtype=torch.float64,
                    policy=fq.RuntimePolicy(
                        execution_options=options,
                        observable="z_sum",
                        observable_qubits=tuple(range(width)),
                    ),
                )

                def prediction(values: Any, selected_executor: Any = model) -> Any:
                    return selected_executor(features, parameters=values) / width

            else:
                observable = route.split("_")[1]

                def quantum(
                    values: Any, inputs: Any, selected_observable: str = observable
                ) -> Any:
                    return jax.vmap(lambda x: tc_value(values, x, selected_observable))(
                        jax.lax.stop_gradient(inputs)
                    )

                if "whole_jax" not in route:
                    bridge = tc.interfaces.torch_interface(
                        quantum, jit=True, enable_dlpack=route.endswith("dlpack")
                    )

                    def prediction(values: Any, selected_executor: Any = bridge) -> Any:
                        return selected_executor(values, features)

            if "whole_jax" in route:
                jp = jnp.asarray(initial.numpy())
                jx = jnp.asarray(features.numpy())

                def loss(values: Any, inputs: Any) -> Any:
                    return jnp.mean((quantum(values, inputs) - 0.3) ** 2)

                @jax.jit
                def update(values: Any, inputs: Any) -> Any:
                    value, gradient = jax.value_and_grad(loss)(values, inputs)
                    return value, values - 0.01 * gradient

                value, next_parameters = update(jp, jx)
                next_parameters.block_until_ready()
                current_value = float(value)
                current_gradient = (np.asarray(jp) - np.asarray(next_parameters)) / 0.01

                def step(selected_inputs: Any = jx) -> None:
                    nonlocal jp
                    _, jp = update(jp, selected_inputs)
                    jp.block_until_ready()

            else:
                value = ((prediction(parameters) - 0.3) ** 2).mean()
                gradient = torch.autograd.grad(value, parameters)[0]
                current_value = float(value.detach())
                current_gradient = gradient.numpy()

                def step(selected_inputs: Any = None) -> None:
                    nonlocal parameters
                    value = ((prediction(parameters) - 0.3) ** 2).mean()
                    gradient = torch.autograd.grad(value, parameters)[0]
                    parameters = (
                        (parameters - 0.01 * gradient).detach().requires_grad_()
                    )
                    value.item()

            record["setup_and_first_forward_backward_s"] = time.perf_counter() - started
            if oracle_value is None:
                oracle_value, oracle_gradient = current_value, current_gradient.copy()
            record["loss_abs_error"] = abs(current_value - oracle_value)
            record["gradient_max_abs_error"] = float(
                np.max(np.abs(current_gradient - oracle_gradient))
            )
            np.testing.assert_allclose(
                current_value, oracle_value, atol=1e-10, rtol=1e-10
            )
            np.testing.assert_allclose(
                current_gradient, oracle_gradient, atol=1e-9, rtol=1e-9
            )
            for _ in range(3):
                step()
            times = []
            for _ in range(5):
                started_block = time.perf_counter()
                for _ in range(20):
                    step()
                times.append((time.perf_counter() - started_block) / 20)
            record["process_peak_rss_bytes_through_route"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            record["final_parameters"] = (
                np.asarray(jp).tolist()
                if "whole_jax" in route
                else parameters.detach().tolist()
            )
            record.update(
                status="ok", seconds_per_update=times, median_s=statistics.median(times)
            )
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        records.append(record)
        print(json.dumps(record), file=sys.stderr, flush=True)
    return dict(width=width, batch=batch, depth=depth, records=records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=4, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker)))
        return
    if args.output is None:
        parser.error("--output is required")
    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    result: dict[str, Any] = dict(
        protocol="public-batched-sgd-cpu-v1",
        source_commit=source,
        working_tree_diff_sha256=hashlib.sha256(
            subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=root,
                capture_output=True,
                check=True,
                timeout=10,
            ).stdout
        ).hexdigest(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(),
        processor=platform.processor(),
        logical_cpu_count=os.cpu_count(),
        python=sys.version,
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "jax", "jaxlib", "tensorcircuit-ng")
        },
        dtype="complex128/float64",
        threads=1,
        warmup=3,
        blocks=5,
        updates_per_block=20,
        measurement_scope="finite CPU development comparison; not a universal ranking",
        memory_scope="macOS process peak RSS through route, cumulative across earlier routes; not per-route tensor allocation",
        gradient_demand="parameters only; fixed feature values remain dynamic runtime arguments",
        optimizer="SGD learning_rate=0.01; 103 updates per route",
        exact=True,
        distribution_semantics="single_device_fast_path",
        scalability_claim_allowed=False,
        runs=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for width in (2, 8, 12):
            for batch in (1, 32):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(width),
                    str(batch),
                    "2",
                    str((repeat * 3) % 9),
                ]
                environment = dict(
                    os.environ,
                    PYTHONPATH=str(root),
                    JAX_PLATFORMS="cpu",
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                )
                try:
                    process = subprocess.run(
                        command,
                        cwd=root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    payload = (
                        json.loads(process.stdout.splitlines()[-1])
                        if process.returncode == 0
                        else None
                    )
                    if payload is not None:
                        completed = [
                            item
                            for item in payload["records"]
                            if item["status"] == "ok"
                        ]
                        if completed:
                            reference = completed[0]["final_parameters"]
                            for item in completed:
                                item["final_parameter_max_abs_error"] = max(
                                    abs(actual - expected)
                                    for actual, expected in zip(
                                        item["final_parameters"], reference, strict=True
                                    )
                                )
                                if item["final_parameter_max_abs_error"] > 1e-9:
                                    item.update(
                                        status="error",
                                        error="103-update trajectory mismatch",
                                    )
                    run = dict(
                        payload=payload,
                        repeat=repeat,
                        width=width,
                        batch=batch,
                        returncode=process.returncode,
                        stdout=process.stdout,
                        stderr=process.stderr,
                    )
                except subprocess.TimeoutExpired as error:
                    payload = (
                        json.loads(process.stdout.splitlines()[-1])
                        if process.returncode == 0
                        else None
                    )
                    run = dict(
                        payload=payload,
                        repeat=repeat,
                        width=width,
                        batch=batch,
                        status="timeout",
                        stdout=str(error.stdout),
                        stderr=str(error.stderr),
                    )
                result["runs"].append(run)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    f"repeat={repeat} width={width} batch={batch} complete", flush=True
                )
    failed = any(
        run.get("status") == "timeout"
        or run.get("returncode") != 0
        or any(item["status"] != "ok" for item in run["payload"]["records"])
        for run in result["runs"]
    )
    result["status"] = "failed" if failed else "ok"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
