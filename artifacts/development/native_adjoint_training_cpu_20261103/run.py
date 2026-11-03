"""Measure frozen CPU tasks through fq.train and the existing CPU adjoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ROUTES = ("fq_train", "native_adjoint")


def worker(task: str, route: str, seed: int, started: float) -> dict[str, Any]:
    import torch

    import flagquantum as fq
    import flagquantum.algorithms as fqa
    from benchmarks.training_comparison import (
        _optimizer,
        ground_energy,
        source_identity,
    )
    from benchmarks.training_workloads import (
        binary_loss,
        dataset,
        dense_loss,
        finite_difference,
        initial_parameters,
        recipe,
        terms,
    )
    from flagquantum.runtime.executors.statevector.reverse import (
        execute_torch_distributed_statevector_reverse,
    )
    from flagquantum.simulation.native_cpu.adjoint import native_cpu_adjoint_available

    imported = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    train, validation, test = dataset()
    initial = initial_parameters(task, seed).detach()
    classifier = task == "classifier"
    qubits = 2 if classifier else 4
    gates = recipe(task)
    paulis = terms(task)
    energy = None if classifier else ground_energy()
    groups = {
        word: tuple(
            (coefficient, wires)
            for coefficient, label, wires in paulis
            if label == word
        )
        for word in sorted({label for _, label, _ in paulis})
    }

    def circuit(p, x, basis="Z"):
        batched = x.ndim == 2
        result = fq.Circuit(
            qubits, bsz=x.shape[0] if batched else 1, dtype=torch.complex128
        )
        for gate in gates:
            if gate.feature is not None:
                result.gate(
                    gate.name, gate.wires, theta=x[..., gate.feature] * gate.scale
                )
            elif gate.parameter is not None:
                result.gate(gate.name, gate.wires, theta=p[gate.parameter] * gate.scale)
            else:
                result.gate(gate.name, gate.wires)
        if basis != "Z":
            for wire in sorted({wire for _, wires in groups[basis] for wire in wires}):
                if basis == "YY":
                    result.sdg(wire)
                result.h(wire)
        return result

    def public_builder(p, x):
        return circuit(p, x)

    observable = fqa.Hamiltonian(
        [
            fqa.pauli_term(coefficient, word, wires)
            for coefficient, word, wires in paulis
        ]
    )

    def new_module():
        return fq.Module(
            public_builder,
            len(initial),
            init=initial.clone(),
            dtype=torch.float64,
            policy=fq.RuntimePolicy(
                observable="hamiltonian",
                observable_qubits=tuple(range(qubits)),
                execution_options=fq.ExecutionOptions(
                    mode="statevector", precision="complex128"
                ),
            ),
            hamiltonian=observable,
        )

    model = new_module() if route == "fq_train" else None
    evaluator = new_module()
    parameters = (
        model.parameters_tensor
        if model is not None
        else initial.clone().requires_grad_()
    )
    optimizer = _optimizer(parameters, task)
    native_results = []
    native_counts = {"sweeps": 0, "fused_parameter_adjoint_count": 0}

    def native_value(x):
        values = []
        # The existing native reverse executor accepts Z/ZZ observables. Group
        # commuting terms by measurement basis and keep their exact coefficients.
        with torch.enable_grad():
            for word, coefficients in groups.items():
                basis = "Z" if word in {"Z", "ZZ"} else word
                execution = execute_torch_distributed_statevector_reverse(
                    circuit(parameters, x, basis),
                    observable_terms=coefficients,
                    device="cpu",
                )
                values.append(execution.value)
                if training_native:
                    native_results.append(execution)
        return sum(values)

    def measured(split):
        features = split.features if classifier else torch.zeros(2)
        if model is not None:
            return model(features).reshape(-1)
        if classifier:
            return torch.stack([native_value(row).reshape(()) for row in features])
        return native_value(features).reshape(-1)

    def loss(value, split):
        if classifier:
            return binary_loss(parameters[4] * value.reshape(-1) + parameters[5], split)
        return value.reshape(())

    def score(split):
        with torch.no_grad():
            inputs = split.features if classifier else torch.zeros(2)
            value = evaluator(inputs, parameters=parameters).reshape(-1)
            if classifier:
                prediction = parameters[4] * value + parameters[5] > 0
                return float((prediction == split.labels.bool()).double().mean())
            return abs(float(value.detach()) - energy) / abs(energy)

    if route == "native_adjoint" and not native_cpu_adjoint_available():
        raise RuntimeError("native CPU adjoint extension is unavailable")

    def target(value):
        return value >= 0.8 if classifier else value <= 0.01

    setup_end = time.perf_counter()
    training_native = False
    initial_score = score(validation)
    initial_end = time.perf_counter()
    trace = []
    previous = initial_end

    def observe(step, value, execution=None, *, checkpoint_time=0.0):
        nonlocal previous
        update_end = time.perf_counter()
        validation_score = score(validation)
        checked = time.perf_counter()
        trace.append(
            {
                "step": step,
                "loss": value,
                "score": validation_score,
                "update_seconds": update_end - previous - checkpoint_time,
                "validation_seconds": checked - update_end,
                "elapsed_seconds": checked - started,
                "ir_hash": (
                    None if execution is None else execution.provenance["ir_hash"]
                ),
            }
        )
        previous = time.perf_counter()

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "step30.pt"
        checkpoint_seconds = 0.0

        def callback(step, value, execution):
            nonlocal checkpoint_seconds
            if step == 30:
                begin = time.perf_counter()
                model.save_checkpoint(
                    checkpoint, optimizer=optimizer, seed=seed, step=step
                )
                checkpoint_seconds = time.perf_counter() - begin
            observe(
                step,
                value,
                execution,
                checkpoint_time=checkpoint_seconds if step == 30 else 0.0,
            )

        if model is not None:
            result = fq.train(
                model,
                optimizer=optimizer,
                objective=lambda value: loss(value, train),
                inputs=train.features if classifier else torch.zeros(2),
                steps=60,
                callback=callback,
            )
            diagnostics = result.last_execution.runtime["training_compilation"]
        else:
            diagnostics = {}
            for step in range(1, 61):
                optimizer.zero_grad(set_to_none=True)
                training_native = True
                value = loss(measured(train), train)
                value.backward()
                training_native = False
                for execution in native_results:
                    evidence = execution.backward_evidence
                    if evidence.status != "completed":
                        raise RuntimeError("CPU adjoint backward did not complete")
                    native_counts["sweeps"] += 1
                    native_counts[
                        "fused_parameter_adjoint_count"
                    ] += evidence.fused_parameter_adjoint_count
                native_results.clear()
                optimizer.step()
                if step == 30:
                    begin = time.perf_counter()
                    torch.save(
                        {
                            "parameters": parameters.detach(),
                            "optimizer": optimizer.state_dict(),
                        },
                        checkpoint,
                    )
                    checkpoint_seconds = time.perf_counter() - begin
                observe(
                    step,
                    float(value.detach()),
                    checkpoint_time=checkpoint_seconds if step == 30 else 0.0,
                )
            if native_counts["fused_parameter_adjoint_count"] == 0:
                raise RuntimeError("no fused native parameter adjoint executed")
        finished = time.perf_counter()
        final_score = score(test if classifier else validation)
        final_parameters = parameters.detach().tolist()
        final_gradient = parameters.grad.detach().tolist()
        final_slots = {
            key: value.detach().tolist()
            for key, value in optimizer.state[parameters].items()
        }
        # Replay the remaining updates with a fresh optimizer. Public replay
        # uses a fresh Module and the existing checkpoint API.
        replay_started = time.perf_counter()
        if model is not None:
            replay = new_module()
            parameters = replay.parameters_tensor
            replay_optimizer = _optimizer(parameters, task)
            replay(train.features if classifier else torch.zeros(2))
            replay.load_checkpoint(checkpoint, optimizer=replay_optimizer)
            fq.train(
                replay,
                optimizer=replay_optimizer,
                objective=lambda value: loss(value, train),
                inputs=train.features if classifier else torch.zeros(2),
                steps=30,
            )
        else:
            saved = torch.load(checkpoint, weights_only=True)
            parameters = saved["parameters"].requires_grad_()
            replay_optimizer = _optimizer(parameters, task)
            replay_optimizer.load_state_dict(saved["optimizer"])
            for _ in range(30):
                replay_optimizer.zero_grad(set_to_none=True)
                loss(measured(train), train).backward()
                replay_optimizer.step()
        for observed, expected in (
            (parameters.detach(), final_parameters),
            (parameters.grad, final_gradient),
            *(
                (replay_optimizer.state[parameters][key], value)
                for key, value in final_slots.items()
            ),
        ):
            torch.testing.assert_close(
                observed,
                torch.as_tensor(expected),
                atol=1e-10,
                rtol=0,
                check_dtype=False,
            )
        replay_seconds = time.perf_counter() - replay_started

    # Dense finite differences run outside the measured train and replay region.
    parameters = initial.clone().requires_grad_()
    if model is not None:
        with torch.no_grad():
            model.parameters_tensor.copy_(initial)
        parameters = model.parameters_tensor
    probe = type(train)(train.features[:4], train.labels[:4])
    value = loss(measured(probe), probe)
    gradient = torch.autograd.grad(value, parameters)[0]
    value_error = abs(float(value.detach()) - dense_loss(task, initial.numpy(), probe))
    gradient_error = float(
        (gradient - torch.from_numpy(finite_difference(task, initial, probe)))
        .abs()
        .max()
    )
    if value_error > 1e-9 or gradient_error > 2e-6:
        raise RuntimeError(
            f"independent parity failed: value={value_error}, gradient={gradient_error}"
        )

    reached = next((row for row in trace if target(row["score"])), None)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "status": "passed",
        "task": task,
        "seed": seed,
        "route": route,
        "imports_seconds": imported - started,
        "setup_seconds": setup_end - imported,
        "initial_validation_seconds": initial_end - setup_end,
        "initial_score": initial_score,
        "target_step": (
            0 if target(initial_score) else reached["step"] if reached else None
        ),
        "target_seconds": (
            initial_end - started
            if target(initial_score)
            else reached["elapsed_seconds"] if reached else None
        ),
        "training_seconds": finished - initial_end,
        "total_seconds": finished - started,
        "checkpoint_seconds": checkpoint_seconds,
        "replay_seconds": replay_seconds,
        "peak_rss_bytes": rss if sys.platform == "darwin" else 1024 * rss,
        "final_score": final_score,
        "final_parameters": final_parameters,
        "final_gradient": final_gradient,
        "final_optimizer_slots": final_slots,
        "dense_value_error": value_error,
        "dense_gradient_error": gradient_error,
        "native_counts": native_counts,
        "diagnostics": dict(diagnostics),
        "trace": trace,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
        },
        "identity": source_identity(),
    }


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", nargs=3, metavar=("TASK", "ROUTE", "SEED"))
    args = parser.parse_args()
    if args.worker:
        task, route, seed = args.worker
        result = worker(task, route, int(seed), started)
        args.output.write_text(json.dumps(result))
        return 0

    sys.path.insert(0, str(ROOT))
    from tools.sanitize_public_evidence import sanitize_text

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory() as temporary:
        for index, seed in enumerate((11, 29, 47)):
            for task in ("classifier", "heisenberg"):
                for route in ROUTES[:: (-1 if index % 2 else 1)]:
                    output = Path(temporary) / f"{task}-{route}-{seed}.json"
                    try:
                        process = subprocess.run(
                            [
                                sys.executable,
                                __file__,
                                "--output",
                                str(output),
                                "--worker",
                                task,
                                route,
                                str(seed),
                            ],
                            env=env,
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            timeout=240,
                        )
                        result = (
                            json.loads(output.read_text())
                            if process.returncode == 0
                            else {
                                "status": "failed",
                                "error": process.stderr[-6000:],
                            }
                        )
                    except subprocess.TimeoutExpired:
                        result = {"status": "timeout"}
                    result.update(task=task, route=route, seed=seed)
                    results.append(result)
                    print(task, route, seed, result["status"], flush=True)
                    args.output.write_text(
                        sanitize_text(
                            json.dumps(
                                {
                                    "producer_sha256": hashlib.sha256(
                                        Path(__file__).read_bytes()
                                    ).hexdigest(),
                                    "results": results,
                                },
                                indent=2,
                            )
                        )
                        + "\n"
                    )
    return int(any(result["status"] != "passed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
