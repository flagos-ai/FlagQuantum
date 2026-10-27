"""Measure a public ``fq.Module`` Hamiltonian energy on every representation.

The tensor-network Module route answers a Hamiltonian observable without
materializing a dense state and without contracting the shared state once per
term. Both properties are easy to misstate, so this producer measures them.

First, it must not materialize a dense state. The tensor-network state's own Z
readout materializes one below its observable threshold, so an energy built from
``expectation_z`` would look correct while silently taking the dense path the
caller selected the contraction representation to avoid. The payload records the
state's own ``full_state_materialized`` flag and the executed observable mode.

Second, the shared ``Hamiltonian`` implementation must agree with the exact
statevector route in value and in parameter gradient, including a declared batch
and a nonlinear outer loss. Agreement is measured against two references: the
statevector module for the many-body workload, and a closed-form single-qubit
energy for the small case, so a shared mistake in both simulators would not pass
as agreement.

Cost is reported as measured elapsed time for the same circuit and term count on
the statevector and tensor-network routes, with raw repeated samples, a median
and a median absolute deviation. The term-count row holds the circuit fixed and
records the operator network's node count beside its build and contract times,
so an independence claim rests on a structure count as well as on a wall clock.

Examples
--------
Write a development artifact payload::

    OMP_NUM_THREADS=1 python benchmarks/tn_module_hamiltonian.py \
        --output artifacts/development/tn_module_hamiltonian_cpu_20261027/summary.json

A single-width smoke run::

    python benchmarks/tn_module_hamiltonian.py --widths 2 --repeats 3 \
        --output /tmp/tn-module-hamiltonian.json

This is a single-process CPU correctness and cost measurement. It establishes no
GPU, distributed, or universal performance claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.simulation.tensor_network.observables import (
    build_tensor_network_hamiltonian_expectation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODES = ("statevector", "mps", "tensor_network")


def build_ansatz(width: int, depth: int, dtype: torch.dtype, bsz: int = 1):
    """Return one parameterized ansatz builder over ``width`` wires."""

    def build(parameters: torch.Tensor) -> fq.Circuit:
        angles = parameters.reshape(depth, width, 2)
        circuit = fq.Circuit(width, bsz=bsz, dtype=dtype)
        for layer in range(depth):
            for wire in range(width):
                circuit.ry(wire, angles[layer, wire, 0])
                circuit.rz(wire, angles[layer, wire, 1])
            for wire in range(width - 1):
                circuit.cx(wire, wire + 1)
        return circuit

    return build


def transverse_field_sum(width: int) -> Hamiltonian:
    """Return a Z-on-every-wire and X-on-every-wire Pauli sum."""

    return Hamiltonian(
        [pauli_term(-0.5, "Z", (wire,)) for wire in range(width)]
        + [pauli_term(-0.7, "X", (wire,)) for wire in range(width)]
    )


def build_module(
    mode: str,
    *,
    width: int,
    depth: int,
    observable: str = "hamiltonian",
    bsz: int = 1,
    dtype: torch.dtype = torch.float64,
    seed: int = 260720,
    terms: int | None = None,
) -> fq.Module:
    """Build a public Module for one representation and observable.

    ``terms`` shortens the transverse-field sum to its first ``terms`` entries,
    which is how the term-count scaling row keeps the circuit fixed while the
    observable grows.
    """

    generator = torch.Generator().manual_seed(seed)
    initial = 0.1 * torch.randn(depth, width, 2, generator=generator)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    hamiltonian = transverse_field_sum(width)
    if terms is not None:
        hamiltonian = Hamiltonian(hamiltonian.terms[:terms])
    return fq.Module(
        build_ansatz(width, depth, complex_dtype, bsz),
        initial.numel(),
        init=initial.reshape(-1).to(dtype),
        dtype=dtype,
        hamiltonian=hamiltonian,
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                mode=mode,
                precision="complex128" if dtype == torch.float64 else "complex64",
                allow_backend_fallback=False,
            ),
            observable=observable,
            observable_qubits=tuple(range(width)),
        ),
    )


def timed(operation, *, repeats: int) -> list[float]:
    """Return repeated elapsed seconds for one operation."""

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return samples


def summarize(samples: list[float]) -> dict[str, Any]:
    """Report the raw samples beside a median and its absolute deviation."""

    median = statistics.median(samples)
    deviation = statistics.median([abs(value - median) for value in samples])
    return {
        "samples_seconds": [round(value, 6) for value in samples],
        "median_seconds": round(median, 6),
        "median_absolute_deviation_seconds": round(deviation, 6),
    }


def representation_agreement(depth: int) -> dict[str, Any]:
    """Compare every representation's energy and gradient on one workload."""

    records: dict[str, Any] = {}
    for mode in MODES:
        module = build_module(mode, width=3, depth=depth)
        result = module.execute()
        energy = float(result.value.detach().reshape(-1)[0])
        loss = (result.value.sum() - 0.4) ** 2
        loss.backward()
        assert module.parameters_tensor.grad is not None
        records[mode] = {
            "energy": energy,
            "loss": float(loss.detach()),
            "parameter_gradient": [
                float(value) for value in module.parameters_tensor.grad.detach()
            ],
            "runtime_observable_execution": result.runtime.get("observable_execution"),
            "runtime_full_state_materialized": result.runtime.get(
                "full_state_materialized"
            ),
            "runtime_observable_peak_size": result.runtime.get("observable_peak_size"),
            "metrics_observable_count": result.metrics.get("observable_count"),
        }

    reference = records["statevector"]
    disagreements = {
        mode: {
            "energy_abs_error": abs(record["energy"] - reference["energy"]),
            "gradient_max_abs_error": max(
                abs(left - right)
                for left, right in zip(
                    record["parameter_gradient"],
                    reference["parameter_gradient"],
                    strict=True,
                )
            ),
        }
        for mode, record in records.items()
        if mode != "statevector"
    }
    return {"representations": records, "maximum_disagreement": disagreements}


def closed_form_check() -> dict[str, Any]:
    """Check one tensor-network energy against an analytic single-qubit value.

    ``RY(theta)`` on ``|0>`` gives ``<Z> = cos(theta)`` and ``<X> = sin(theta)``,
    so ``Z + 0.5 X`` has the value below and the analytic parameter derivative
    follows by differentiating it. Neither simulator is consulted.
    """

    theta = 0.7
    coefficient_z, coefficient_x = 1.0, 0.5
    expected_value = coefficient_z * math.cos(theta) + coefficient_x * math.sin(theta)
    expected_gradient = -coefficient_z * math.sin(theta) + coefficient_x * math.cos(
        theta
    )
    module = fq.Module(
        lambda parameters: fq.Circuit(1, dtype=torch.complex128).ry(0, parameters[0]),
        1,
        init=torch.tensor([theta], dtype=torch.float64),
        dtype=torch.float64,
        hamiltonian=Hamiltonian(
            [
                pauli_term(coefficient_z, "Z", (0,)),
                pauli_term(coefficient_x, "X", (0,)),
            ]
        ),
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                mode="tensor_network",
                precision="complex128",
                allow_backend_fallback=False,
            ),
            observable="hamiltonian",
            observable_qubits=(0,),
        ),
    )
    value = module().sum()
    value.backward()
    assert module.parameters_tensor.grad is not None
    measured_value = float(value.detach())
    measured_gradient = float(module.parameters_tensor.grad.detach()[0])
    return {
        "theta": theta,
        "analytic_value": expected_value,
        "measured_value": measured_value,
        "value_abs_error": abs(measured_value - expected_value),
        "analytic_gradient": expected_gradient,
        "measured_gradient": measured_gradient,
        "gradient_abs_error": abs(measured_gradient - expected_gradient),
    }


def declared_batch_check(depth: int) -> dict[str, Any]:
    """Compare a declared batch against the shared single-example energy."""

    single = build_module("tensor_network", width=3, depth=depth)
    batched = build_module("tensor_network", width=3, depth=depth, bsz=4)
    single_value = float(single().detach().reshape(-1)[0])
    batched_values = [float(value) for value in batched().detach()]
    return {
        "batch_size": 4,
        "single_value": single_value,
        "batched_values": batched_values,
        "maximum_abs_error": max(abs(value - single_value) for value in batched_values),
        "shared_parameters": True,
    }


def training_and_recovery(
    depth: int, steps: int, checkpoint_step: int
) -> dict[str, Any]:
    """Train, checkpoint, and compare a resumed run against the uninterrupted one."""

    source = build_module("tensor_network", width=3, depth=depth)
    optimizer = torch.optim.Adam(source.parameters(), lr=0.05)
    losses: list[float] = []
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "tn_module_hamiltonian.pt")
        for _ in range(checkpoint_step):
            optimizer.zero_grad()
            loss = (source().sum() - 0.4) ** 2
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        source.save_checkpoint(
            checkpoint, optimizer=optimizer, seed=11, step=checkpoint_step
        )
        expected_losses = []
        for _ in range(steps - checkpoint_step):
            optimizer.zero_grad()
            loss = (source().sum() - 0.4) ** 2
            loss.backward()
            optimizer.step()
            expected_losses.append(float(loss.detach()))
        expected_parameters = source.parameters_tensor.detach().clone()

        target = build_module("tensor_network", width=3, depth=depth)
        target_optimizer = torch.optim.Adam(target.parameters(), lr=0.05)
        restored = target.load_checkpoint(checkpoint, optimizer=target_optimizer)
        resumed_losses = []
        for _ in range(steps - checkpoint_step):
            target_optimizer.zero_grad()
            loss = (target().sum() - 0.4) ** 2
            loss.backward()
            target_optimizer.step()
            resumed_losses.append(float(loss.detach()))

    return {
        "optimizer": "Adam(lr=0.05)",
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "losses_before_checkpoint": losses[:checkpoint_step],
        "losses_after_checkpoint_uninterrupted": expected_losses,
        "losses_after_checkpoint_resumed": resumed_losses,
        "maximum_loss_abs_error": max(
            (
                abs(left - right)
                for left, right in zip(expected_losses, resumed_losses, strict=True)
            ),
            default=0.0,
        ),
        "maximum_parameter_abs_error": float(
            (target.parameters_tensor.detach() - expected_parameters).abs().max()
        ),
        "restored_step": restored["step"],
    }


def cost(widths: tuple[int, ...], depth: int, repeats: int) -> list[dict[str, Any]]:
    """Compare statevector and tensor-network energy cost at one fixed workload."""

    cells: list[dict[str, Any]] = []
    for width in widths:
        records: dict[str, Any] = {}
        for mode in ("statevector", "tensor_network"):
            module = build_module(mode, width=width, depth=depth)
            started = time.perf_counter()
            result = module.execute()
            cold = time.perf_counter() - started
            forward = timed(module, repeats=repeats)

            def forward_backward(current: fq.Module = module) -> None:
                current.zero_grad()
                current().sum().backward()

            records[mode] = {
                "cold_execute_seconds": round(cold, 6),
                "forward": summarize(forward),
                "forward_and_backward": summarize(
                    timed(forward_backward, repeats=repeats)
                ),
                "observable_execution": result.runtime.get("observable_execution"),
                "full_state_materialized": result.runtime.get(
                    "full_state_materialized"
                ),
                "observable_peak_size": result.runtime.get("observable_peak_size"),
                "observable_count": result.metrics.get("observable_count"),
            }
        readout = build_module(
            "tensor_network", width=width, depth=depth, observable="z_sum"
        )
        readout_result = readout.execute()
        median_sv = records["statevector"]["forward"]["median_seconds"]
        median_tn = records["tensor_network"]["forward"]["median_seconds"]
        cells.append(
            {
                "width": width,
                "depth": depth,
                "hamiltonian_terms": 2 * width,
                "routes": records,
                "tensor_network_z_sum_readout": {
                    "observable_execution": readout_result.runtime.get(
                        "observable_execution"
                    ),
                    "full_state_materialized": readout_result.runtime.get(
                        "full_state_materialized"
                    ),
                },
                "tensor_network_over_statevector_forward_ratio": (
                    round(median_tn / median_sv, 4) if median_sv else None
                ),
            }
        )
    return cells


def term_scaling(width: int, depth: int, repeats: int) -> list[dict[str, Any]]:
    """Hold the circuit fixed and grow the observable term count.

    The sum is contracted as one operator network, so the cost of that network
    does not depend on how many terms it carries. This row exists to show that
    independence instead of asserting it: only ``terms`` changes across the
    cells. Each cell records the end-to-end module forward, which is what a user
    pays, and the operator network's own node count with its build and contract
    times, which is where the independence is provable rather than inferred from
    a wall clock.
    """

    from flagquantum.simulation.tensor_network.entrypoints import run_tensor_network

    full = 2 * width
    generator = torch.Generator().manual_seed(260720)
    initial = 0.1 * torch.randn(depth, width, 2, generator=generator)
    state = run_tensor_network(
        build_ansatz(width, depth, torch.complex128)(initial.reshape(-1)),
        dense_observable_qubits=12,
    )
    cells: list[dict[str, Any]] = []
    for terms in (1, 2, 4, full):
        hamiltonian = Hamiltonian(transverse_field_sum(width).terms[:terms])
        build_seconds = timed(
            lambda current=hamiltonian: build_tensor_network_hamiltonian_expectation(
                state.plan, current
            ),
            repeats=repeats,
        )
        operator = build_tensor_network_hamiltonian_expectation(state.plan, hamiltonian)
        contract_seconds = timed(
            lambda current=operator: torch.real(current.contract()), repeats=repeats
        )
        module = build_module("tensor_network", width=width, depth=depth, terms=terms)
        result = module.execute()
        forward = timed(module, repeats=repeats)
        cells.append(
            {
                "terms": terms,
                "forward": summarize(forward),
                "operator_nodes": len(operator.nodes),
                "operator_build": summarize(build_seconds),
                "operator_contract": summarize(contract_seconds),
                "observable_execution": result.runtime.get("observable_execution"),
                "full_state_materialized": result.runtime.get(
                    "full_state_materialized"
                ),
                "observable_peak_size": result.runtime.get("observable_peak_size"),
                "energy": float(result.value.detach().reshape(-1)[0]),
            }
        )
    return cells


def peak_scaling(widths: tuple[int, ...], depth: int) -> list[dict[str, Any]]:
    """Record the operator contraction peak against the dense state it replaces.

    The reason to pick this representation is capacity, not speed, so the peak is
    the number that decides. A dense state holds ``2**width`` amplitudes; the
    operator contraction holds a peak intermediate of its own, and whether that
    peak is smaller is a measured question rather than an assumption.
    """

    from flagquantum.simulation.tensor_network.entrypoints import run_tensor_network

    cells: list[dict[str, Any]] = []
    for width in widths:
        generator = torch.Generator().manual_seed(260720)
        initial = 0.1 * torch.randn(depth, width, 2, generator=generator)
        circuit = build_ansatz(width, depth, torch.complex128)(initial.reshape(-1))
        state = run_tensor_network(circuit, dense_observable_qubits=12)
        hamiltonian = transverse_field_sum(width)
        state.expectation_hamiltonian(hamiltonian)
        peak = state.summary()["observable_peak_size"]
        cells.append(
            {
                "width": width,
                "terms": 2 * width,
                "dense_state_elements": 2**width,
                "observable_peak_size": peak,
                "peak_over_dense": (
                    None if peak is None else round(peak / 2**width, 4)
                ),
            }
        )
    return cells


def git_revision(*arguments: str) -> str:
    """Return one git revision or an empty string when it is unavailable."""

    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True, timeout=15
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--peak-widths", type=int, nargs="+", default=[8, 10, 12, 14])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.manual_seed(260720)

    started = time.perf_counter()
    payload = {
        "schema": "flagquantum.tn_module_hamiltonian.v1",
        "benchmark": "tn_module_hamiltonian",
        "artifact_class": "measured_local_run",
        "claim": (
            "a public fq.Module Hamiltonian energy runs on the tensor-network "
            "representation by contracting the whole Pauli sum as one operator "
            "network, without materializing a dense state"
        ),
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_blockers": [
            "single-process CPU correctness and cost measurement; it shards no "
            "workload and is not distributed or accelerator evidence"
        ],
        "workload": {
            "ansatz": "depth x (ry, rz on every wire, cx chain), shared parameters",
            "hamiltonian": "Z on every wire at -0.5, X on every wire at -0.7",
            "dtype": "complex128 amplitudes, float64 parameters",
            "depth": arguments.depth,
        },
        "closed_form_check": closed_form_check(),
        "representation_agreement": representation_agreement(arguments.depth),
        "declared_batch": declared_batch_check(arguments.depth),
        "training_and_checkpoint_recovery": training_and_recovery(
            arguments.depth, arguments.steps, checkpoint_step=2
        ),
        "cost": cost(tuple(arguments.widths), arguments.depth, arguments.repeats),
        "peak_scaling": peak_scaling(tuple(arguments.peak_widths), arguments.depth),
        "term_count_scaling": term_scaling(
            arguments.widths[-1], arguments.depth, arguments.repeats
        ),
        "limitations": [
            "the sum is one matrix-product operator over the state's own ket and "
            "bra copies; its node count is independent of the term count, and "
            "the per-term route this replaces was not retained as an option",
            "the tensor-network route reports no observable contraction peak; "
            "estimating it would repeat the path search that answered the call",
            "measured wall time on one CPU process, with no accelerator, "
            "distributed, memory, or time-to-target claim",
            "gradients are first-order PyTorch autograd through the contraction",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": " ".join(sys.argv),
        "threads": {
            name: os.environ.get(name, "")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "base_commit": git_revision("rev-parse", "origin/main"),
        "source_commit": git_revision("rev-parse", "HEAD"),
        "measuring_source_committed": not git_revision(
            "status", "--porcelain", "--", str(Path(__file__).resolve())
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "maximum_representation_energy_error": max(
                    item["energy_abs_error"]
                    for item in payload["representation_agreement"][
                        "maximum_disagreement"
                    ].values()
                ),
                "closed_form_value_abs_error": payload["closed_form_check"][
                    "value_abs_error"
                ],
                "checkpoint_parameter_abs_error": payload[
                    "training_and_checkpoint_recovery"
                ]["maximum_parameter_abs_error"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
