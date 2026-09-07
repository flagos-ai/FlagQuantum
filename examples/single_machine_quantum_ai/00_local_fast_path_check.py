"""Check that local FlagQuantum fast paths are ready.

This script intentionally stays single-process and single-device. It validates
statevector, MPS, and tensor-network modes against the local statevector
reference before users start training examples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    exact_ground_energy,
    print_kv,
    print_section,
    print_speed_compare,
    time_value_and_grad,
)

import flagquantum as fq  # noqa: E402
from flagquantum.algorithms import Hamiltonian, pauli_term  # noqa: E402


def tiny_circuit(theta: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(2)
    circuit.ry(0, theta=theta[0])
    circuit.rx(1, theta=theta[1])
    circuit.cx(0, 1)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-torch", action="store_true")
    args = parser.parse_args()

    preflight_circuit = (
        fq.Circuit(4)
        .h(0)
        .rx(1, theta=0.2)
        .cx(0, 2)
        .ry(3, theta=-0.4)
        .rzz(2, 3, theta=0.3)
    )
    measurements = (fq.MeasurementNode("expectation_z", (0, 1, 2, 3)),)
    results = {}
    reference = None
    for mode in ("statevector", "mps", "tensor_network"):
        result = fq.run(
            preflight_circuit,
            options=fq.ExecutionOptions(mode=mode, precision="complex64"),
            measurements=measurements,
        )
        value = result.measurements[0].value
        if reference is None:
            reference = value
        results[mode] = {
            "distribution_semantics": result.plan.summary()["distribution_semantics"],
            "max_abs_error_vs_statevector": float(
                torch.max(torch.abs(value - reference))
            ),
        }
    passed = all(
        item["distribution_semantics"] == "single_device_fast_path"
        and item["max_abs_error_vs_statevector"] <= 1e-6
        for item in results.values()
    )
    print_section("Local Fast Path Preflight")
    print_kv({"status": "passed" if passed else "failed", "device": "cpu"})
    for mode, item in results.items():
        print(
            f"  {mode}: semantics={item['distribution_semantics']} "
            f"error={item['max_abs_error_vs_statevector']:.3e}"
        )
    if not passed:
        raise SystemExit("local fast-path validation failed")

    theta = torch.tensor([0.2, -0.3], requires_grad=True)
    hamiltonian = Hamiltonian(
        [pauli_term(1.0, "ZZ", (0, 1)), pauli_term(-0.2, "X", (0,))]
    )
    native_speed = None
    if args.compare_torch:
        native_speed = time_value_and_grad(
            lambda params: hamiltonian.expectation(tiny_circuit(params)).sum(),
            theta.detach(),
        )
    print_section("Tiny Hamiltonian Correctness Reference")
    print_kv({"theoretical_ground_energy": exact_ground_energy(hamiltonian, 2)})
    print_speed_compare(
        {
            "pytorch_native": native_speed
            if native_speed is not None
            else {"status": "unavailable", "reason": "run with --compare-torch"},
            "jax_kernel": {
                "status": "see examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py"
            },
        }
    )


if __name__ == "__main__":
    main()
