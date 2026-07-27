"""Check that local FlagQuantum fast paths are ready.

This script intentionally stays single-process and single-device. It validates
statevector, MPS, and tensor-network modes against the local statevector
reference before users start training examples.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
from common import exact_ground_energy, jax_available, print_kv, print_section, print_speed_compare, speedup, time_value_and_grad  # noqa: E402


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

    report = fq.local_fast_path_preflight()
    summary = report.summary()
    print_section("Local Fast Path Preflight")
    print_kv({"status": "passed" if report.passed else "failed", "device": summary["device"]})
    for mode, item in summary["results"].items():
        print(
            f"  {mode}: semantics={item['distribution_semantics']} "
            f"error={item['max_abs_error_vs_statevector']:.3e}"
        )
    if not report.passed:
        raise SystemExit(summary["errors"])

    theta = torch.tensor([0.2, -0.3], requires_grad=True)
    hamiltonian = fq.Hamiltonian([fq.pauli_term(1.0, "ZZ", (0, 1)), fq.pauli_term(-0.2, "X", (0,))])
    native_speed = None
    if args.compare_torch:
        native_speed = time_value_and_grad(
            lambda params: hamiltonian.expectation(tiny_circuit(params)).sum(),
            theta.detach(),
        )
    has_jax, jax_error = jax_available()
    jax_speed = None
    if has_jax:
        kernel = fq.compile_quantum_kernel(
            tiny_circuit,
            theta.detach(),
            backend="jax",
            interface="torch",
            mode="statevector",
            n_wires=2,
            hamiltonian=hamiltonian,
            jit=True,
        )
        jax_speed = time_value_and_grad(lambda params: kernel(params).sum(), theta.detach())
    print_section("Tiny Hamiltonian Correctness Reference")
    print_kv({"theoretical_ground_energy": exact_ground_energy(hamiltonian, 2)})
    print_speed_compare(
        {
            "pytorch_native": native_speed if native_speed is not None else {"status": "unavailable", "reason": "run with --compare-torch"},
            "jax_kernel": jax_speed if jax_speed is not None else {"status": "unavailable", "reason": jax_error},
            "speedup_jax_over_pytorch": speedup(
                native_speed["avg_seconds"] if native_speed else None,
                jax_speed["avg_seconds"] if jax_speed else None,
            ),
        }
    )


if __name__ == "__main__":
    main()
