"""Train a parameterized circuit, bind optimized parameters, then deploy.

This example shows the intended FlagQuantum train-to-deploy flow:

1. Train a parameterized circuit with PyTorch autograd on the native runtime.
2. Bind the optimized values into a named-parameter circuit template.
3. Compile/package the bound circuit for a quantum-cloud target.
4. Submit it through a provider interface for inference.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flagquantum as fq


PARAMETERS = (fq.Parameter("theta_0"), fq.Parameter("theta_1"), fq.Parameter("theta_2"))


def circuit_template() -> fq.Circuit:
    """Build the deployable circuit template with named parameters."""

    circuit = fq.Circuit(2)
    circuit.ry(0, theta=PARAMETERS[0])
    circuit.rx(1, theta=PARAMETERS[1])
    circuit.cx(0, 1)
    circuit.rz(1, theta=PARAMETERS[2])
    return circuit


def ansatz(parameters: torch.Tensor) -> fq.Circuit:
    """Bind concrete parameter tensors for differentiable training."""

    return circuit_template().bind_parameters(
        {parameter: value for parameter, value in zip(PARAMETERS, parameters)}
    )


def main() -> None:
    initial_parameters = torch.tensor([0.2, -0.1, 0.3], requires_grad=True)
    hamiltonian = fq.Hamiltonian(
        [
            fq.pauli_term(1.0, "ZZ", (0, 1)),
            fq.pauli_term(0.2, "Z", (0,)),
        ]
    )

    training_result = fq.run_vqe(
        ansatz,
        initial_parameters,
        hamiltonian,
        steps=100,
        lr=0.1,
    )
    
    # This is the key deployment step: bind the optimized values into the same
    # named-parameter template before exporting/submitting to a real backend.
    optimized_parameters = training_result.parameters
    trained_circuit = circuit_template().bind_parameters(
        {parameter: value for parameter, value in zip(PARAMETERS, optimized_parameters)}
    )

    provider = fq.LocalSimulatorProvider()
    backend = fq.CloudBackendProfile(
        provider="local",
        name="line2",
        n_wires=2,
        coupling_map=fq.CouplingMap.line(2),
        is_simulator=True,
    )
    package = fq.create_deployment_package(
        trained_circuit,
        backend=backend,
        name="trained_parameterized_inference",
        shots=1024,
        metadata={
            "optimized_parameters": optimized_parameters.tolist(),
            "training_energy": float(training_result.energy),
        },
    )
    inference_result = provider.run(package)
    deployed_energy = fq.hamiltonian_expectation_from_counts(
        inference_result.counts,
        hamiltonian,
    )

    print("optimized_parameters:", optimized_parameters.tolist())
    print("training_energy:", float(training_result.energy))
    print("deployment_counts:", dict(inference_result.counts))
    print("deployment_energy_estimate:", float(deployed_energy))
    print("qasm:")
    print(package.qasm)

    print("\n" + "="*60)
    print("THEORETICAL EXACT DIAGONALIZATION:")
    print(f"  Ground state energy: -1.2")
    print(f"  Full spectrum: [-1.2, -0.8, 0.8, 1.2]")
    print("="*60)

    print("\nVALIDATION:")
    gap = float(deployed_energy) - (-1.2)
    print(f"  Gap to ground state: {gap:.10f}")

    tolerance = 1e-6
    if gap >= -tolerance:
        if abs(gap) <= tolerance:
            print(f"  ✅ PERFECT: Reached ground state within {tolerance} tolerance")
        else:
            print(f"  ✅ Energy is physically valid (above ground state by {gap:.8f})")
    else:
        print(f"  ❌ ERROR: Energy is below ground state by {-gap:.8f}!")
        print("     This indicates either:")
        print("     - Hamiltonian mapping has sign errors")
        print("     - Counts weighting is incorrect")
        print("     - Bug in hamiltonian_expectation_from_counts")


if __name__ == "__main__":
    main()
