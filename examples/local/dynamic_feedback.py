"""Run a local dynamic circuit with measurement, feedback, and reset."""

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    circuit.reset(0)
    circuit.measure(0, classical_bit=1)

    options = fq.ExecutionOptions(shots=128, seed=7)
    plan = fq.plan(circuit, options=options)
    result = fq.run(plan)

    assert result.samples is not None
    classical_bits = result.measurement("mid_circuit_measurement").value
    assert isinstance(classical_bits, torch.Tensor)
    assert torch.equal(result.samples[:, 1], classical_bits[:, 0])
    assert torch.count_nonzero(result.samples[:, 0]) == 0
    assert torch.count_nonzero(classical_bits[:, 1]) == 0
    assert result.plan is plan
    assert result.runtime["mid_circuit_measurements_available"] is True
    print(
        {
            "final_samples": result.samples[:8].tolist(),
            "classical_bits": classical_bits[:8].tolist(),
            "execution_semantics": result.runtime["mode"],
            "plan_identity": plan.identity,
        }
    )


if __name__ == "__main__":
    main()
