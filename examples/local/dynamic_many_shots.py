"""Run many local dynamic shots under an explicit memory budget."""

from __future__ import annotations

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)

    memory_limit_bytes = 64 * 1024**2
    options = fq.ExecutionOptions(
        shots=4096,
        seed=7,
        memory_limit_bytes=memory_limit_bytes,
    )
    plan = fq.plan(circuit, options=options)
    result = fq.run(plan)

    assert result.samples is not None
    classical_bits = result.measurement("mid_circuit_measurement").value
    assert isinstance(classical_bits, torch.Tensor)
    assert result.samples.shape == (4096, 2)
    assert torch.equal(result.samples[:, 1], classical_bits[:, 0])
    assert result.plan is plan
    print(
        {
            "sample_shape": tuple(result.samples.shape),
            "plan_identity": plan.identity,
            "memory_limit_mib": memory_limit_bytes // 1024**2,
        }
    )


if __name__ == "__main__":
    main()
