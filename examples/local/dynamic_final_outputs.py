"""Get final samples and counts from one run with mid-circuit feedback."""

from collections import Counter

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    circuit = (
        DynamicCircuit(2)
        .h(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
    )
    result = fq.run(
        circuit,
        outputs=(fq.samples(name="final"), fq.counts()),
        options=fq.ExecutionOptions(seed=7),
        shots=100,
    )

    mid = result.measurement("mid_circuit_measurement").value
    final = result.measurement("final").value
    counts = result.counts[0]

    assert isinstance(mid, torch.Tensor)
    assert isinstance(final, torch.Tensor)
    assert result.samples is final
    assert torch.equal(final[:, 0], mid[:, 0])
    assert torch.equal(final[:, 1], mid[:, 0])
    assert counts == dict(Counter("".join(map(str, row)) for row in final.tolist()))
    assert sum(counts.values()) == 100
    print("mid", tuple(mid.shape))
    print("final", tuple(final.shape))
    print("counts total", sum(counts.values()))


if __name__ == "__main__":
    main()
