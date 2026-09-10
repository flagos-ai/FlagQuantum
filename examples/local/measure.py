"""Request exact and sampled measurements from one local circuit."""

import torch

import flagquantum as fq


def main() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    probabilities = fq.run(circuit, outputs=fq.probabilities()).probabilities
    correlation = fq.run(
        circuit,
        outputs=fq.expectation(fq.Z(0) @ fq.Z(1)),
    ).expectation()
    counts = fq.run(circuit, outputs=fq.counts(), shots=1024).counts[0]

    assert probabilities.shape == (1, 4)
    torch.testing.assert_close(correlation, torch.ones_like(correlation))
    assert sum(counts.values()) == 1024
    assert set(counts) <= {"00", "11"}
    print({"probabilities": probabilities, "ZZ": correlation, "counts": counts})


if __name__ == "__main__":
    main()
