"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.functions.amplitude_amplification.exact_amplitude_amplification.main import (
    BETA,
    THETA,
    main,
)
from flagquantum import synth


def run_example() -> None:
    c = fq.Circuit(2, dtype=torch.complex128).ry(0, theta=THETA).ry(1, theta=BETA)
    for _ in range(3):
        c.x(1).cz(0, 1).x(1)
        c.ry(1, theta=-BETA).ry(0, theta=-THETA)
        c.z(0).z(1).cz(0, 1)
        c.ry(0, theta=THETA).ry(1, theta=BETA)
    reference = fq.run(c).statevector().reshape(-1)
    for program in (synth.synthesize(main),):
        state = fq.run(program).statevector().reshape(4, -1)
        torch.testing.assert_close(state[:, 0], reference, atol=1e-12, rtol=0)
        assert float(state[:, 1:].abs().square().sum()) < 1e-24
        print(program.report["resources"])
    print(
        "Native qfunc and Circuit agree, including absolute phase and clean workspace."
    )


if __name__ == "__main__":
    run_example()
