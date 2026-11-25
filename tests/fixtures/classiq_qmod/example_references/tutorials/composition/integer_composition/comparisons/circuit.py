"""Compare Qmod and native qfunc against the independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.tutorials.composition.integer_composition.main import (
    main,
)
from flagquantum import synth

QMOD = Path(__file__).with_name("source.qmod").read_text(encoding="utf-8")


def run_example() -> None:
    reference = (
        fq.run(fq.Circuit(2, dtype=torch.complex128).h(0).h(1).cphase(0, 1, theta=1.2))
        .statevector()
        .reshape(-1)
    )
    programs = (
        synth.synthesize(QMOD, source_format="qmod"),
        synth.synthesize(main),
    )
    for program in programs:
        state = fq.run(program).statevector().reshape(-1)
        workspace = program.circuit.n_wires - 2
        expected = torch.zeros_like(state)
        expected[:: 1 << workspace] = reference
        torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print(
        "Qmod integer composition, static native qfunc and Circuit agree, including global phase."
    )


if __name__ == "__main__":
    run_example()
