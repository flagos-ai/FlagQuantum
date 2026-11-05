"""Unchanged hidden-shift Qmod, native phase composition and public Circuit.

The native form composes existing qfunc functions; this batch does not add
higher-order Python callback annotations or a native phase_oracle export.
"""

import math
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def oracle(a: Const[QBit], b: Const[QBit], c: Const[QBit], d: Const[QBit]) -> None:
    phase(a * b + c * d, math.pi)


@qfunc
def shifted(a: Const[QBit], b: Const[QBit], c: Const[QBit], d: Const[QBit]) -> None:
    phase(a * b, math.pi)
    phase(c, math.pi)
    phase(c * d, -math.pi)


@qfunc
def main(a: Output[QBit], b: Output[QBit], c: Output[QBit], d: Output[QBit]) -> None:
    allocate(a)
    allocate(b)
    allocate(c)
    allocate(d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    shifted(a, b, c, d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    oracle(a, b, c, d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)


def run_example() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/classiq_qmod/hidden_shift.qmod"
    )
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for bit in range(4):
        circuit.h(bit)
    circuit.x(3).cz(3, 2).x(3).cz(1, 0)
    for bit in range(4):
        circuit.h(bit)
    circuit.cz(3, 2).cz(1, 0)
    for bit in range(4):
        circuit.h(bit)
    expected = fq.run(circuit).statevector().reshape(-1)
    for program in (synth.synthesize(source), synth.synthesize(main)):
        actual = fq.run(program).statevector().reshape(16, -1)
        torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
        assert float(actual[:, 1:].abs().square().sum()) < 1e-24
        print(program.report["resources"])
    print(
        "Original Qmod, native qfunc and Circuit agree: +|0001>, with clean workspace."
    )


if __name__ == "__main__":
    run_example()
