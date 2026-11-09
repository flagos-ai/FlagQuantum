"""Run original Qmod pair gates and their native and Circuit counterparts."""

from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CRX,
    CX,
    SWAP,
    Output,
    QBit,
    allocate,
    hadamard_transform,
    qfunc,
)

FIXTURES = Path(__file__).resolve().parents[5] / "tests/fixtures/classiq_qmod"


@qfunc
def native_pair(target: Output[QBit], control: Output[QBit]) -> None:
    allocate(target)
    allocate(control)
    hadamard_transform(control)
    CX(control, target)
    CRX(1.25, control, target)
    SWAP(target, control)


QMOD_PAIR = """qfunc main(output target:qbit, output control:qbit) {
    allocate(target);
    allocate(control);
    H(control);
    CX(control, target);
    CRX(1.25, control, target);
    SWAP(target, control);
}"""


def pair_circuit() -> fq.Circuit:
    return (
        fq.Circuit(2, dtype=torch.complex128)
        .h(1)
        .cx(1, 0)
        .crx(1, 0, theta=1.25)
        .swap(0, 1)
    )


def main() -> None:
    for name in ("CX", "SWAP", "CRX"):
        qmod = synth.synthesize(FIXTURES / f"{name}.qmod", source_format="qmod")
        circuit = fq.Circuit(2, dtype=torch.complex128)
        if name == "CX":
            circuit.cx(1, 0)
        elif name == "SWAP":
            circuit.swap(0, 1)
        else:
            circuit.crx(1, 0, theta=1.0)
        result = fq.run(qmod)
        imported = result.statevector()
        torch.testing.assert_close(
            imported, fq.run(circuit).statevector(), atol=1e-12, rtol=0
        )
        print(name, dict(qmod.probabilities(result, registers=tuple(qmod.registers))))
    native = fq.run(synth.synthesize(native_pair)).statevector()
    imported = fq.run(synth.synthesize(QMOD_PAIR, source_format="qmod")).statevector()
    reference = fq.run(pair_circuit()).statevector()
    torch.testing.assert_close(native, imported, atol=1e-12, rtol=0)
    torch.testing.assert_close(native, reference, atol=1e-12, rtol=0)
    print("Coherent pair gate state:", native)


if __name__ == "__main__":
    main()
