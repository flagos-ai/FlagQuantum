"""Run original Qmod pair gates and their native and Circuit counterparts."""

from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, allocate, hadamard_transform, qfunc

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"


@qfunc
def zero_pair(first: Output[QBit], second: Output[QBit]) -> None:
    # All three original programs leave their freshly allocated pair in |00>.
    allocate(first)
    allocate(second)


@qfunc
def bell(q_target: Output[QBit], q_control: Output[QBit]) -> None:
    # Fresh coherent assignment expresses CX with a zero-initialized target.
    allocate(q_control)
    hadamard_transform(q_control)
    q_target |= q_control


def main() -> None:
    native = fq.run(synth.synthesize(zero_pair)).statevector()
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
        torch.testing.assert_close(imported, native, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            imported, fq.run(circuit).statevector(), atol=1e-12, rtol=0
        )
        print(name, dict(qmod.probabilities(result, registers=tuple(qmod.registers))))
    prepared = synth.synthesize(bell)
    print("Native Bell state:", fq.run(prepared).statevector())


if __name__ == "__main__":
    main()
