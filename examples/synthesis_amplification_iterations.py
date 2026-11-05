"""Static amplification in Qmod, existing native scalar qfunc, and Circuit.

Native programs spell out zero/one/two rounds with existing phase/CX primitives.
No native amplification export, mutable helper port or Python loop is added.
Run tools/validate_qmod_iterations.py for the three unchanged counting/QMCI files.
"""

import math

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Const,
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

PI = math.pi


@qfunc
def reflect(flag: Const[QBit], data: Const[QBit]) -> None:
    # Odd parity on every nonzero word implements 2|00><00|-I.
    phase(flag + data + flag * data, PI)


@qfunc
def zero(flag: Output[QBit], data: Output[QBit]) -> None:
    allocate(flag)
    allocate(data)
    hadamard_transform(data)
    CX(data, flag)


@qfunc
def one(flag: Output[QBit], data: Output[QBit]) -> None:
    allocate(flag)
    allocate(data)
    hadamard_transform(data)
    CX(data, flag)
    phase(flag, PI)
    CX(data, flag)
    hadamard_transform(data)
    reflect(flag, data)
    hadamard_transform(data)
    CX(data, flag)


@qfunc
def two(flag: Output[QBit], data: Output[QBit]) -> None:
    allocate(flag)
    allocate(data)
    hadamard_transform(data)
    CX(data, flag)
    phase(flag, PI)
    CX(data, flag)
    hadamard_transform(data)
    reflect(flag, data)
    hadamard_transform(data)
    CX(data, flag)
    phase(flag, PI)
    CX(data, flag)
    hadamard_transform(data)
    reflect(flag, data)
    hadamard_transform(data)
    CX(data, flag)


QMOD = """qperm oracle(const q:qbit[]) {Z(q[q.len-1]);}
qfunc prepare(q:qbit[]) {H(q[0]);CX(q[0],q[1]);}
qfunc main(k:int,output flag:qbit,output data:qbit[1]) {
 packed:qbit[];allocate(flag);allocate(data);
 within {{data,flag}->packed;} apply {amplitude_amplification(k,oracle,prepare,packed);}
}"""


def run_example() -> None:
    # Physical order is flag, data; A applies H(data) followed by CX(data,flag).
    a = torch.tensor(
        [[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]],
        dtype=torch.complex128,
    ) / math.sqrt(2)
    oracle = torch.diag(torch.tensor([1, 1, -1, -1], dtype=torch.complex128))
    reflection = torch.diag(torch.tensor([1, -1, -1, -1], dtype=torch.complex128))
    grover = a @ reflection @ a.conj().T @ oracle
    for k, function in enumerate((zero, one, two)):
        circuit = fq.Circuit(2, dtype=torch.complex128).any(0, 1, matrix=a)
        for _ in range(k):
            circuit.any(0, 1, matrix=grover)
        expected = fq.run(circuit).statevector().reshape(-1)
        for p in (
            synth.synthesize(function),
            synth.synthesize(QMOD, source_format="qmod", classical_inputs={"k": k}),
        ):
            result = fq.run(p)
            state = result.statevector().reshape(4, -1)
            torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
            assert float(state[:, 1:].abs().square().sum()) < 1e-24
            print(k, dict(p.probabilities(result, registers=("flag",))))
            print(p.report["resources"])
    print(
        "Native scalar phase composition, Qmod and Circuit agree, including absolute round signs."
    )


if __name__ == "__main__":
    run_example()
