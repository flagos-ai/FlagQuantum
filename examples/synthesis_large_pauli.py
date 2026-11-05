"""Pauli Z evolution as native qfunc phase intent, Qmod and an independent Circuit.

Run this module for the small three-form example. Reproduce the unchanged
12-bit originals with tools/validate_qmod_large_pauli.py; this example does not
claim a native Python qfunc Hamiltonian-array API.
"""

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, allocate, hadamard_transform, phase, qfunc


@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    hadamard_transform(q)
    # exp(-i*0.37*Z): the zero and one amplitudes acquire opposite phases.
    phase(1, -0.37)
    phase(q, 0.74)


QMOD = """qfunc main(output q:qbit[1]) {
  allocate(q);hadamard_transform(q);
  suzuki_trotter([PauliTerm {pauli=[Pauli::Z],coefficient=1}],0.37,1,1,q);
}"""


def run_example() -> None:
    reference = (
        fq.run(fq.Circuit(1, dtype=torch.complex128).h(0).rz(0, theta=0.74))
        .statevector()
        .reshape(-1)
    )
    for program in (
        synth.synthesize(main),
        synth.synthesize(QMOD, source_format="qmod"),
    ):
        actual = fq.run(program).statevector().reshape(-1)
        workspace = program.circuit.n_wires - 1
        expected = torch.zeros_like(actual)
        expected[:: 1 << workspace] = reference
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print(
        "Native phase intent, compact Qmod Pauli evolution and Circuit agree, including global phase."
    )


if __name__ == "__main__":
    run_example()
