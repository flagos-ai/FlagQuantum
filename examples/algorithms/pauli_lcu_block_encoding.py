"""Verify a Pauli LCU block encoding and its gate-level qubitization walk."""

import math
from collections import Counter

import torch

import flagquantum as fq
from flagquantum.algorithms.primitives import arbitrary_state
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.compiler.lowering.deferred_operations import (
    lower_select_pauli_lcu_to_gates,
)
from flagquantum.simulation.unitary import get_unitary


def main() -> None:
    # H = 1.0 X - 0.6 Y + 0.4 Z. Use three index qubits to exercise ladder workspace.
    coefficients = (1.0, -0.6, 0.4)
    paulis = ("X", "Y", "Z")
    alpha = sum(abs(weight) for weight in coefficients)
    amplitudes = torch.tensor(
        [math.sqrt(abs(weight) / alpha) for weight in coefficients] + [0.0] * 5,
        dtype=torch.float64,
    )
    prepare = arbitrary_state(amplitudes)
    selection, target = (0, 1, 2), (3,)
    signed_paulis = tuple(
        "-" + pauli if weight < 0 else pauli
        for weight, pauli in zip(coefficients, paulis, strict=True)
    )
    program = fq.Circuit(4)
    op = SelectPauliLCU(3, 1, signed_paulis).make_on(selection=selection, target=target)
    program.append(op)
    select = lower_select_pauli_lcu_to_gates(program.to_ir().instructions[0])
    encoding = fq.Circuit(select.n_qubits)
    encoding.compose(prepare, qubits=selection)
    encoding.compose(select)
    encoding.compose(prepare.adjoint(), qubits=selection)

    # Workspace is the least significant qubit. Verify it is restored for
    # every basis input with clean workspace before extracting the block.
    unitary = get_unitary(encoding, dtype=torch.complex128)
    torch.testing.assert_close(
        unitary[1::2, ::2], torch.zeros_like(unitary[1::2, ::2]), atol=1e-7, rtol=0
    )
    clean_encoding = unitary[::2, ::2]
    block = clean_encoding[:2, :2]
    x = torch.tensor([[0, 1], [1, 0]], dtype=unitary.dtype)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=unitary.dtype)
    z = torch.tensor([[1, 0], [0, -1]], dtype=unitary.dtype)
    expected = (coefficients[0] * x + coefficients[1] * y + coefficients[2] * z) / alpha
    torch.testing.assert_close(block, expected, atol=1e-7, rtol=0)
    # R = 2|000><000| - I on the index. X-H-CCX-H-X flips only
    # the |000> phase, so negate that diagonal with XZXZ.
    reflection = fq.Circuit(3)
    reflection.x(0).x(1).x(2).h(2).ccx(0, 1, 2).h(2).x(2).x(1).x(0)
    reflection.x(0).z(0).x(0).z(0)
    walk = fq.Circuit(select.n_qubits).compose(encoding)
    walk.compose(reflection, qubits=selection)
    walk_matrix = get_unitary(walk, dtype=torch.complex128)[::2, ::2]

    # Each eigenvector of H defines a two-dimensional invariant walk space.
    eigenvalues, eigenvectors = torch.linalg.eigh(expected)
    for index, eigenvalue in enumerate(eigenvalues):
        flagged = torch.zeros(clean_encoding.shape[0], dtype=unitary.dtype)
        flagged[:2] = eigenvectors[:, index]
        partner = clean_encoding @ flagged - eigenvalue * flagged
        basis = torch.stack(
            (flagged, partner / torch.linalg.vector_norm(partner)), dim=1
        )
        projected = basis.mH @ walk_matrix @ basis
        torch.testing.assert_close(
            walk_matrix @ basis, basis @ projected, atol=1e-7, rtol=0
        )
        phases = torch.angle(torch.linalg.eigvals(projected)).sort().values
        phase = torch.acos(eigenvalue)
        torch.testing.assert_close(
            phases, torch.stack((-phase, phase)), atol=1e-7, rtol=0
        )

    counts = Counter(gate.name for gate in walk.to_ir().instructions)
    print(f"walk: {walk.n_qubits} qubits (1 workspace), {counts['ccx']} Toffoli gates")
    print("premise: Gate counts describe this circuit before further synthesis.")
    print(f"alpha = {alpha:g}; max block error = {(block - expected).abs().max():.3g}")


if __name__ == "__main__":
    main()
