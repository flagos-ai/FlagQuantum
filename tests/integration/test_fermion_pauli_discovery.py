"""Discovered Pauli constraints replace supplied ones at existing consumers."""

import itertools
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.clifford import conjugate_pauli
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_chain
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.algorithms.fermion.stabilizer import (
    discover_pauli_symmetries,
    reduce_stabilizers,
)
from flagquantum.algorithms.fermion.symmetry import discover_z_parities
from flagquantum.core.algebra.pauli import pauli
from flagquantum.core.ir import Instruction
from flagquantum.observables import FermionOperator, FermionTerm, number

pytestmark = pytest.mark.integration


def test_all_discovered_joint_sectors_match_dense_projectors_and_native_execution() -> (
    None
):
    y0 = pauli(factors=((0, "y"),))
    x1z2 = pauli(factors=((1, "x"), (2, "z")))
    operator = (
        pauli(factors=((2, "z"),))
        + 0.4 * pauli(factors=((1, "z"), (2, "x")))
        + 0.2 * y0
        + 0.3 * x1z2
    )
    discovered = discover_pauli_symmetries(operator, n_qubits=3)
    assert discovered == (y0, x1z2)
    full = pauli_matrix(operator, n_qubits=3)
    identity = torch.eye(8, dtype=torch.complex128)
    matrices = [pauli_matrix(generator, n_qubits=3) for generator in discovered]
    for sectors in itertools.product((-1, 1), repeat=2):
        actual = reduce_stabilizers(
            operator, n_qubits=3, generators=discovered, eigenvalues=sectors
        )
        supplied = reduce_stabilizers(
            operator, n_qubits=3, generators=[y0, x1z2], eigenvalues=sectors
        )
        assert actual == supplied
        embedding = torch.stack(
            [
                lift_stabilizer_state(actual, column)
                for column in torch.eye(2, dtype=torch.complex128)
            ],
            dim=1,
        )
        projector = identity.clone()
        for matrix, sector in zip(matrices, sectors, strict=True):
            projector = projector @ (identity + sector * matrix) / 2
            torch.testing.assert_close(
                matrix @ embedding, sector * embedding, atol=1e-12, rtol=0
            )
        torch.testing.assert_close(
            embedding @ embedding.mH, projector, atol=1e-12, rtol=0
        )
        reduced_matrix = pauli_matrix(actual.operator, n_qubits=1)
        torch.testing.assert_close(
            full @ embedding, embedding @ reduced_matrix, atol=1e-12, rtol=0
        )
        # Independent sector spectrum: Y0 and X1Z2 contribute their requested
        # signed offsets; Z2 and Z1X2 anticommute and square to identity.
        expected = torch.tensor(
            [
                0.2 * sectors[0] + 0.3 * sectors[1] - 1.16**0.5,
                0.2 * sectors[0] + 0.3 * sectors[1] + 1.16**0.5,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            torch.linalg.eigvalsh(reduced_matrix), expected, atol=1e-12, rtol=0
        )
        _, vectors = torch.linalg.eigh(projector)
        sector_basis = vectors[:, -2:]
        torch.testing.assert_close(
            torch.linalg.eigvalsh(sector_basis.mH @ full @ sector_basis),
            expected,
            atol=1e-12,
            rtol=0,
        )
        circuit = fq.Circuit(1, dtype=torch.complex128, device="cpu").ry(0, theta=0.37)
        measured = (
            fq.run(circuit, outputs=[fq.expectation(to_observable(actual.operator))])
            .expectation()
            .reshape(())
        )
        lifted = lift_stabilizer_state(actual, circuit.state().reshape(-1))
        torch.testing.assert_close(
            measured, (lifted.conj() @ full @ lifted).real, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_encoded_physical_constraints_replace_signed_suppliers(
    encoding: Encoding,
) -> None:
    encoded = encode_fermion(
        fermi_hubbard_chain(2, t=1, u=4), n_modes=4, encoding=encoding
    )
    identity = FermionOperator((FermionTerm(1),))
    parities = [
        (identity - 2 * number(spin)) * (identity - 2 * number(spin + 2))
        for spin in (0, 1)
    ]
    supplied = [
        encode_fermion(operator, n_modes=4, encoding=encoding).operator
        for operator in parities
    ]
    operator = encoded.operator
    reference = fq.Circuit(4, dtype=torch.complex128, device="cpu")
    for q, bit in enumerate(encoded.reference_bits((1, 1, 0, 0))):
        if bit:
            reference.x(q)
    # Rotate the encoded qubit frame, retaining the chosen physical reference.
    for gate in (
        Instruction("h", (0,)),
        Instruction("sdg", (0,)),
        Instruction("cx", (1, 2)),
    ):
        operator = conjugate_pauli(operator, gate)
        supplied = [conjugate_pauli(word, gate) for word in supplied]
    reference.h(0).sdg(0).cx(1, 2)
    state = reference.state().reshape(-1)
    discovered = discover_pauli_symmetries(operator, n_qubits=4, candidates=supplied)
    assert len(discovered) == 2
    eigenvalues = []
    # This deliberately selects the reference's joint sector. Discovery itself
    # does not infer sectors from total particle number or signed candidates.
    for word in discovered:
        value = torch.vdot(state, pauli_matrix(word, n_qubits=4) @ state)
        assert abs(float(value.imag)) < 1e-12
        assert abs(abs(float(value.real)) - 1) < 1e-12
        eigenvalues.append(1 if float(value.real) > 0 else -1)
    actual = reduce_stabilizers(
        operator, n_qubits=4, generators=discovered, eigenvalues=eigenvalues
    )
    explicit = reduce_stabilizers(
        operator, n_qubits=4, generators=supplied, eigenvalues=[-1, -1]
    )
    assert actual.n_qubits == explicit.n_qubits == 2
    torch.testing.assert_close(
        torch.linalg.eigvalsh(pauli_matrix(actual.operator, n_qubits=2)),
        torch.linalg.eigvalsh(pauli_matrix(explicit.operator, n_qubits=2)),
        atol=1e-12,
        rtol=0,
    )
    embedding = torch.stack(
        [
            lift_stabilizer_state(actual, column)
            for column in torch.eye(4, dtype=torch.complex128)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        pauli_matrix(operator, n_qubits=4) @ embedding,
        embedding @ pauli_matrix(actual.operator, n_qubits=2),
        atol=1e-12,
        rtol=0,
    )
    for word, sector in zip(supplied, (-1, -1), strict=True):
        torch.testing.assert_close(
            pauli_matrix(word, n_qubits=4) @ embedding,
            sector * embedding,
            atol=1e-12,
            rtol=0,
        )


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_diagonal_candidate_span_preserves_the_existing_z_discovery(
    encoding: Encoding,
) -> None:
    encoded = encode_fermion(
        fermi_hubbard_chain(2, t=1, u=4), n_modes=4, encoding=encoding
    )
    discovered = discover_pauli_symmetries(
        encoded.operator,
        n_qubits=4,
        candidates=[pauli(factors=((q, "z"),)) for q in range(4)],
    )
    expected = tuple(
        pauli(factors=tuple((q, "z") for q in word))
        for word in discover_z_parities(encoded)
    )
    assert discovered == expected


def test_discovery_example_without_external_science_dependencies() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        blocked = {"openfermion", "cirq", "pyscf", "scipy", "h5py", "numpy"}
        class NoScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoScience())
        from examples.algorithms.fermion_pauli_discovery import main
        main()
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
        """,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
