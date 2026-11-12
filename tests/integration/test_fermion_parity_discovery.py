"""Discovery replaces explicit parities without modifying reduction consumers."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_chain
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.symmetry import (
    discover_z_parities,
    reduce_z_parities,
)
from flagquantum.observables import FermionOperator, FermionTerm, number

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_total_number_parity_replaces_supplied_generator(encoding: str) -> None:
    hamiltonian = fermi_hubbard_chain(2, t=1, u=4)
    encoded = encode_fermion(hamiltonian, n_modes=4, encoding=encoding)
    # Build physical parity with existing algebra/encoding; encoded Z supports
    # are not assumed to equal occupation-mode indices for parity or BK.
    identity = FermionOperator((FermionTerm(1),))
    parity = identity
    for mode in range(4):
        parity = parity * (identity - 2 * number(mode))
    image = encode_fermion(parity, n_modes=4, encoding=encoding).operator
    assert len(image.terms) == 1
    term = image.terms[0]
    assert term.coefficient == 1
    assert all(axis == "z" for _, axis in term.factors)
    supplied = tuple(q for q, _ in term.factors)
    discovered = discover_z_parities(encoded, candidates=[supplied])
    assert discovered == (supplied,)
    for eigenvalue in (-1, 1):
        explicit = reduce_z_parities(
            encoded, generators=[supplied], eigenvalues=[eigenvalue]
        )
        actual = reduce_z_parities(
            encoded, generators=discovered, eigenvalues=[eigenvalue]
        )
        assert actual == explicit
        embedding = torch.stack(
            [
                actual.mapping.lift_state(column)
                for column in torch.eye(8, dtype=torch.complex128)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            pauli_matrix(encoded.operator, n_qubits=4) @ embedding,
            embedding @ pauli_matrix(actual.operator, n_qubits=3),
        )
    # Even parity contains N=0 and N=4 as well as N=2. Discovery must not
    # silently replace an explicit number projection with a parity constraint.
    even = reduce_z_parities(encoded, generators=discovered, eigenvalues=[1])
    allowed_bits = {
        even.mapping.full_bits(tuple((i >> (2 - q)) & 1 for q in range(3)))
        for i in range(8)
    }
    counts = {
        index.bit_count()
        for index in range(16)
        if encoded.reference_bits(tuple((index >> (3 - q)) & 1 for q in range(4)))
        in allowed_bits
    }
    assert counts == {0, 2, 4}


def test_discovery_workflow_without_external_science() -> None:
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
        from examples.algorithms.fermion_parity_discovery import main
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
