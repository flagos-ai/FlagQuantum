"""FCIDUMP input contracts and unchanged native chemistry consumers."""

from pathlib import Path

import pytest
import torch

from flagquantum.algorithms.chemistry_domain.active_space import select_active_space
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
)
from flagquantum.algorithms.fermion.encoding import Encoding, encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump

from .oracles import fock_matrix

pytestmark = pytest.mark.unit


def _file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "FCIDUMP"
    path.write_text(text, encoding="utf-8")
    return path


def _dump(problem: MolecularProblem) -> str:
    n = problem.n_orbitals
    records = [f" &FCI NORB={n}, NELEC={problem.n_electrons}, MS2=0,", " &END"]
    pairs = [(i, j) for i in range(n) for j in range(i + 1)]
    for index, (i, j) in enumerate(pairs):
        for k, last in pairs[: index + 1]:
            records.append(
                f"{float(problem.two_body[i, j, k, last]):.17e} {i+1} {j+1} {k+1} {last+1}"
            )
        records.append(f"{float(problem.one_body[i, j]):.17e} {i+1} {j+1} 0 0")
    records.append(f"{problem.energy_offset:.17e} 0 0 0 0")
    return "\n".join(records)


@pytest.mark.parametrize("terminator", ["&END", "/"])
def test_restricted_input_replaces_the_producer_for_native_consumers(
    tmp_path: Path, terminator: str
) -> None:
    generator = torch.Generator().manual_seed(211)
    factor = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    factor = factor + factor.T
    one = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    original = MolecularProblem(
        one + one.T,
        torch.einsum("pq,rs->pqrs", factor, factor),
        4,
        nuclear_repulsion=0.7,
        core_energy=-1.4,
    )
    problem = read_fcidump(_file(tmp_path, _dump(original).replace("&END", terminator)))
    torch.testing.assert_close(problem.one_body, original.one_body)
    torch.testing.assert_close(problem.two_body, original.two_body)
    assert problem.nuclear_repulsion == 0
    assert problem.core_energy == pytest.approx(-0.7)
    assert problem.reference_energy is None
    assert problem.reference_occupation == original.reference_occupation
    encodings: tuple[Encoding, ...] = ("jordan_wigner", "parity", "bravyi_kitaev")
    for encoding in encodings:
        actual = encode_fermion(
            fermion_hamiltonian(problem), n_modes=6, encoding=encoding
        )
        expected = encode_fermion(
            fermion_hamiltonian(original), n_modes=6, encoding=encoding
        )
        torch.testing.assert_close(
            pauli_matrix(actual.operator, n_qubits=6),
            pauli_matrix(expected.operator, n_qubits=6),
        )
    active = select_active_space(
        problem, active_orbitals=(2, 0), frozen_occupied=(1,)
    ).problem
    expected_active = select_active_space(
        original, active_orbitals=(2, 0), frozen_occupied=(1,)
    ).problem
    torch.testing.assert_close(
        fock_matrix(fermion_hamiltonian(active), 4),
        fock_matrix(fermion_hamiltonian(expected_active), 4),
    )


def test_comments_fortran_exponents_symmetry_and_zero_records(tmp_path: Path) -> None:
    problem = read_fcidump(
        _file(
            tmp_path,
            """
! Independent two-orbital input
 &fci norb=2, nelec=2, ms2=0,
 orbsym=1,2, isym=1, iuhf=0 /
! The constant is not split into nuclear and core contributions.
 0.25d+00 1 2 1 2

 0.25D+00 2 1 2 1 ! equivalent eightfold duplicate
 -1D0 1 1 0 0
 2D-1 2 1 0 0
 2D-1 1 2 0 0
 0.75 0 0 0 0
""",
        )
    )
    torch.testing.assert_close(
        problem.one_body, torch.tensor([[-1.0, 0.2], [0.2, 0.0]], dtype=torch.float64)
    )
    assert problem.two_body[0, 1, 0, 1] == 0.25
    assert problem.two_body[1, 0, 1, 0] == 0.25
    assert problem.two_body[0, 0, 1, 1] == 0
    reference = fock_matrix(fermion_hamiltonian(problem), 4)
    assert reference[12, 12].real == pytest.approx(-1.25)


@pytest.mark.parametrize(
    "header,match",
    [
        ("&FCI NORB=2,NELEC=2,MS2=0", "terminator"),
        ("&OTHER NORB=2,NELEC=2,MS2=0 /", "start"),
        ("&FCI /", "named integer"),
        ("&FCI BAD NORB=2,NELEC=2,MS2=0 /", "named integer"),
        ("&FCI NORB=2,NELEC=2 /", "requires NORB"),
        ("&FCI NORB=2,NELEC=2,MS2=0,NORB=2 /", "duplicate"),
        ("&FCI NORB=2,NELEC=2,MS2=0,COMPLEX=1 /", "unsupported"),
        ("&FCI NORB=2,NELEC=2,MS2=0,IUHF=1 /", "restricted"),
        ("&FCI NORB=2,NELEC=2,MS2=2 /", "restricted"),
        ("&FCI NORB=2,NELEC=3,MS2=0 /", "NELEC"),
        ("&FCI NORB=2,NELEC=6,MS2=0 /", "NELEC"),
        ("&FCI NORB=2,NELEC=-2,MS2=0 /", "NELEC"),
        ("&FCI NORB=0,NELEC=0,MS2=0 /", "budget"),
        ("&FCI NORB=33,NELEC=2,MS2=0 /", "budget"),
        ("&FCI NORB=2,NELEC=2,MS2=0,ORBSYM=1 /", "ORBSYM"),
        ("&FCI NORB=2,NELEC=2,MS2=0,ORBSYM=1,-1 /", "ORBSYM"),
        ("&FCI NORB=2,NELEC=2,MS2=0,ISYM=-1 /", "ISYM"),
        ("&FCI NORB=2,3,NELEC=2,MS2=0 /", "one integer"),
        ("&FCI NORB=2.0,NELEC=2,MS2=0 /", "integer values"),
        ("&FCI NORB=2,NELEC=2,MS2=0 / 0.1 0 0 0 0", "terminator"),
    ],
)
def test_rejects_invalid_or_unsupported_headers(
    tmp_path: Path, header: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        read_fcidump(_file(tmp_path, header))


@pytest.mark.parametrize(
    "record,match",
    [
        ("1 1 0 0", "four indices"),
        ("1 1 1 0 0 extra", "four indices"),
        ("x 1 1 0 0", "invalid"),
        ("1 1.0 1 0 0", "invalid"),
        ("NaN 1 1 0 0", "finite"),
        ("1D999 1 1 0 0", "finite"),
        ("1 -1 1 0 0", "outside NORB"),
        ("1 3 1 0 0", "outside NORB"),
        ("1 1 0 0 0", "unsupported.*pattern"),
        ("1 0 0 1 1", "unsupported.*pattern"),
        ("1 1 1 1 0", "unsupported.*pattern"),
        ("1 1 2 0 0\n2 2 1 0 0", "conflicting"),
        ("1 1 2 1 1\n2 1 1 2 1", "conflicting"),
        ("1 0 0 0 0\n2 0 0 0 0", "conflicting"),
    ],
)
def test_rejects_bad_integral_records(tmp_path: Path, record: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        read_fcidump(_file(tmp_path, "&FCI NORB=2,NELEC=2,MS2=0 /\n" + record))


def test_budget_and_file_errors_are_explicit(tmp_path: Path) -> None:
    path = _file(tmp_path, "&FCI NORB=2,NELEC=2,MS2=0 /\n")
    with pytest.raises(ValueError, match="budget"):
        read_fcidump(path, max_orbitals=1)
    with pytest.raises(ValueError, match="positive"):
        read_fcidump(path, max_orbitals=0)
    with pytest.raises(TypeError):
        read_fcidump(path, max_orbitals=True)
    problem = read_fcidump(path, max_orbitals=2)
    assert problem.energy_offset == 0
    assert torch.count_nonzero(problem.two_body) == 0
    with pytest.raises(FileNotFoundError):
        read_fcidump(tmp_path / "missing")
    with pytest.raises(ValueError, match="65536"):
        read_fcidump(_file(tmp_path, "&FCI " + " ".join(["X"] * 65536)))
