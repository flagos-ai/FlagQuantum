"""Independent format fixtures protect indices, offsets and supported boundaries."""

from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.algorithms.chemistry_domain.active_space import select_active_space
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
)
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.ecosystem.chemistry.molecular_hdf5 import read_molecular_hdf5

h5py = pytest.importorskip("h5py")
pytestmark = pytest.mark.unit


@pytest.fixture
def molecule_file(tmp_path: Path) -> tuple[Path, MolecularProblem]:
    one = torch.tensor(
        [[-1.2, 0.12, 0.04], [0.12, -0.4, -0.03], [0.04, -0.03, 0.2]],
        dtype=torch.float64,
    )
    factors = torch.tensor(
        [
            [[0.5, 0.2, 0.1], [0.2, 0.3, -0.1], [0.1, -0.1, 0.4]],
            [[0.2, -0.3, 0.1], [-0.3, 0.1, 0.2], [0.1, 0.2, -0.2]],
        ],
        dtype=torch.float64,
    )
    weights = (1.0, -0.2)
    chemist = torch.einsum(
        "l,lpq,lrs->pqrs", torch.tensor(weights, dtype=torch.float64), factors, factors
    )
    physicist = torch.empty_like(chemist)
    for p in range(3):
        for q in range(3):
            for r in range(3):
                for s in range(3):
                    physicist[p, q, r, s] = sum(
                        weights[index] * factors[index, p, s] * factors[index, q, r]
                        for index in range(2)
                    )
    path = tmp_path / "molecule.hdf5"
    with h5py.File(path, "w") as source:
        for name, value in {
            "n_orbitals": 3,
            "n_electrons": 4,
            "multiplicity": 1,
            "n_qubits": 6,
            "nuclear_repulsion": 0.7,
            "hf_energy": -1.25,
        }.items():
            source[name] = value
        source.create_dataset(
            "one_body_integrals", data=one.numpy(), compression="gzip"
        )
        source.create_dataset(
            "two_body_integrals", data=physicist.numpy(), compression="gzip"
        )
        source["unconsumed_field"] = h5py.ExternalLink("absent-file", "anything")
    return path, MolecularProblem(one, chemist, 4, 0.7, reference_energy=-1.25)


def replace(path: Path, name: str, value: object) -> None:
    with h5py.File(path, "a") as source:
        if name in source:
            del source[name]
        source[name] = value


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_integrals_and_active_space_replace_native_producer(
    molecule_file: tuple[Path, MolecularProblem], encoding: Encoding
) -> None:
    path, expected = molecule_file
    actual = read_molecular_hdf5(path)
    path.unlink()  # Returned snapshots cannot retain an HDF5 resource.
    torch.testing.assert_close(actual.one_body, expected.one_body)
    torch.testing.assert_close(actual.two_body, expected.two_body)
    assert actual.energy_offset == 0.7 and actual.core_energy == 0
    assert actual.reference_energy == -1.25
    assert actual.reference_occupation == (1, 1, 1, 1, 0, 0)
    for producer in (actual, expected):
        active = select_active_space(
            producer, active_orbitals=(2, 1), frozen_occupied=(0,)
        ).problem
        encoded = encode_fermion(
            fermion_hamiltonian(active), n_modes=4, encoding=encoding
        )
        matrix = pauli_matrix(encoded.operator, n_qubits=4)
        if producer is actual:
            reference = matrix
        else:
            torch.testing.assert_close(matrix, reference)


@pytest.mark.parametrize("name,value", [("hf_energy", False), ("n_qubits", False)])
def test_missing_optional_sentinels(
    molecule_file: tuple[Path, MolecularProblem], name: str, value: bool
) -> None:
    path, _ = molecule_file
    replace(path, name, value)
    actual = read_molecular_hdf5(path)
    assert actual.reference_energy == (None if name == "hf_energy" else -1.25)


@pytest.mark.parametrize("name", ["hf_energy", "n_qubits"])
def test_absent_optional_fields(
    molecule_file: tuple[Path, MolecularProblem], name: str
) -> None:
    path, _ = molecule_file
    with h5py.File(path, "a") as source:
        del source[name]
    assert read_molecular_hdf5(path).n_orbitals == 3


@pytest.mark.parametrize(
    "name,value",
    [
        ("n_orbitals", False),
        ("n_orbitals", 0),
        ("n_orbitals", 3.0),
        ("n_electrons", True),
        ("n_electrons", -2),
        ("n_electrons", 3),
        ("n_electrons", 8),
        ("multiplicity", 2),
        ("multiplicity", 1.0),
        ("n_qubits", 4),
        ("n_qubits", True),
        ("nuclear_repulsion", False),
        ("nuclear_repulsion", complex(0.7, 0)),
        ("nuclear_repulsion", float("nan")),
        ("hf_energy", True),
        ("hf_energy", float("inf")),
        ("hf_energy", [-1.25]),
        ("one_body_integrals", False),
        ("two_body_integrals", [0.0]),
    ],
)
def test_reject_invalid_metadata(
    molecule_file: tuple[Path, MolecularProblem], name: str, value: object
) -> None:
    path, _ = molecule_file
    replace(path, name, value)
    with pytest.raises(ValueError):
        read_molecular_hdf5(path)


@pytest.mark.parametrize("name", ["one_body_integrals", "two_body_integrals"])
@pytest.mark.parametrize("change", ["complex", "nonfinite", "symmetry"])
def test_reject_invalid_integrals(
    molecule_file: tuple[Path, MolecularProblem], name: str, change: str
) -> None:
    path, _ = molecule_file
    with h5py.File(path, "r") as source:
        value = torch.from_numpy(source[name][()])
    if change == "complex":
        value = value.to(torch.complex128)
    elif change == "nonfinite":
        value.reshape(-1)[0] = float("nan")
    else:
        value.reshape(-1)[1] += 0.25
    replace(path, name, value.numpy())
    with pytest.raises(ValueError):
        read_molecular_hdf5(path)


def test_budget_before_integral_access(
    molecule_file: tuple[Path, MolecularProblem],
) -> None:
    path, _ = molecule_file
    replace(path, "n_orbitals", 33)
    with h5py.File(path, "a") as source:
        del source["one_body_integrals"]
    with pytest.raises(ValueError, match="budget"):
        read_molecular_hdf5(path)
    with pytest.raises(ValueError, match="budget"):
        read_molecular_hdf5(path, max_orbitals=2)


@pytest.mark.parametrize(
    "limit,error",
    [(False, ValueError), (0, ValueError), (-1, ValueError), (1.5, TypeError)],
)
def test_invalid_budget(tmp_path: Path, limit: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        read_molecular_hdf5(tmp_path / "absent", max_orbitals=limit)


@pytest.mark.parametrize("variant", ["missing", "soft", "external", "group", "virtual"])
def test_reject_nonembedded_dataset(
    molecule_file: tuple[Path, MolecularProblem], variant: str
) -> None:
    path, _ = molecule_file
    with h5py.File(path, "a") as source:
        del source["one_body_integrals"]
        if variant == "soft":
            source["one_body_integrals"] = h5py.SoftLink("/two_body_integrals")
        elif variant == "external":
            source["one_body_integrals"] = h5py.ExternalLink(
                "absent", "/one_body_integrals"
            )
        elif variant == "group":
            source.create_group("one_body_integrals")
        elif variant == "virtual":
            layout = h5py.VirtualLayout(shape=(3, 3), dtype="f8")
            layout[:] = h5py.VirtualSource("absent", "x", shape=(3, 3))
            source.create_virtual_dataset("one_body_integrals", layout)
    with pytest.raises(ValueError, match="embedded"):
        read_molecular_hdf5(path)


def test_missing_and_invalid_files(tmp_path: Path) -> None:
    path = tmp_path / "absent"
    with pytest.raises(OSError):
        read_molecular_hdf5(path)
    path.write_text("not HDF5")
    with pytest.raises(OSError):
        read_molecular_hdf5(path)


@pytest.mark.parametrize("storage", [">f8", "i8"])
def test_numeric_storage_and_zero_energy(
    molecule_file: tuple[Path, MolecularProblem], storage: str
) -> None:
    path, _ = molecule_file
    with h5py.File(path, "a") as source:
        del source["one_body_integrals"]
        source.create_dataset(
            "one_body_integrals", data=[[-1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=storage
        )
    replace(path, "hf_energy", 0.0)
    actual = read_molecular_hdf5(path)
    assert actual.reference_energy == 0.0
    assert actual.one_body.dtype == torch.float64
    assert actual.one_body[0, 0] == -1


def test_constant_enters_reference_energy_once(tmp_path: Path) -> None:
    path = tmp_path / "one-orbital"
    with h5py.File(path, "w") as source:
        for name, value in {
            "n_orbitals": 1,
            "n_electrons": 2,
            "multiplicity": 1,
            "nuclear_repulsion": 0.7,
        }.items():
            source[name] = value
        source["one_body_integrals"] = [[-1.2]]
        source["two_body_integrals"] = [[[[0.3]]]]
    problem = read_molecular_hdf5(path)
    assert problem.reference_energy is None
    encodings: tuple[Encoding, ...] = ("jordan_wigner", "parity", "bravyi_kitaev")
    for encoding in encodings:
        encoded = encode_fermion(
            fermion_hamiltonian(problem), n_modes=2, encoding=encoding
        )
        bits = encoded.reference_bits(problem.reference_occupation)
        index = 2 * bits[0] + bits[1]
        assert pauli_matrix(encoded.operator, n_qubits=2)[
            index, index
        ].real == pytest.approx(-1.4)


def test_missing_dependency_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    original = builtins.__import__

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "h5py":
            raise ModuleNotFoundError("h5py")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="chemistry-hdf5") as captured:
        read_molecular_hdf5(tmp_path / "absent")
    assert isinstance(captured.value.__cause__, ModuleNotFoundError)
