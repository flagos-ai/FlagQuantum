"""Native acceptance of the optional in-memory molecular input boundary."""

from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import torch

from flagquantum.ecosystem.chemistry import molecular_data as adapter

pytestmark = pytest.mark.unit


class _Array:
    def __init__(self, values: torch.Tensor, kind: str = "f") -> None:
        self.values = values
        self.shape = tuple(values.shape)
        self.dtype = SimpleNamespace(kind=kind)

    def astype(self, dtype: str, *, copy: bool) -> "_Array":
        assert dtype == "float64" and copy
        return _Array(self.values.to(torch.float64).clone())


class _Molecule:
    def __init__(self) -> None:
        self.n_orbitals = 2
        self.n_electrons = 2
        self.multiplicity = 1
        self.n_qubits = 4
        self.nuclear_repulsion = 0.5
        self.hf_energy = -1.5
        self._one_body_integrals = _Array(torch.diag(torch.tensor([-1.0, -0.2])))
        self._two_body_integrals = _Array(torch.zeros((2,) * 4))


@pytest.fixture
def molecule(monkeypatch: pytest.MonkeyPatch) -> _Molecule:
    upstream = ModuleType("openfermion")
    setattr(upstream, "MolecularData", _Molecule)
    numpy = ModuleType("numpy")
    setattr(numpy, "ndarray", _Array)
    monkeypatch.setattr(adapter, "_upstream", lambda: upstream)
    monkeypatch.setattr(adapter, "import_module", lambda name: numpy)
    monkeypatch.setattr(
        adapter, "torch", SimpleNamespace(from_numpy=lambda array: array.values)
    )
    return _Molecule()


def test_owned_snapshots_and_default_reference(molecule: _Molecule) -> None:
    problem = adapter.from_openfermion_molecular_data(molecule)
    molecule._one_body_integrals.values.fill_(99)
    molecule._two_body_integrals.values.fill_(99)
    assert problem.n_modes == 4
    assert problem.reference_occupation == (1, 1, 0, 0)
    assert problem.core_energy == 0
    assert problem.energy_offset == 0.5
    assert problem.reference_energy == -1.5
    torch.testing.assert_close(
        problem.one_body, torch.diag(torch.tensor([-1.0, -0.2], dtype=torch.float64))
    )
    assert not bool(problem.two_body.any())


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_orbitals", 0),
        ("n_orbitals", 33),
        ("n_orbitals", True),
        ("n_electrons", 3),
        ("n_electrons", 6),
        ("n_electrons", -2),
        ("n_electrons", 2.0),
        ("n_electrons", torch.tensor(2)),
        ("multiplicity", 3),
        ("multiplicity", True),
        ("n_qubits", 3),
        ("n_qubits", False),
        ("nuclear_repulsion", float("nan")),
        ("nuclear_repulsion", float("inf")),
        ("nuclear_repulsion", 0.5j),
        ("nuclear_repulsion", True),
        ("hf_energy", float("nan")),
        ("hf_energy", False),
    ],
)
def test_corrupted_metadata_fails(
    molecule: _Molecule, field: str, value: object
) -> None:
    setattr(molecule, field, value)
    with pytest.raises((TypeError, ValueError)):
        adapter.from_openfermion_molecular_data(molecule)


@pytest.mark.parametrize("field", ["_one_body_integrals", "_two_body_integrals"])
@pytest.mark.parametrize(
    "value", [None, [], _Array(torch.zeros(2)), _Array(torch.zeros((2, 2)), kind="b")]
)
def test_missing_or_malformed_cached_arrays_fail(
    molecule: _Molecule, field: str, value: object
) -> None:
    setattr(molecule, field, value)
    with pytest.raises((TypeError, ValueError)):
        adapter.from_openfermion_molecular_data(molecule)


@pytest.mark.parametrize("field", ["_one_body_integrals", "_two_body_integrals"])
def test_nonfinite_integrals_fail(molecule: _Molecule, field: str) -> None:
    array = getattr(molecule, field)
    array.values.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        adapter.from_openfermion_molecular_data(molecule)


def test_native_integral_symmetry_is_authoritative(molecule: _Molecule) -> None:
    molecule._one_body_integrals.values[0, 1] = 0.3
    with pytest.raises(ValueError, match="symmetric"):
        adapter.from_openfermion_molecular_data(molecule)


def test_optional_metadata_can_be_absent(molecule: _Molecule) -> None:
    setattr(molecule, "hf_energy", None)
    setattr(molecule, "n_qubits", None)
    problem = adapter.from_openfermion_molecular_data(molecule)
    assert problem.reference_energy is None and problem.n_modes == 4


@pytest.mark.parametrize("limit", [0, -1, True, 2.0, torch.tensor(2)])
def test_invalid_budget_fails_before_loading_upstream(
    limit: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> ModuleType:
        raise AssertionError("upstream must not load for an invalid budget")

    monkeypatch.setattr(adapter, "_upstream", forbidden)
    # Invalid dynamic inputs intentionally exercise the external call boundary.
    with pytest.raises((TypeError, ValueError)):
        adapter.from_openfermion_molecular_data(None, max_orbitals=cast(int, limit))


def test_wrong_external_type_fails(molecule: _Molecule) -> None:
    with pytest.raises(TypeError, match="MolecularData"):
        adapter.from_openfermion_molecular_data(object())


def test_budget_is_checked_before_any_integral_copy(molecule: _Molecule) -> None:
    setattr(molecule, "_one_body_integrals", None)
    with pytest.raises(ValueError, match="budget"):
        adapter.from_openfermion_molecular_data(molecule, max_orbitals=1)
