"""Native-environment conversion contracts; real upstream agreement runs separately."""

from collections.abc import Callable
from types import ModuleType
from typing import cast

import pytest

from flagquantum.core.algebra.pauli import PauliOperator, PauliTerm
from flagquantum.ecosystem import openfermion as adapter
from flagquantum.observables.fermion import FermionOperator, FermionTerm

pytestmark = pytest.mark.unit


class _ForeignFermion:
    def __init__(self) -> None:
        self.terms: dict[object, object] = {}


class _ForeignQubit:
    def __init__(self) -> None:
        self.terms: dict[object, object] = {}


@pytest.fixture(autouse=True)
def foreign_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only the external object provider, keeping native consumers unchanged."""
    module = ModuleType("operator_provider")
    setattr(module, "FermionOperator", _ForeignFermion)
    setattr(module, "QubitOperator", _ForeignQubit)
    monkeypatch.setattr(adapter, "import_module", lambda name: module)


def test_missing_provider_reports_the_optional_installation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(adapter, "import_module", unavailable)
    with pytest.raises(ImportError, match="separately installed") as error:
        adapter.to_openfermion_fermion(FermionOperator())
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_ordered_contraction_and_foreign_mutation_preserve_native_value() -> None:
    source = _ForeignFermion()
    source.terms = {((0, 0), (0, 1)): 2}
    native = adapter.from_openfermion_fermion(source)
    assert native == FermionOperator((FermionTerm(2), FermionTerm(-2, (0,), (0,))))
    exported = adapter.to_openfermion_fermion(native)
    assert exported.terms == {(): 2, ((0, 1), (0, 0)): -2}
    source.terms.clear()
    exported.terms.clear()
    assert native == FermionOperator((FermionTerm(2), FermionTerm(-2, (0,), (0,))))


def test_pauli_axis_translation_complex_terms_and_ownership() -> None:
    source = _ForeignQubit()
    source.terms = {((2, "Z"), (0, "Y")): 2j, (): 1e-30}
    native = adapter.from_openfermion_qubit(source)
    expected = PauliOperator((PauliTerm(2j, ((0, "y"), (2, "z"))), PauliTerm(1e-30)))
    assert native == expected
    exported = adapter.to_openfermion_qubit(native)
    assert exported.terms == {((0, "Y"), (2, "Z")): 2j, (): 1e-30}
    source.terms.clear()
    exported.terms.clear()
    assert native == expected


@pytest.mark.parametrize("coefficient", [1e-30, 1 + 2j, 0])
def test_fermion_scalar_and_empty_operator_round_trips(coefficient: complex) -> None:
    native = FermionOperator((FermionTerm(coefficient),))
    exported = adapter.to_openfermion_fermion(native)
    assert adapter.from_openfermion_fermion(exported) == native
    assert (
        adapter.from_openfermion_qubit(adapter.to_openfermion_qubit(PauliOperator()))
        == PauliOperator()
    )


@pytest.mark.parametrize(
    "convert", [adapter.from_openfermion_fermion, adapter.from_openfermion_qubit]
)
@pytest.mark.parametrize("source", [None, {}, FermionOperator(), PauliOperator()])
def test_nonnative_input_is_rejected_before_consumption(
    convert: Callable[[object], object], source: object
) -> None:
    with pytest.raises(TypeError, match="source must be"):
        convert(source)


def test_export_rejects_foreign_and_wrong_native_algebras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="native FermionOperator"):
        adapter.to_openfermion_fermion(cast(FermionOperator, PauliOperator()))
    with pytest.raises(TypeError, match="native PauliOperator"):
        adapter.to_openfermion_qubit(cast(PauliOperator, FermionOperator()))
    source = _ForeignFermion()
    monkeypatch.setattr(source, "terms", [])
    with pytest.raises(TypeError, match="mapping"):
        adapter.from_openfermion_fermion(source)
    with pytest.raises(TypeError, match="source must be"):
        adapter.from_openfermion_qubit(_ForeignFermion())
    with pytest.raises(TypeError, match="source must be"):
        adapter.from_openfermion_fermion(_ForeignQubit())


@pytest.mark.parametrize(
    "word",
    [
        "0^",
        (0,),
        ((0,),),
        ((True, 1),),
        ((0.5, 1),),
        ((-1, 1),),
        ((0, 2),),
        ((0, True),),
    ],
)
def test_malformed_ladder_words_fail_at_the_boundary(word: object) -> None:
    source = _ForeignFermion()
    source.terms = {word: 1}
    with pytest.raises((TypeError, ValueError)):
        adapter.from_openfermion_fermion(source)


@pytest.mark.parametrize("word", [((0, "x"),), ((0, 1),), ((0, "X"), (0, "Y"))])
def test_malformed_pauli_words_fail_at_the_boundary(word: object) -> None:
    source = _ForeignQubit()
    source.terms = {word: 1}
    with pytest.raises((TypeError, ValueError)):
        adapter.from_openfermion_qubit(source)


@pytest.mark.parametrize("coefficient", [True, float("nan"), float("inf"), "symbol"])
def test_invalid_coefficients_are_not_admitted_to_native_algebras(
    coefficient: object,
) -> None:
    fermion = _ForeignFermion()
    fermion.terms = {(): coefficient}
    qubit = _ForeignQubit()
    qubit.terms = {(): coefficient}
    with pytest.raises((TypeError, ValueError)):
        adapter.from_openfermion_fermion(fermion)
    with pytest.raises((TypeError, ValueError)):
        adapter.from_openfermion_qubit(qubit)
