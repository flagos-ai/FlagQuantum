"""Source/native indexed collections feed the same component binding consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import sympy

from flagquantum.compiler.registers.component_bindings import create_register_bindings
from flagquantum.core.ir.program.model import ValueRef
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.value_math import SymbolicInt
from tests.unit.test_register_bindings import (
    BindingContract,
    BindingT,
    Index,
    NativeContract,
    SourceContract,
)

if TYPE_CHECKING:
    from qualtran._infra.quantum_graph import Soquet


class BindingFactory(Protocol[BindingT]):
    """The source/native collection factory shares availability ownership."""

    def __call__(
        self, register: Register, available: set[BindingT] | None = None
    ) -> BindingT | NDArray[np.object_]: ...


CreationBackend: TypeAlias = tuple[
    type[BindingContract[BindingT]], BindingFactory[BindingT]
]

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
N = sympy.Symbol("n", integer=True, positive=True)


def create_source(
    register: Register, available: set[Soquet] | None = None
) -> Soquet | NDArray[np.object_]:
    from qualtran._infra.composite_bloq import _reg_to_soq
    from qualtran._infra.quantum_graph import LeftDangle

    if available is None:
        return _reg_to_soq(LeftDangle, SourceContract.register(register))
    return _reg_to_soq(LeftDangle, SourceContract.register(register), available)


def create_native(
    register: Register, available: set[RegisterBinding] | None = None
) -> RegisterBinding | NDArray[np.object_]:
    from flagquantum.compiler.registers.component_bindings import (
        create_register_bindings,
    )

    return create_register_bindings(ValueRef("entry", 0, "qubit"), register, available)


@pytest.fixture(params=["source", "native"])
def backend(
    request: pytest.FixtureRequest,
) -> CreationBackend[Soquet] | CreationBackend[RegisterBinding]:
    return (
        (SourceContract, create_source)
        if request.param == "source"
        else (NativeContract, create_native)
    )


@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (0, 2), (2, 0), (N, 2), (-1, 2)])
@pytest.mark.parametrize("width", [0, 3, N])
@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU"])
def test_collections_preserve_shape_indices_and_optional_availability(
    backend: CreationBackend[BindingT],
    shape: tuple[SymbolicInt, ...],
    width: SymbolicInt,
    side: str,
) -> None:
    contract, create = backend
    register = Register("word", RegisterType("uint", width), shape, RegisterSide[side])
    available: set[BindingT] = set()
    if shape == (N, 2):
        with pytest.raises(ValueError, match="Cannot get real-valued shape"):
            create(register, available)
        assert available == set()
        return
    if shape == (-1, 2):
        with pytest.raises(ValueError, match="negative dimensions"):
            create(register, available)
        assert available == set()
        return
    result = create(register, available)
    untracked = create(register)
    if shape:
        assert isinstance(result, np.ndarray)
        assert isinstance(untracked, np.ndarray)
        assert result.shape == shape
        assert result.dtype == object
        assert list(result.flat) == list(untracked.flat)
        cells = list(result.flat)
    else:
        assert not isinstance(result, np.ndarray)
        assert result == untracked
        cells = [result]
    assert available == set(cells)
    assert [contract.index(binding) for binding in cells] == list(register.all_idxs())
    seen: list[tuple[Index, Index]] = []
    contract.process(
        [register],
        {"word": result},
        lambda value, port, index: seen.append((contract.index(value), index)),
    )
    assert seen == [(index, index) for index in register.all_idxs()]


def test_availability_registration_failure_retains_prefix(
    backend: CreationBackend[BindingT],
) -> None:
    contract, create = backend

    class FailingSet(set[BindingT]):
        def add(self, value: BindingT) -> None:
            if len(self) == 2:
                raise RuntimeError("availability registration failed")
            super().add(value)

    available: FailingSet = FailingSet()
    register = Register("word", RegisterType("uint", 3), (2, 2))
    with pytest.raises(RuntimeError, match="availability registration failed"):
        create(register, available)
    assert {contract.index(value) for value in available} == {(0, 0), (0, 1)}


def test_existing_availability_and_scalar_identity_are_preserved(
    backend: CreationBackend[BindingT],
) -> None:
    contract, create = backend
    first = Register("first", RegisterType("uint", 0))
    second = Register("second", RegisterType("uint", 3), (2,))
    available: set[BindingT] = set()
    scalar = create(first, available)
    shaped = create(second, available)
    assert not isinstance(scalar, np.ndarray)
    assert isinstance(shaped, np.ndarray)
    assert scalar in available
    assert available == {scalar, *shaped}
    assert contract.index(scalar) == ()


@pytest.mark.parametrize("shape", [(), (2, 3), (0, 2)])
def test_allocator_visits_exactly_the_same_elements_and_preserves_fixed_identity_callers(
    shape: tuple[int, ...],
) -> None:
    register = Register("word", RegisterType("uint", 0), shape)
    seen: list[ValueRef] = []

    def allocate() -> ValueRef:
        value = ValueRef("entry", len(seen), "register")
        seen.append(value)
        return value

    created = create_register_bindings(allocate, register)
    assert [binding.value for binding in np.asarray(created).flat] == seen
    assert len(seen) == len(tuple(register.all_idxs()))
    fixed = ValueRef("entry", 42, "register")
    original = create_register_bindings(fixed, register)
    assert all(binding.value == fixed for binding in np.asarray(original).flat)


def test_allocator_failure_retains_prior_definitions_and_availability() -> None:
    from flagquantum.compiler.registers.component_bindings import (
        create_register_bindings,
    )

    available: set[RegisterBinding] = set()
    seen: list[ValueRef] = []

    def allocate() -> ValueRef:
        if len(seen) == 2:
            raise RuntimeError("allocation stopped")
        value = ValueRef("entry", len(seen), "register")
        seen.append(value)
        return value

    with pytest.raises(RuntimeError, match="allocation stopped"):
        create_register_bindings(
            allocate, Register("words", RegisterType("uint", 0), (2, 2)), available
        )
    assert {b.value for b in available} == set(seen)
    assert {b.index for b in available} == {(0, 0), (0, 1)}
