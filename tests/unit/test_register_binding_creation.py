"""Source/native indexed collections feed the same component binding consumers."""

import numpy as np
import pytest
import sympy

from flagquantum.compiler.registers.component_bindings import create_register_bindings
from flagquantum.core.ir.program.model import ValueRef
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from tests.unit.test_register_bindings import NativeContract, SourceContract

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
N = sympy.Symbol("n", integer=True, positive=True)


def create_source(register, available=None):
    from qualtran import LeftDangle
    from qualtran._infra.composite_bloq import _reg_to_soq

    if available is None:
        return _reg_to_soq(LeftDangle, SourceContract.register(register))
    return _reg_to_soq(LeftDangle, SourceContract.register(register), available)


def create_native(register, available=None):
    from flagquantum.compiler.registers.component_bindings import (
        create_register_bindings,
    )

    return create_register_bindings(ValueRef("entry", 0, "qubit"), register, available)


@pytest.fixture(params=["source", "native"])
def backend(request):
    return (
        (SourceContract, create_source)
        if request.param == "source"
        else (NativeContract, create_native)
    )


@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (0, 2), (2, 0), (N, 2), (-1, 2)])
@pytest.mark.parametrize("width", [0, 3, N])
@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU"])
def test_collections_preserve_shape_indices_and_optional_availability(
    backend, shape, width, side
):
    contract, create = backend
    register = Register("word", RegisterType("uint", width), shape, RegisterSide[side])
    available = set()
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
    seen = []
    contract.process(
        [register],
        {"word": result},
        lambda value, port, index: seen.append((contract.index(value), index)),
    )
    assert seen == [(index, index) for index in register.all_idxs()]


def test_availability_registration_failure_retains_prefix(backend):
    contract, create = backend

    class FailingSet(set):
        def add(self, value):
            if len(self) == 2:
                raise RuntimeError("availability registration failed")
            super().add(value)

    available = FailingSet()
    register = Register("word", RegisterType("uint", 3), (2, 2))
    with pytest.raises(RuntimeError, match="availability registration failed"):
        create(register, available)
    assert {contract.index(value) for value in available} == {(0, 0), (0, 1)}


def test_existing_availability_and_scalar_identity_are_preserved(backend):
    contract, create = backend
    first = Register("first", RegisterType("uint", 0))
    second = Register("second", RegisterType("uint", 3), (2,))
    available = set()
    scalar = create(first, available)
    shaped = create(second, available)
    assert scalar in available
    assert available == {scalar, *shaped}
    assert contract.index(scalar) == ()


@pytest.mark.parametrize("shape", [(), (2, 3), (0, 2)])
def test_allocator_visits_exactly_the_same_elements_and_preserves_fixed_identity_callers(
    shape,
):
    register = Register("word", RegisterType("uint", 0), shape)
    seen = []

    def allocate():
        value = ValueRef("entry", len(seen), "register")
        seen.append(value)
        return value

    created = create_register_bindings(allocate, register)
    assert [binding.value for binding in np.asarray(created).flat] == seen
    assert len(seen) == len(tuple(register.all_idxs()))
    fixed = ValueRef("entry", 42, "register")
    original = create_register_bindings(fixed, register)
    assert all(binding.value == fixed for binding in np.asarray(original).flat)


def test_allocator_failure_retains_prior_definitions_and_availability():
    from flagquantum.compiler.registers.component_bindings import (
        create_register_bindings,
    )

    available = set()
    seen = []

    def allocate():
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
