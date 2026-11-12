"""Typed binding and collection preserve the unchanged source state path."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import networkx as nx
import numpy as np
import pytest

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.simulation.classical_distributions import ClassicalDistribution
from flagquantum.simulation.classical_typed_values import (
    bind_typed_register_values,
    collect_typed_register_value,
)
from tests.unit.test_classical_register_codec import KINDS, pair
from tests.unit.test_finite_field_metadata import CASES, field_pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def compare(
    dtype: RegisterType, source: Any, shape: tuple[int, ...], value: object
) -> None:
    from qualtran import LeftDangle, Soquet
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import ClassicalSimState, _get_in_vals

    reg = Register("x", dtype, shape)
    original = SourceRegister("x", source, shape=shape)
    state = ClassicalSimState(SourceSignature([]), nx.DiGraph(), {})
    try:
        state._update_assign_from_vals([original], LeftDangle, {"x": value})
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)):
            bind_typed_register_values(reg, {"x": value})
        return
    assignments = bind_typed_register_values(reg, {"x": value})
    assert list(assignments) == list(reg.all_idxs())
    for index, actual in assignments.items():
        expected = state.soq_assign[Soquet(LeftDangle, original, idx=index)]
        np.testing.assert_equal(actual, expected)
        assert type(actual) is type(expected)
        if not shape:
            assert actual is value
    try:
        expected = _get_in_vals(LeftDangle, original, state.soq_assign)
    except (ValueError, TypeError, OverflowError) as error:
        with pytest.raises(type(error)):
            collect_typed_register_value(reg, assignments)
        return
    actual = collect_typed_register_value(reg, assignments)
    if isinstance(actual, np.ndarray) and actual.dtype == object:
        assert actual.shape == expected.shape
        for a, b in zip(actual.flat, expected.flat, strict=True):
            np.testing.assert_equal(a, b)
            assert type(a) is type(b)
    else:
        np.testing.assert_equal(actual, expected)
    assert type(actual) is type(expected)
    if isinstance(actual, np.ndarray):
        assert isinstance(expected, np.ndarray)
        assert actual.dtype == expected.dtype
    else:
        assert actual is value


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_word_binding_and_source_storage(kind: str, shape: tuple[int, ...]) -> None:
    dtype, source = pair(kind)
    for scalar in [0, 1, -1, 2, 4, 0.5, np.nan, np.int16(1), True]:
        value = np.full(shape, scalar) if shape else scalar
        compare(dtype, source, shape, value)


@pytest.mark.parametrize("p,degree,mode", CASES)
@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_fields_keep_quantum_field_and_classical_object_storage(
    p: int, degree: int, mode: str, classical: bool, shape: tuple[int, ...]
) -> None:
    field, source = field_pair(p, degree, mode, classical)
    dtype = field.register_type(classical=classical)
    for value in [
        field.gf_type.Ones(shape) if shape else field.gf_type(1),
        np.ones(shape, dtype=int) if shape else 1,
    ]:
        compare(dtype, source, shape, value)


@pytest.mark.parametrize("kind", KINDS)
def test_symbolic_scalar_bypasses_validator_but_sampled_value_does_not(
    kind: str,
) -> None:
    from sympy import Symbol

    dtype, source = pair(kind)
    expression = Symbol("value")
    compare(dtype, source, (), expression)
    calls: list[object] = []
    distribution = ClassicalDistribution(2)

    def resolve(value: ClassicalDistribution) -> object:
        calls.append(value)
        return expression

    register = Register("x", dtype)
    if kind == "opaque":
        assert bind_typed_register_values(
            register, {"x": distribution}, resolve_distribution=resolve
        ) == {(): expression}
    else:
        # Validators may reject symbolic comparisons through different source errors.
        try:
            source.assert_valid_classical_val(expression)
        except (ValueError, TypeError) as error:
            with pytest.raises(type(error)):
                bind_typed_register_values(
                    register, {"x": distribution}, resolve_distribution=resolve
                )
        else:
            assert bind_typed_register_values(
                register, {"x": distribution}, resolve_distribution=resolve
            ) == {(): expression}
    assert calls == [distribution]
    with pytest.raises(ValueError, match="non-deterministic"):
        bind_typed_register_values(register, {"x": distribution})


def test_metadata_refusal_and_field_override_replaces_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.classical_typed_values as owner

    dtype, _ = pair("uint")
    functions: list[Callable[[Register], object]] = [
        lambda r: bind_typed_register_values(r, {"x": 1}),
        lambda r: collect_typed_register_value(r, {(): 1}),
    ]
    for function in functions:
        with pytest.raises(ValueError, match="no native classical"):
            function(Register("x", replace(dtype, encoding="unknown")))
    field, _ = field_pair(2, 2, "default", False)
    register = Register("x", field.register_type())
    monkeypatch.setattr(
        owner, "resolve_finite_field_encoding", lambda dtype: pytest.fail("resolve")
    )
    value = field.gf_type(1)
    assert (
        bind_typed_register_values(register, {"x": value}, field_encoding=field)[()]
        is value
    )
    assert (
        collect_typed_register_value(register, {(): value}, field_encoding=field)
        is value
    )


def test_existing_physical_consumer_uses_replaced_state_binding() -> None:
    from tests.unit.test_classical_truth_table import (
        test_table_evaluator_uses_native_graph_and_complete_measured_branches,
    )

    test_table_evaluator_uses_native_graph_and_complete_measured_branches()
