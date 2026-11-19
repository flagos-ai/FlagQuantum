"""Caller-owned integer descriptors survive the shared Add component graph."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def integer_types(
    kind: str, left: int, right: int
) -> tuple[RegisterType, RegisterType, Any, Any]:
    from qualtran import QInt, QMontgomeryUInt, QUInt

    encoding = {
        "unsigned": "uint",
        "signed": "signed",
        "montgomery": "montgomery",
        "mixed": "uint",
    }[kind]
    source_type = {
        "unsigned": QUInt,
        "signed": QInt,
        "montgomery": QMontgomeryUInt,
        "mixed": QUInt,
    }[kind]
    if kind == "montgomery":
        return (
            RegisterType(encoding, left, parameters=(("modulus", (1 << left) - 1),)),
            RegisterType(encoding, right, parameters=(("modulus", (1 << right) - 1),)),
            source_type(left, (1 << left) - 1),
            source_type(right, (1 << right) - 1),
        )
    return (
        RegisterType(encoding, left),
        RegisterType("signed" if kind == "mixed" else encoding, right),
        source_type(left),
        QInt(right) if kind == "mixed" else source_type(right),
    )


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 2), (2, 3), (3, 5), (8, 8)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_typed_add_preserves_descriptors_and_original_gate_feedback(
    kind: str, left: int, right: int, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic import Add
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    a_dtype, b_dtype, source_a, source_b = integer_types(kind, left, right)
    graph = addition_component_program(
        left, right, left_dtype=a_dtype, right_dtype=b_dtype, adjoint=adjoint
    )
    if kind == "unsigned":
        assert graph == addition_component_program(left, right, adjoint=adjoint)
    root = next(function for function in graph.functions if function.name == "main")
    assert root.registers is not None
    assert tuple(register.dtype for register in root.registers) == (a_dtype, b_dtype)
    a = tuple(reversed(range(left)))
    b = tuple(reversed(range(left, left + right)))
    work = tuple(range(left + right, left + 2 * right - 1))
    bound = bind_register_program(graph, (a, b), clean=work)
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    source = Add(source_a, source_b)
    if adjoint:
        source = source.adjoint()
    expected = reference_arithmetic_gates(source, {"a": a, "b": b}, work, slots)
    assert_source(bound.circuit.instructions, expected)
    assert bound.outputs == (a, b)
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (right - 1)
    assert counts["measure"] == counts["reset"] == model.measurement == right - 1


@pytest.mark.parametrize("kind", ("signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_typed_add_full_source_tensor_and_each_raw_measurement_history(
    kind: str, left: int, right: int, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic import Add

    a_dtype, b_dtype, source_a, source_b = integer_types(kind, left, right)
    graph = addition_component_program(
        left, right, left_dtype=a_dtype, right_dtype=b_dtype, adjoint=adjoint
    )
    logical = left + right
    work = right - 1
    data = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    state = torch.complex(data, 3 - 2 * data)
    state /= state.norm()
    indexes = ((torch.arange(len(state)) >> 1) << (work + 1)) | (
        torch.arange(len(state)) & 1
    )
    initial = torch.zeros(1 << (logical + work + 1), dtype=torch.complex128)
    initial[indexes] = state
    source = Add(source_a, source_b)
    if adjoint:
        source = source.adjoint()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    expected[indexes] = (
        matrix.reshape(1 << logical, 1 << logical) @ state.reshape(-1, 2)
    ).flatten()
    bound = bind_register_program(
        graph,
        (tuple(range(left)), tuple(range(left, logical))),
        clean=tuple(range(logical, logical + work)),
    )
    for history in product((0, 1), repeat=work):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected / (2 ** (work / 2)), rtol=0, atol=2e-12
        )
        assert torch.count_nonzero(actual.reshape(-1, 2 ** (work + 1))[:, 2:]) == 0


@pytest.mark.parametrize(
    "dtype",
    (
        RegisterType("uint", 2),
        RegisterType("opaque", 3),
        RegisterType("ones_complement", 3),
    ),
)
def test_typed_add_rejects_wrong_width_or_non_source_integer(
    dtype: RegisterType,
) -> None:
    with pytest.raises(ValueError, match="integer descriptors"):
        addition_component_program(3, left_dtype=dtype)


def test_typed_add_direction_requires_explicit_boolean() -> None:
    with pytest.raises(ValueError, match="direction requires a boolean"):
        addition_component_program(2, adjoint=1)  # type: ignore[arg-type]
