from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from flagquantum._compiler.ir.types import QUBIT, IRType, tensor_type
from flagquantum._compiler.ir.values import ValueId, ValueRef

pytestmark = pytest.mark.unit


def test_qubit_is_a_linear_semantic_type() -> None:
    assert QUBIT == IRType("qubit", linear=True)
    assert QUBIT.linear is True
    with pytest.raises(FrozenInstanceError):
        QUBIT.name = "bit"  # type: ignore[misc]


def test_tensor_type_preserves_dtype_and_shape() -> None:
    value = tensor_type("torch.complex128", (2, 4))

    assert value.canonical() == ("tensor", ("complex128", (2, 4)), False)
    with pytest.raises(ValueError, match="cannot be negative"):
        tensor_type("float32", (-1,))

    with pytest.raises(TypeError, match="unsupported IR type parameter"):
        IRType("invalid", ([],))


def test_value_identity_is_deterministic_and_scoped() -> None:
    first = ValueRef(ValueId(7, "body"), QUBIT)
    same = ValueRef(ValueId("7", "body"), QUBIT)
    different_scope = ValueRef(ValueId(7, "nested"), QUBIT)

    assert first == same
    assert first.canonical() == ("body", 7, QUBIT.canonical())
    assert str(first.id) == "%body.7"
    assert first != different_scope


@pytest.mark.parametrize(
    ("index", "scope", "message"),
    [(-1, "entry", "negative"), (0, "", "scope")],
)
def test_value_identity_rejects_invalid_components(
    index: int, scope: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ValueId(index, scope)
