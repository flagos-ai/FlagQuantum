"""Pinned integer storage rebindings preserve dtype and single consumption."""

from collections.abc import Callable
from typing import Any

import pytest

from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    montgomery_register_type,
    ones_complement_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_rebinding_contract import SourceRebindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

INTEGER_WORDS = (
    ones_complement_register_type(2),
    ones_complement_register_type(3),
    bounded_unsigned_register_type(2, 3),
    bounded_unsigned_register_type(3, 5),
    montgomery_register_type(2),
    montgomery_register_type(3, 5),
)


@pytest.mark.parametrize("builder_type", [SourceRebindings])
@pytest.mark.parametrize("dtype", INTEGER_WORDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_integer_storage_connections_and_consumption(
    builder_type: Callable[[Signature], Any], dtype: RegisterType, reverse: bool
) -> None:
    register = Register("word", dtype)
    builder = builder_type(Signature((register,)))
    original = builder.inputs["word"]
    width = int(dtype.num_qubits)
    bits = builder.split(original)
    assert builder.identity(original) not in builder.available
    order = tuple(reversed(range(width))) if reverse else tuple(range(width))
    result = builder.join(bits[list(order)], dtype)
    assert not set(builder.identities(bits)) & builder.available
    builder.finalize({"word": result})
    assert builder.instances == (0, 1)
    assert not builder.available
    assert builder.connections[1 : width + 1] == tuple(
        ((0, "reg", (source,)), (1, "reg", (target,)))
        for target, source in enumerate(order)
    )
    assert len(builder.connections) == width + 2
