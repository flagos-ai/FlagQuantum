"""Unchanged-source declaration calls establish ownership before implementation."""

from collections.abc import Callable
from typing import Any

import pytest

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")


@pytest.mark.parametrize("factory", [SourceCalls])
@pytest.mark.parametrize("shape", [(), (2,), (0,)])
@pytest.mark.parametrize("directional", [False, True])
def test_declaration_call_retains_signature_and_linear_connections(
    factory: Callable[[Signature], Any], shape: tuple[int, ...], directional: bool
) -> None:
    dtype = RegisterType("uint", 3)
    registers = (
        (
            Register("input", dtype, shape, RegisterSide.LEFT),
            Register("output", dtype, shape, RegisterSide.RIGHT),
        )
        if directional
        else (Register("word", dtype, shape),)
    )
    signature = Signature(registers)
    builder = factory(signature)
    before = builder.available
    result = builder.call("declared", signature, builder.inputs)
    assert tuple(result) == tuple(register.name for register in signature.rights())
    assert builder.instances == (0,)
    assert not before & builder.available
    count = 0 if shape == (0,) else (2 if shape else 1)
    assert len(builder.available) == count
    assert len(builder.connections) == count
    builder.finalize(result)
    assert not builder.available
    assert len(builder.connections) == 2 * count
