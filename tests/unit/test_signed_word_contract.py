"""Source-backed contract scenarios for executable signed word rebindings."""

from collections.abc import Callable
from typing import Any

import pytest

from flagquantum.core.ir.program.register_types import signed_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_rebinding_contract import SourceRebindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")


@pytest.mark.parametrize("builder_type", [SourceRebindings])
@pytest.mark.parametrize("width", [1, 3, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_signed_word_connections_and_single_consumption(
    builder_type: Callable[[Signature], Any], width: int, reverse: bool
) -> None:
    signature = Signature((Register("word", signed_register_type(width)),))
    builder = builder_type(signature)
    word = builder.inputs["word"]
    bits = builder.split(word)
    assert builder.identity(word) not in builder.available
    order = tuple(reversed(range(width))) if reverse else tuple(range(width))
    result = builder.join(bits[list(order)], signature[0].dtype)
    assert not set(builder.identities(bits)) & builder.available
    builder.finalize({"word": result})
    assert builder.instances == (0, 1)
    assert not builder.available
    assert builder.connections[1 : width + 1] == tuple(
        ((0, "reg", (source,)), (1, "reg", (target,)))
        for target, source in enumerate(order)
    )
    assert len(builder.connections) == width + 2
