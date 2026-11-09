"""Pinned builder workspace lifetime witnesses before native replacement."""

import numpy as np
import pytest

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_rebinding_contract import SourceRebindings
from tests.unit.test_register_bindings import SourceContract

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class SourceWorkspace(SourceRebindings):
    def allocate(self, n=1, dtype=None, dirty=False):
        converted = (
            None
            if dtype is None
            else SourceContract.register(Register("reg", dtype)).dtype
        )
        return self.builder.allocate(n, dtype=converted, dirty=dirty)

    def free(self, binding, dirty=False):
        return self.builder.free(binding, dirty=dirty)


@pytest.fixture(params=[SourceWorkspace], ids=["source-fake"])
def workspace(request):
    return request.param


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("dirty", [False, True])
def test_allocated_word_can_be_returned_and_input_can_be_freed(workspace, width, dirty):
    dtype = RegisterType("uint", width)
    signature = Signature(
        [
            Register("input", dtype, side=RegisterSide.LEFT),
            Register("output", dtype, side=RegisterSide.RIGHT),
        ]
    )
    builder = workspace(signature)
    word = builder.allocate(n=-123, dtype=dtype, dirty=dirty)
    assert builder.identity(word) == (0, "reg", ())
    builder.free(builder.inputs["input"], dirty=not dirty)
    builder.finalize({"output": word})
    assert not builder.available
    assert builder.instances == (0, 1)
    assert builder.connections == (
        (("entry", "input", ()), (1, "reg", ())),
        ((0, "reg", ()), ("return", "output", ())),
    )


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("dirty", [False, True])
def test_workspace_survives_split_join_with_fresh_successors(workspace, width, dirty):
    builder = workspace(Signature([]))
    word = builder.allocate(width, dirty=dirty)
    bits = builder.split(word)
    successor = builder.join(bits[::-1], RegisterType("opaque", width))
    builder.free(successor, dirty=dirty)
    builder.finalize({})
    assert not builder.available
    assert builder.instances == (0, 1, 2, 3)
    assert len(builder.connections) == width + 2


def test_stale_free_reserves_failed_instance_and_preserves_connections(workspace):
    builder = workspace(Signature([]))
    word = builder.allocate()
    builder.free(word)
    before = builder.connections
    with pytest.raises(builder.error_type, match="not an available"):
        builder.free(word)
    assert builder.instances == (0, 1, 2)
    assert builder.connections == before
    assert not builder.available


@pytest.mark.parametrize("binding", [None, [], np.empty((0,), dtype=object)])
def test_free_requires_one_scalar_handle_before_reserving_a_call(workspace, binding):
    builder = workspace(Signature([]))
    with pytest.raises(ValueError, match="single"):
        builder.free(binding)
    assert builder.instances == ()
