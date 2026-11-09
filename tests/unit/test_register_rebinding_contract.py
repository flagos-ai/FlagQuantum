"""Source witnesses for owned packed words crossing the existing gate boundary.

The replaceable fixture exposes source builder observations without storing a
second graph. Native register construction remains a separate implementation.
"""

import numpy as np
import pytest

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_component_calls import SourceCalls, port

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class SourceRebindings(SourceCalls):
    """Retain original mutation and normalize identities at the test boundary."""

    def split(self, value):
        return self.builder.split(value)

    def join(self, values, dtype):
        from tests.unit.test_register_bindings import SourceContract

        return self.builder.join(
            values, dtype=SourceContract.register(Register("reg", dtype)).dtype
        )


@pytest.fixture(params=[SourceRebindings], ids=["source-fake"])
def rebindings(request):
    return request.param


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_every_word_has_ownership_even_when_it_has_no_physical_bits(
    rebindings, width, shape
):
    register = port("word", width, shape)
    builder = rebindings(Signature([register]))
    initial = np.asarray(builder.inputs["word"], dtype=object)
    returned = np.empty(shape, dtype=object)
    for ordinal, index in enumerate(register.all_idxs()):
        word = initial[index]
        before = builder.identity(word)
        assert before in builder.available
        bits = builder.split(word)
        assert bits.shape == (width,)
        assert before not in builder.available
        assert builder.identities(bits) == tuple(
            (2 * ordinal, "reg", (i,)) for i in range(width)
        )
        assert set(builder.identities(bits)) <= builder.available
        joined = builder.join(bits, register.dtype)
        returned[index] = joined
        assert builder.identity(joined) == (2 * ordinal + 1, "reg", ())
        assert builder.identity(joined) in builder.available
        assert not set(builder.identities(bits)) & builder.available
    result = returned[()] if not shape else returned
    builder.finalize({"word": result})
    assert not builder.available
    assert builder.instances == tuple(range(2 * initial.size))
    assert len(builder.connections) == initial.size * (width + 2)
    final = builder.final_bindings()
    # No RightDangle is created for an empty-shaped output in the source graph.
    assert list(final) == (["word"] if initial.size else [])


@pytest.mark.parametrize("encoding", ["opaque", "uint"])
@pytest.mark.parametrize("permutation", [(0, 1, 2), (2, 0, 1)])
def test_split_join_reorders_connections_without_inserting_a_gate(
    rebindings, encoding, permutation
):
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    register = Register("word", RegisterType(encoding, 3))
    builder = rebindings(Signature([register]))
    bits = builder.split(builder.inputs["word"])
    result = builder.join(bits[list(permutation)], register.dtype)
    builder.finalize({"word": result})
    assert builder.instances == (0, 1)
    assert builder.connections[1:4] == tuple(
        ((0, "reg", (source,)), (1, "reg", (target,)))
        for target, source in enumerate(permutation)
    )
    assert get_cost_value(builder.program, QECGatesCost()).asdict() == {}


@pytest.mark.parametrize("failure", ["duplicate", "short", "rank"])
def test_failed_join_retains_exact_consumption_prefix(rebindings, failure):
    register = port(width=3)
    builder = rebindings(Signature([register]))
    bits = builder.split(builder.inputs["word"])
    before = builder.available
    connections = builder.connections
    if failure == "duplicate":
        values = bits[[0, 0, 2]]
        exception = builder.error_type
        consumed = 1
    elif failure == "short":
        values = bits[:2]
        exception = IndexError
        consumed = 2
    else:
        values = bits.reshape(3, 1)
        exception = ValueError
        consumed = 0
    with pytest.raises(exception):
        builder.join(values, register.dtype)
    assert builder.instances == ((0,) if failure == "rank" else (0, 1))
    assert builder.available == before - {
        builder.identity(bits[i]) for i in range(consumed)
    }
    assert builder.connections[: len(connections)] == connections
    assert len(builder.connections) == len(connections) + consumed
    if consumed:
        with pytest.raises(builder.error_type):
            builder.join(bits, register.dtype)
        assert builder.instances == (0, 1, 2)
        assert builder.available == before - {
            builder.identity(bits[i]) for i in range(consumed)
        }
