"""Typed multiplex traversal retains original calls and measured exchanges."""

from collections import Counter
from itertools import product
from math import prod

import pytest
import torch

from flagquantum.compiler.registers.lookup.multiplexed_swap import MultiplexedSwapLayout
from flagquantum.compiler.registers.lookup.multiplexed_swap_component import (
    multiplexed_swap_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_multiplexed_swap import source_swap
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_unary_component_calls import source_slot_names

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "shape,widths,controls,word",
    [
        ((5,), (3,), 0, 2),
        ((3,), (2,), 0, 1),
        ((2,), (3,), 0, 1),
        ((3,), (2,), 1, 1),
        ((3,), (2,), 2, 1),
        ((2,), (1,), 3, 1),
        ((2, 3), (1, 2), 1, 1),
        ((2, 1), (1, 1), 0, 1),
        ((1,), (1,), 1, 1),
    ],
)
def test_ordered_source_calls_complex_branches_work_and_resources(
    shape: tuple[int, ...], widths: tuple[int, ...], controls: int, word: int
) -> None:
    layout = MultiplexedSwapLayout(
        tuple(
            Register(f"s{i}", bounded_unsigned_register_type(width, extent))
            for i, (width, extent) in enumerate(zip(widths, shape, strict=True))
        ),
        word,
        tuple(Register(f"c{i}", quantum_bit_type()) for i in range(controls)),
    )
    program = multiplexed_swap_program(layout)
    source = source_swap(shape, widths, controls, word)
    signature = Signature(program.functions[0].registers or ())
    assert _native_trace(program) == _source_trace(
        _source(signature), source.decompose_bloq()
    )
    bindings = []
    offset = 0
    for register in signature:
        for _ in register.all_idxs():
            width = int(register.bitsize)
            bindings.append(tuple(range(offset, offset + width)))
            offset += width
    size, _ = layout.requirements()
    work = tuple(range(offset, offset + size))
    bound = bind_register_program(program, tuple(bindings), clean=work)
    assert bound.outputs == tuple(bindings)
    gates = source_slot_names(bound.circuit.instructions, work)
    previous = layout.bind(tuple(range(offset)), work, tuple(range(size)))
    assert_source(gates, previous)
    assert Counter(g.name for g in gates) == Counter(g.name for g in previous)
    model = layout.resource_model().gates
    assert sum(g.name == "measure" for g in gates) == model.measurement
    assert sum(g.name in {"t", "tdg"} for g in gates) == (
        7 * prod(shape) * word + 4 * model.and_bloq
    )
    if offset > 11:
        return
    indices = torch.arange(1 << offset, dtype=torch.float64)
    ket = torch.complex(indices + 1, indices.flip(0) - 3)
    ket /= ket.norm()
    initial = torch.zeros(1 << (offset + size), dtype=ket.dtype)
    initial[:: 1 << size] = ket
    for history in product((0, 1), repeat=int(model.measurement)):
        actual = forced_branch(initial, gates, history)
        expected = forced_branch(initial, previous, history)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
        assert torch.linalg.vector_norm(actual.reshape(-1, 1 << size)[:, 1:]) < 2e-12


def test_existing_singleton_refusal_is_preserved() -> None:
    layout = MultiplexedSwapLayout(
        (Register("selection", bounded_unsigned_register_type(1, 1)),), 1
    )
    with pytest.raises(ValueError, match="singleton first axis"):
        multiplexed_swap_program(layout)
