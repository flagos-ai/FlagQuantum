"""Check the interior allocation fixture with an independent source workspace."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.primitives.swap import source_swap_resource_formula
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.interior_workspace_fixture import (
    interior_call_kinds,
    interior_workspace_fixture,
    interior_workspace_signature,
)
from tests.unit.register_source_workspace import source_workspace_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 3, 10])
def test_interior_workspace_gate_order_calls_and_ownership(width: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = InteriorAlloc(width)
    native = interior_workspace_fixture(width)
    x = tuple(range(width - 1, -1, -1))
    y = tuple(range(width, 2 * width))
    work = tuple(range(2 * width, 3 * width))
    bound = bind_register_program(native, (x, y), clean=work)
    expected, events = source_workspace_gates(source, {"x": x, "y": y}, work, ())
    assert_source(bound.circuit.instructions, expected)
    assert [(e.kind, e.wires, e.dirty) for e in bound.events] == list(events)
    assert bound.outputs == (x, y)
    assert interior_call_kinds(native) == tuple(
        type(node.bloq).__name__
        for node, _, _ in source.decompose_bloq().iter_bloqsoqs()
    )
    _, counts = source.call_graph(max_depth=1)
    assert Counter(interior_call_kinds(native)) == {
        type(call).__name__: count for call, count in counts.items()
    }
    assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == 3 * width
    assert get_cost_value(source, QECGatesCost()).clifford == 2 * width
    assert Counter(g.name for g in bound.circuit.instructions) == {"swap": 2 * width}


@pytest.mark.parametrize("width", [1, 2, 3])
def test_clean_input_subspace_matches_source_tensor_and_returns_work(
    width: int,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc

    source = InteriorAlloc(width)
    words = 1 << width
    active = torch.arange(1, words + 1, dtype=torch.float64)
    amplitudes = torch.complex(active, 1 - active)
    amplitudes /= amplitudes.norm()
    logical = torch.zeros(words**2, dtype=amplitudes.dtype)
    logical[::words] = amplitudes
    vacuum = torch.zeros(words, dtype=amplitudes.dtype)
    vacuum[0] = 1
    expected = (
        torch.as_tensor(source.tensor_contract().copy(), dtype=logical.dtype) @ logical
    )
    x, y = tuple(range(width)), tuple(range(width, 2 * width))
    work = tuple(range(2 * width, 3 * width))
    body = bind_register_program(interior_workspace_fixture(width), (x, y), clean=work)
    state = Circuit(3 * width, inputs=torch.kron(logical, vacuum), dtype=logical.dtype)
    state.append(FieldOperation(3 * width, body.circuit.instructions))
    torch.testing.assert_close(
        state.state().flatten(), torch.kron(expected, vacuum), atol=2e-14, rtol=0
    )
    torch.testing.assert_close(expected[:words], amplitudes, atol=2e-14, rtol=0)
    assert torch.count_nonzero(expected[words:]) == 0
    with pytest.raises(ValueError, match="Free|free"):
        source.call_classically(x=0, y=1)

    outside = torch.zeros_like(torch.kron(logical, vacuum))
    outside[words] = 1
    physical = Circuit(3 * width, inputs=outside, dtype=logical.dtype)
    physical.append(FieldOperation(3 * width, body.circuit.instructions))
    residual = torch.zeros_like(outside)
    residual[1] = 1
    torch.testing.assert_close(physical.state().flatten(), residual, atol=0, rtol=0)


def test_symbolic_interior_signature() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    width = Symbol("n", positive=True, integer=True)
    source = InteriorAlloc(width)
    signature = interior_workspace_signature(width)
    assert [(r.name, r.bitsize) for r in signature] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    assert signature.n_qubits() + width == get_cost_value(source, QubitCount())
    assert 2 * source_swap_resource_formula(width).clifford == (
        get_cost_value(source, QECGatesCost()).clifford
    )
