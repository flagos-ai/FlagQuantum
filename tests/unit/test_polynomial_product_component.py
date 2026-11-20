"""Original recursive binary polynomial calls preserve arbitrary targets and padding."""

from collections import Counter

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.polynomial import (
    binary_polynomial_product_ir,
)
from flagquantum.compiler.registers.finite_field.polynomial_product_component import (
    binary_polynomial_program,
    one_plus_polynomial_program,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.field_product_checks import bind_product, product_source_stream
from tests.unit.register_source_graph import assert_source_wire_streams

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = tuple((n, k) for n in (1, 2, 3, 5, 8) for k in (None, *sorted({1, n, n + 1})))


@pytest.mark.parametrize("width,shift", CASES)
def test_complete_source_graph_and_existing_owner(
    width: int, shift: int | None
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        BinaryPolynomialMultiplication,
        MultiplyPolyByOnePlusXk,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    program = (
        binary_polynomial_program(width)
        if shift is None
        else one_plus_polynomial_program(width, shift)
    )
    bound, ports = bind_product(program)
    start = max(w for word in ports.values() for w in word) + 1
    expected = binary_polynomial_product_ir(
        ports["f"][::-1],
        ports["g"][::-1],
        ports["h"][::-1],
        tuple(range(start, start + 16)),
        shift=shift,
    )
    assert bound.circuit.instructions == expected
    assert tuple(word[0] for word in bound.outputs) == tuple(
        w for word in ports.values() for w in word
    )
    source = (
        BinaryPolynomialMultiplication(width)
        if shift is None
        else MultiplyPolyByOnePlusXk(width, shift)
    )
    gates, outputs = product_source_stream(source, bound, ports)
    assert_source_wire_streams(bound.circuit.instructions, gates)
    assert outputs == ports
    count = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert (count["ccx"], count["cx"]) == (model.toffoli, model.clifford)
    assert bound.classical_outputs == ()
    assert sum(len(e.wires) for e in bound.events if e.kind == "allocate") == sum(
        len(e.wires) for e in bound.events if e.kind == "free"
    )


@pytest.mark.parametrize(
    "width,shift", [(1, None), (1, 1), (1, 2), (2, None), (2, 1), (3, None)]
)
def test_every_arbitrary_target_complex_amplitude_and_padding_zero(
    width: int, shift: int | None
) -> None:
    program = (
        binary_polynomial_program(width)
        if shift is None
        else one_plus_polynomial_program(width, shift)
    )
    bound, ports = bind_product(program)
    data = sum(map(len, ports.values()))
    # Size from actual lifetimes, never the unused binder reserve.
    active = max([data, *(w + 1 for e in bound.events for w in e.wires)])
    count = active + 1
    logical = torch.arange(1, (1 << (data + 1)) + 1, dtype=torch.float64)
    logical = torch.complex(logical, logical.flip(0))
    logical /= logical.norm()
    state = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for value in range(1 << data):
        bits = [(value >> (data - 1 - w)) & 1 for w in range(data)]
        f = sum(bits[w] << i for i, w in enumerate(ports["f"]))
        g = sum(bits[w] << i for i, w in enumerate(ports["g"]))
        product = 0
        for i in range(width):
            if f >> i & 1:
                product ^= g << i
        if shift is not None:
            product ^= product << shift
        for i, w in enumerate(ports["h"]):
            bits[w] ^= (product >> i) & 1
        output = sum(bit << (data - 1 - i) for i, bit in enumerate(bits))
        for spectator in (0, 1):
            src = value << (count - data) | spectator
            dst = output << (count - data) | spectator
            state[src] = logical[value * 2 + spectator]
            expected[dst] = logical[value * 2 + spectator]
    result = (
        Circuit(count, inputs=state, dtype=torch.complex128)
        .compose(CircuitIR(count, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=0)
