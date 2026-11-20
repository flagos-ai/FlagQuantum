"""Shared stored shift/constant chains retain source gates, aliases and inverses."""

from collections import Counter

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.constant_multiplication_component import (
    constant_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.field_linear import (
    gf2_constant_multiplication,
    gf2_monomial_multiplication,
)
from flagquantum.compiler.registers.finite_field.monomial_component import (
    monomial_multiplication_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from tests.unit.field_product_checks import product_source_stream
from tests.unit.register_source_graph import assert_source_wire_streams
from tests.unit.test_field_frobenius_component import dtype
from tests.unit.test_field_linear import CONSTANT_CASES, SHIFT_CASES, _source_type

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,modulus,power", SHIFT_CASES)
def test_shift_stored_owner_and_original_graph(
    width: int, modulus: int, power: int
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2ShiftLeft,
        GF2ShiftRight,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    wires = tuple(range(width))
    bound = bind_register_program(
        monomial_multiplication_program(dtype(width, modulus), power, modulus=modulus),
        (wires,),
    )
    plan = gf2_monomial_multiplication(wires, power, modulus=modulus)
    assert bound.circuit.instructions == plan.instructions
    assert bound.outputs == plan.outputs and bound.events == ()
    source_type = _source_type(width, modulus)
    source = (
        GF2ShiftRight(source_type, power)
        if power >= 0
        else GF2ShiftLeft(source_type, -power)
    )
    gates, outputs = product_source_stream(source, bound, {"f": wires})
    assert_source_wire_streams(bound.circuit.instructions, gates)
    assert bound.outputs == (outputs["f"],)
    assert len(gates) == get_cost_value(source, QECGatesCost()).clifford


@pytest.mark.parametrize("width,modulus,value", CONSTANT_CASES)
def test_constant_stored_owner_original_graph_and_model(
    width: int, modulus: int, value: int
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    wires = tuple(range(width))
    bound = bind_register_program(
        constant_multiplication_program(dtype(width, modulus), value, modulus=modulus),
        (wires,),
    )
    plan = gf2_constant_multiplication(wires, value, modulus=modulus)
    assert bound.circuit.instructions == plan.instructions
    assert bound.outputs == plan.outputs and bound.events == ()
    source = GF2MulK(_source_type(width, modulus), value)
    gates, outputs = product_source_stream(source, bound, {"g": wires})
    assert_source_wire_streams(bound.circuit.instructions, gates)
    assert bound.outputs == (outputs["g"],)
    assert (
        Counter(g.name for g in gates)["cx"]
        == get_cost_value(source, QECGatesCost()).clifford
    )


@pytest.mark.parametrize("width,modulus", [(1, 3), (2, 7), (3, 11), (4, 19)])
def test_all_scales_full_complex_spectator_clean_and_exact_inverse(
    width: int, modulus: int
) -> None:
    wires = tuple(range(width))
    field = _source_type(width, modulus).gf_type
    values = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    state = torch.zeros(1 << (width + 2), dtype=torch.complex128)
    state[::2] = torch.complex(values, values.flip(0))
    state /= state.norm()
    for constant in range(1, 1 << width):
        forward = bind_register_program(
            constant_multiplication_program(
                dtype(width, modulus), constant, modulus=modulus
            ),
            (wires,),
        )
        expected = torch.zeros_like(state)
        for value in range(1 << width):
            output = int(field(value) * field(constant))
            index = sum(
                ((output >> (width - 1 - i)) & 1) << (width + 1 - w)
                for i, w in enumerate(forward.outputs[0])
            )
            for spectator in (0, 1):
                expected[index | spectator << 1] = state[value << 2 | spectator << 1]
        circuit = Circuit(width + 2, inputs=state, dtype=torch.complex128).compose(
            CircuitIR(width + 2, forward.circuit.instructions)
        )
        torch.testing.assert_close(
            circuit.state().flatten(), expected, atol=1e-12, rtol=0
        )
        inverse = bind_register_program(
            constant_multiplication_program(
                dtype(width, modulus), constant, modulus=modulus, adjoint=True
            ),
            forward.outputs,
        )
        assert inverse.outputs == (wires,)
        restored = (
            circuit.compose(CircuitIR(width + 2, inverse.circuit.instructions))
            .state()
            .flatten()
        )
        torch.testing.assert_close(restored, state, atol=1e-12, rtol=0)
        assert torch.count_nonzero(restored[1::2]) == 0


@pytest.mark.parametrize("constant", [0, -1, 8, True])
def test_nonunitary_or_nonconcrete_constant_rejected(constant: int) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        constant_multiplication_program(dtype(3, 11), constant, modulus=11)


def test_monomial_retains_nonzero_polynomial_root_rejection() -> None:
    with pytest.raises(ValueError, match="nonzero polynomial root"):
        monomial_multiplication_program(dtype(1, 2), -1, modulus=2)
    with pytest.raises(ValueError, match="concrete integer"):
        monomial_multiplication_program(dtype(2, 7), True, modulus=7)
    with pytest.raises(ValueError, match="match"):
        constant_multiplication_program(dtype(3, 13), 1, modulus=11)
