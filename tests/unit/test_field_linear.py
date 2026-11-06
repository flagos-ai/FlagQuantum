"""Literal-source CNOT networks, wire maps, amplitudes and costs for field scaling."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.field_linear import (
    gf2_constant_multiplication,
    gf2_monomial_multiplication,
)
from flagquantum.compiler.registers.field_multiplication import (
    FieldDecomposition,
    bind_field_outputs,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = pytest.mark.unit

FIELDS = ((1, 3), (2, 7), (3, 11), (4, 19), (5, 37), (8, 283))
CONSTANT_CASES = tuple(
    (width, modulus, value)
    for width, modulus in FIELDS
    for value in sorted(
        {1, (1 << width) - 1, 2 if width > 1 else 1, 0x53 if width == 8 else 1}
    )
)
SHIFT_CASES = tuple(
    (width, modulus, power)
    for width, modulus in FIELDS
    for power in (-width, -1, 0, 1, width + 1)
)


def _source_type(width: int, modulus: int) -> Any:
    pytest.importorskip("qualtran")
    from galois import Poly
    from qualtran import QGF

    return QGF(2, width) if width == 1 else QGF(2, width, Poly.Int(modulus))


def _assert_source(plan: FieldDecomposition, source: Any, register: str) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    target = tuple(range(len(plan.outputs[0])))
    gates, outputs = reference_gates(source, {register: target}, ())
    assert_gate_streams([(g.name, g.wires) for g in plan.instructions], gates)
    assert plan.outputs == (outputs[register],)
    assert plan.scratch == ()
    counts = Counter(g.name for g in plan.instructions)
    assert set(counts) <= {"cx"}
    model = get_cost_value(source, QECGatesCost())
    assert counts["cx"] == model.clifford
    assert model.toffoli == 0


@pytest.mark.parametrize("width,modulus,value", CONSTANT_CASES)
def test_constant_source_gate_order_mapping_and_resources(
    width: int, modulus: int, value: int
) -> None:
    dtype = _source_type(width, modulus)
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulK

    plan = gf2_constant_multiplication(tuple(range(width)), value, modulus=modulus)
    _assert_source(plan, GF2MulK(dtype, value), "g")


@pytest.mark.parametrize("width,modulus,power", SHIFT_CASES)
def test_monomial_source_gate_order_mapping_and_resources(
    width: int, modulus: int, power: int
) -> None:
    dtype = _source_type(width, modulus)
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2ShiftLeft,
        GF2ShiftRight,
    )

    source = GF2ShiftRight(dtype, power) if power >= 0 else GF2ShiftLeft(dtype, -power)
    plan = gf2_monomial_multiplication(tuple(range(width)), power, modulus=modulus)
    _assert_source(plan, source, "f")


@dataclass(frozen=True)
class LinearFieldOperation:
    plan: FieldDecomposition

    def append_to(self, circuit: Circuit) -> Circuit:
        target = tuple(range(circuit.n_qubits))
        return circuit.compose(
            CircuitIR(circuit.n_qubits, bind_field_outputs(self.plan, (target,)))
        )


@pytest.mark.integration
@pytest.mark.parametrize("width,modulus", FIELDS)
def test_all_complex_amplitudes_and_inverse_round_trips(
    width: int, modulus: int
) -> None:
    galois = pytest.importorskip("galois")
    field = (
        galois.GF(2**width)
        if width == 1
        else galois.GF(2**width, irreducible_poly=galois.Poly.Int(modulus))
    )
    basis = torch.arange(1 << width, dtype=torch.float64)
    state = torch.complex(basis + 1, 2 * basis - 3)
    state /= state.norm()
    target = tuple(range(width))
    constants = (
        field.elements[1:]
        if width <= 4
        else [field(1), field(2), field((1 << width) - 1)]
    )
    cases = [
        (gf2_constant_multiplication(target, int(c), modulus=modulus), c)
        for c in constants
    ]
    generator = field(2) if width > 1 else field(1)
    cases.extend(
        (gf2_monomial_multiplication(target, power, modulus=modulus), generator**power)
        for power in (-width, -1, 0, 1, width + 1)
    )
    for plan, scale in cases:
        expected = torch.zeros_like(state)
        for value in range(1 << width):
            expected[int(field(value) * scale)] = state[value]
        circuit = Circuit(width, dtype=torch.complex128, inputs=state).append(
            LinearFieldOperation(plan)
        )
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=1e-12
        )
        inverse = gf2_constant_multiplication(target, int(scale**-1), modulus=modulus)
        restored = circuit.append(LinearFieldOperation(inverse))
        torch.testing.assert_close(
            restored.state().flatten(), state, rtol=0, atol=1e-12
        )


def test_wire_maps_can_be_consumed_without_binding_swaps() -> None:
    target = tuple(range(8))
    forward = gf2_monomial_multiplication(target, 3, modulus=283)
    reverse = gf2_monomial_multiplication(forward.outputs[0], -3, modulus=283)
    assert reverse.outputs == (target,)
    assert all(g.name == "cx" for g in (*forward.instructions, *reverse.instructions))
    basis = torch.arange(256, dtype=torch.float64)
    state = torch.complex(basis + 1, 2 * basis - 3)
    state /= state.norm()
    result = Circuit(8, dtype=torch.complex128, inputs=state).compose(
        CircuitIR(8, (*forward.instructions, *reverse.instructions))
    )
    torch.testing.assert_close(result.state().flatten(), state, rtol=0, atol=1e-12)


@pytest.mark.parametrize("value", [0, -1, 8, True])
def test_nonunitary_or_out_of_range_constants_are_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        gf2_constant_multiplication((0, 1, 2), value, modulus=11)


def test_invalid_monomial_and_register_inputs() -> None:
    with pytest.raises(ValueError, match="concrete integer"):
        gf2_monomial_multiplication((0, 1), True, modulus=7)
    with pytest.raises(ValueError, match="nonzero polynomial root"):
        gf2_monomial_multiplication((0,), -1, modulus=2)
    with pytest.raises(ValueError, match="distinct"):
        gf2_constant_multiplication((0, 0), 1, modulus=7)
