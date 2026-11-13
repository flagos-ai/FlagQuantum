"""Literal source plus-equal gates, arbitrary targets and complex append states."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    bind_field_outputs,
    gf2_multiplication,
)
from flagquantum.compiler.registers.finite_field.multiplication_model import (
    CX,
    TOFFOLI,
    field_multiplication_gate_model,
    polynomial_multiplication_graph,
)
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,modulus", [(1, 3), (2, 7), (3, 11), (4, 19), (5, 37), (6, 67), (8, 283)]
)
def test_source_plus_equal_exact_gate_operands_and_free_outputs(
    width: int, modulus: int
) -> None:
    from galois import Poly
    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    x, y, z = [tuple(range(i * width, (i + 1) * width)) for i in range(3)]
    plan = gf2_multiplication(
        x, y, z, (), modulus=modulus, algorithm="quadratic", plus_equal_prod=True
    )
    original = GF2Multiplication(QGF(2, width, Poly.Int(modulus)), plus_equal_prod=True)
    gates, outputs = reference_gates(original, {"x": x, "y": y, "result": z}, ())
    assert_gate_streams([(g.name, g.wires) for g in plan.instructions], gates)
    assert plan.outputs == (outputs["x"], outputs["y"], outputs["result"])
    assert plan.scratch == ()
    counts = Counter(g.name for g in plan.instructions)
    model = field_multiplication_gate_model(
        width, modulus=modulus, plus_equal_prod=True
    )
    source = get_cost_value(original, QECGatesCost())
    assert counts["ccx"] == model[TOFFOLI] == source.toffoli
    assert counts["cx"] == model[CX] == source.clifford


@pytest.mark.parametrize("width,modulus", [(2, 7), (3, 11)])
def test_all_nonzero_targets_complex_spectator_append_and_inverse(
    width: int, modulus: int
) -> None:
    import galois

    n = 3 * width + 2
    x, y, z = [tuple(range(1 + i, 3 * width + 1, 3)) for i in range(3)]
    state = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    field = galois.GF(1 << width, irreducible_poly=galois.Poly.Int(modulus))

    def index(a: int, b: int, c: int, s: int) -> int:
        result = s << (n - 1)
        for value, wires in [(a, x), (b, y), (c, z)]:
            for i, wire in enumerate(wires):
                result |= ((value >> (width - 1 - i)) & 1) << (n - 1 - wire)
        return result

    for a in range(1 << width):
        for b in range(1 << width):
            for c in range(1 << width):
                for s in range(2):
                    amplitude = complex(a + 2 * c + s + 1, b - c - 3 * s)
                    state[index(a, b, c, s)] = torch.tensor(
                        amplitude, dtype=torch.complex128
                    )
                    expected[index(a, b, c ^ int(field(a) * field(b)), s)] = (
                        torch.tensor(amplitude, dtype=torch.complex128)
                    )
    expected /= state.norm()
    state /= state.norm()
    plan = gf2_multiplication(
        x, y, z, (), modulus=modulus, algorithm="quadratic", plus_equal_prod=True
    )
    operation = FieldOperation(n, bind_field_outputs(plan, (x, y, z)))
    circuit = Circuit(n, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    torch.testing.assert_close(
        circuit.append(operation).state().flatten(), state, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "width,modulus", [(2, 7), (3, 11), (4, 19), (5, 37), (6, 67), (8, 283)]
)
@pytest.mark.parametrize("algorithm", ["quadratic", "karatsuba"])
def test_field_models_independent_of_emission(
    width: int, modulus: int, algorithm: str
) -> None:
    from typing import Literal, cast

    from galois import Poly
    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2Multiplication,
        GF2MulViaKaratsuba,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    selected = cast(Literal["quadratic", "karatsuba"], algorithm)
    factory = GF2Multiplication if selected == "quadratic" else GF2MulViaKaratsuba
    source = get_cost_value(factory(QGF(2, width, Poly.Int(modulus))), QECGatesCost())
    model = field_multiplication_gate_model(width, modulus=modulus, algorithm=selected)
    assert (model[TOFFOLI], model[CX]) == (source.toffoli, source.clifford)


@pytest.mark.parametrize("log_width", range(11))
def test_original_polynomial_power_cost_assertions(log_width: int) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        BinaryPolynomialMultiplication,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    width = 2**log_width
    leaves = polynomial_multiplication_graph(width).leaf_counts
    cost = get_cost_value(BinaryPolynomialMultiplication(width), QECGatesCost())
    assert leaves[TOFFOLI] == cost.toffoli == 3**log_width
    assert leaves.get(CX, 0) == cost.clifford
    assert leaves.get(CX, 0) < (10 + 1 / 3) * 3**log_width


def test_accumulation_and_model_admission() -> None:
    from sympy import Symbol, log

    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
    )

    args = ((0, 1), (2, 3), (4, 5), ())
    with pytest.raises(ValueError, match="quadratic"):
        gf2_multiplication(*args, modulus=7, plus_equal_prod=True)
    invalid: Any = 1
    with pytest.raises(ValueError, match="boolean"):
        gf2_multiplication(*args, modulus=7, plus_equal_prod=invalid)
    m = Symbol("m")
    assert field_multiplication_gate_model(m) == {TOFFOLI: m**2, CX: 3 * m}
    assert field_multiplication_gate_model(m, plus_equal_prod=True) == {
        TOFFOLI: m**2,
        CX: 6 * m,
    }
    assert field_multiplication_gate_model(m, algorithm="karatsuba") == {
        TOFFOLI: m ** (log(3) / log(2)),
        CX: m**2,
    }
    assert field_multiplication_gate_model(m, algorithm="erasure") == {
        CliffordCall("cz", 2): m**2
    }


def test_resource_model_admission_and_degree_one() -> None:
    for width in [0, -1]:
        with pytest.raises(ValueError, match="positive"):
            polynomial_multiplication_graph(width)
    for shift in [0, -1, 4]:
        with pytest.raises(ValueError, match="shift"):
            polynomial_multiplication_graph(2, shift)
    with pytest.raises(ValueError, match="modulus"):
        field_multiplication_gate_model(2)
    invalid: Any = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        field_multiplication_gate_model(2, modulus=7, algorithm=invalid)
    with pytest.raises(ValueError, match="clean"):
        field_multiplication_gate_model(
            2, modulus=7, algorithm="karatsuba", plus_equal_prod=True
        )
    assert field_multiplication_gate_model(1, modulus=3, algorithm="karatsuba") == {
        TOFFOLI: 1
    }
