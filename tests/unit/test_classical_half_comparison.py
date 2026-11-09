"""Source half-comparison difference, target and logical erasure contracts."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_half_comparison import (
    half_comparison_classical_values,
)
from tests.unit.test_classical_comparison import PAIRS, compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = [
    0,
    1,
    3,
    -1,
    np.int64(3),
    sympy.Integer(3),
    sympy.Symbol("n", integer=True),
    3.0,
]
RELATIONS = {">": "gt", ">=": "ge", "<": "lt", "<=": "le"}


def source(width, kind, relation, uncompute=False):
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        LinearDepthHalfGreaterThan,
        LinearDepthHalfGreaterThanEqual,
        LinearDepthHalfLessThan,
        LinearDepthHalfLessThanEqual,
    )

    dtype = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[kind](
        width
    )
    cls = {
        ">": LinearDepthHalfGreaterThan,
        ">=": LinearDepthHalfGreaterThanEqual,
        "<": LinearDepthHalfLessThan,
        "<=": LinearDepthHalfLessThanEqual,
    }[relation]
    return cls(dtype, uncompute=uncompute)


def packed(values):
    return (tuple(values), *values.values())


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("kind", ["unsigned", "signed", "montgomery"])
@pytest.mark.parametrize("relation", RELATIONS)
def test_original_compute_cast_order_modulus_types_and_errors(width, kind, relation):
    original = source(width, kind, relation)
    for left, right in PAIRS:
        compare_calls(
            lambda a, b: packed(original.on_classical_vals(a, b)),
            lambda a, b: packed(
                half_comparison_classical_values(a, b, width, relation=relation)
            ),
            (left, right),
        )


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("kind", ["unsigned", "signed", "montgomery"])
@pytest.mark.parametrize("relation", RELATIONS)
def test_original_erase_checks_correct_wrong_and_missing_outputs(width, kind, relation):
    original = source(width, kind, relation)
    erase = original.adjoint()
    values = original.on_classical_vals(2, 1)
    outputs = [
        (values["c"], values["target"]),
        (values["c"] + 1, values["target"]),
        (values["c"], 1 - values["target"]),
        (None, values["target"]),
        (values["c"], None),
    ]
    for difference, target in outputs:
        compare_calls(
            lambda a, b, c, t: packed(erase.on_classical_vals(a, b, c, t)),
            lambda a, b, c, t: packed(
                half_comparison_classical_values(
                    a,
                    b,
                    width,
                    relation=relation,
                    uncompute=True,
                    difference=c,
                    target=t,
                )
            ),
            (2, 1, difference, target),
        )


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize(
    "difference,target", [(0, None), (None, 0), (0, 0), (False, False)]
)
def test_forward_outputs_must_be_absent_after_difference_arithmetic(
    relation, difference, target
):
    original = source(3, "unsigned", relation)
    compare_calls(
        lambda a, b, c, t: packed(original.on_classical_vals(a, b, c, t)),
        lambda a, b, c, t: packed(
            half_comparison_classical_values(
                a, b, 3, relation=relation, difference=c, target=t
            )
        ),
        (2, 1, difference, target),
    )


def test_shared_add_replacement_retains_integer_operand_casts_and_symbolic_width(
    monkeypatch,
):
    import flagquantum.simulation.classical_half_comparison as native

    marker, calls = np.int64(7), []

    def add(left, right, *, num_bits, signed):
        calls.append((left, right, num_bits, signed))
        assert type(left) is type(right) is int
        assert type(num_bits) is sympy.Integer
        return marker

    monkeypatch.setattr(native, "add_integer_values", add)
    left, right = np.int8(3), np.int8(1)
    result = native.half_comparison_classical_values(
        left, right, sympy.Integer(3), relation="<"
    )
    assert result["a"] is left and result["b"] is right and result["c"] is marker
    assert result["target"] == 0
    restored = native.half_comparison_classical_values(
        left,
        right,
        sympy.Integer(3),
        relation="<",
        uncompute=True,
        difference=marker,
        target=0,
    )
    assert restored["a"] is left and restored["b"] is right
    assert calls == [(3, -1, sympy.Integer(4), False)] * 2


def test_invalid_private_source_relation_keeps_original_else_branch():
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import _LinearDepthHalfComparisonBase

    original = _LinearDepthHalfComparisonBase(QUInt(3), op_symbol="unexpected")
    compare_calls(
        lambda a, b: packed(original.on_classical_vals(a, b)),
        lambda a, b: packed(
            half_comparison_classical_values(a, b, 3, relation="unexpected")
        ),
        (2, 1),
    )


@pytest.mark.parametrize("relation", [">=", "<"])
def test_original_classical_erase_and_physical_port_aliases_remain_distinct(relation):
    from flagquantum.compiler.registers.comparison.half_ordering import (
        half_compare_outputs,
    )

    original = source(2, "unsigned", relation)
    values = original.on_classical_vals(1, 0)
    native = half_comparison_classical_values(1, 0, 2, relation=relation)
    assert native == values
    assert half_comparison_classical_values(
        1,
        0,
        2,
        relation=relation,
        uncompute=True,
        difference=values["c"],
        target=values["target"],
    ) == {"a": 1, "b": 0}
    assert original.adjoint().on_classical_vals(**values) == {"a": 1, "b": 0}
    assert original.adjoint().decompose_bloq().call_classically(**values) == (0, 1)
    assert half_compare_outputs(
        (0, 1), (2, 3), (5, 6, 7), 4, relation=RELATIONS[relation], adjoint=True
    ) == {"a": (2, 3), "b": (0, 1)}


@pytest.mark.parametrize("relation", RELATIONS)
def test_value_outputs_drive_full_complex_half_compute_and_all_erase_histories(
    relation,
):
    import torch
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq
    from flagquantum.compiler.registers.comparison.half_ordering import (
        half_compare_ir,
        half_compare_outputs,
    )
    from flagquantum.core.ir import CircuitIR
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_half_comparison import assert_half_source, layout

    a, b, target, difference, signs, bits = layout(2)
    compute = half_compare_ir(
        a, b, difference, target[0], signs, relation=RELATIONS[relation]
    )
    erase = half_compare_ir(
        a,
        b,
        difference,
        target[0],
        signs,
        relation=RELATIONS[relation],
        adjoint=True,
        classical_bits=bits,
    )
    original = source(2, "unsigned", relation)
    assert_half_source(
        2,
        False,
        original,
        compute,
        erase,
        flipped=relation in (">=", "<="),
        erased_outputs=half_compare_outputs(
            a, b, difference, target[0], relation=RELATIONS[relation], adjoint=True
        ),
    )
    grid = torch.arange(1, 33, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(2048, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(16), (0, 1)):
        left, right = data >> 2, data & 3
        values = half_comparison_classical_values(left, right, 2, relation=relation)
        initial[(data << 7) | spectator] = weights[2 * data + spectator]
        expected[
            (data << 7) | (values["target"] << 6) | (values["c"] << 3) | spectator
        ] = weights[2 * data + spectator]
    forward = (
        fq.Circuit(11, inputs=initial, dtype=torch.complex128)
        .compose(CircuitIR(11, compute))
        .state()
        .flatten()
    )
    torch.testing.assert_close(forward, expected, rtol=0, atol=2e-12)
    assert torch.count_nonzero(forward.reshape(16, 2, 8, 4, 2)[:, :, :, 1:, :]) == 0
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(forward, erase, history), initial / 2, rtol=0, atol=2e-12
        )
    model_compute = get_cost_value(original, QECGatesCost())
    model_erase = get_cost_value(original.adjoint(), QECGatesCost())
    assert (
        sum(g.name in ("t", "tdg") for g in (*compute, *erase))
        == model_compute.total_t_count() + model_erase.total_t_count()
        == 8
    )
    assert (
        sum(g.name == "measure" for g in (*compute, *erase))
        == model_compute.measurement + model_erase.measurement
        == 2
    )
