"""Signed mean powers retain source categories and direction-sensitive peaks."""

from dataclasses import fields
from typing import Any

import pytest
import sympy

from flagquantum.algorithms.primitives.mean_estimation_operator import (
    CodeForRandomVariable,
    MeanEstimationOperator,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _native(precision: int, power: int) -> MeanEstimationOperator:
    q = Register("q", bounded_unsigned_register_type(1, 2))
    t = Register("t", opaque_register_type(1))
    return (
        MeanEstimationOperator(CodeForRandomVariable((q,), (q,), (t,)), precision)
        ** power
    )


def _formula(source: Any) -> SymbolicResourceFormula:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    gates = get_cost_value(source, QECGatesCost())
    return SymbolicResourceFormula(
        logical_qubits=get_cost_value(source, QubitCount()),
        **{f.name: getattr(gates, f.name) for f in fields(GateCost)},
    )


@pytest.mark.parametrize("precision", [1, 2, 5, 32])
@pytest.mark.parametrize("power", [1, 2, 3, -1, -2, -3])
def test_uncontrolled_powers_match_all_original_gate_categories_and_peak(
    precision: int,
    power: int,
) -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator as SourceMean,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
    )

    prep, encoder = BernoulliSynthesizer(0.25, 1), BernoulliEncoder(0.25, (0, 1), 1, 1)
    source = SourceMean(SourceCode(prep, encoder), precision)
    encoder_cost, inverse_cost = _formula(encoder), _formula(encoder.adjoint())
    prep_cost, unprep_cost = _formula(prep), _formula(prep.adjoint())
    left = ComponentCost(
        "prepare",
        int(prep_cost.logical_qubits),
        GateCost(**{f.name: getattr(prep_cost, f.name) for f in fields(GateCost)}),
    )
    right = ComponentCost(
        "unprepare",
        int(unprep_cost.logical_qubits),
        GateCost(**{f.name: getattr(unprep_cost, f.name) for f in fields(GateCost)}),
    )
    actual = _native(precision, power).source_resource_model(
        encoder_cost, inverse_cost, left, right
    )
    expected = _formula(source**power)
    for field in fields(expected):
        assert (
            sympy.simplify(getattr(actual, field.name) - getattr(expected, field.name))
            == 0
        ), field.name


def test_controlled_source_failure_remains_explicit() -> None:
    from qualtran import CtrlSpec, DecomposeTypeError
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator as SourceMean,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliEncoder,
        BernoulliSynthesizer,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = SourceMean(
        SourceCode(BernoulliSynthesizer(0.25, 1), BernoulliEncoder(0.25, (0, 1), 1, 1)),
        2,
    )
    for cv in (0, 1):
        with pytest.raises(DecomposeTypeError, match=r"C\[And\]"):
            get_cost_value(source.controlled(CtrlSpec(cvs=cv)), QECGatesCost())
        native = _native(2, 2).controlled(cv)
        with pytest.raises(ValueError, match="forward uncontrolled"):
            native.source_resource_model(
                SymbolicResourceFormula(0, 1, 0, 2),
                SymbolicResourceFormula(0, 1, 0, 2),
                ComponentCost("p", 1),
                ComponentCost("p_inv", 1),
            )


@pytest.mark.parametrize(
    "forward_peak,inverse_peak,expected_peak", [(4, 7, 7), (11, 4, 11), (2, 2, 5)]
)
def test_encoder_lifetimes_match_both_original_call_directions(
    forward_peak: int,
    inverse_peak: int,
    expected_peak: int,
) -> None:
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        CodeForRandomVariable as SourceCode,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator import (
        MeanEstimationOperator as SourceMean,
    )
    from qualtran.bloqs.mean_estimation.mean_estimation_operator_test import (
        BernoulliSynthesizer,
    )

    from tests.unit.mean_cost_reference import source_encoder

    encoder = source_encoder(forward_peak, inverse_peak)
    prep = BernoulliSynthesizer(0.25, 1)
    source = SourceMean(SourceCode(prep, encoder), 1)
    forward, inverse = _formula(encoder), _formula(encoder.adjoint())
    prep_formula = _formula(prep)
    prep_cost = ComponentCost(
        "prepare",
        int(prep_formula.logical_qubits),
        GateCost(**{f.name: getattr(prep_formula, f.name) for f in fields(GateCost)}),
    )
    for native_phase, original_phase in (
        (_native(1, 1).select.source_resource_model(forward, inverse), source.select),
        (
            _native(1, 1).select.source_resource_model(inverse, forward),
            source.select.adjoint(),
        ),
    ):
        original_cost = _formula(original_phase)
        for field in fields(original_cost):
            assert (
                sympy.simplify(
                    getattr(native_phase, field.name)
                    - getattr(original_cost, field.name)
                )
                == 0
            ), field.name
    peaks = []
    for power in (1, 2, -1, -2):
        actual = _native(1, power).source_resource_model(
            forward, inverse, prep_cost, prep_cost
        )
        expected = _formula(source**power)
        for field in fields(expected):
            assert (
                sympy.simplify(
                    getattr(actual, field.name) - getattr(expected, field.name)
                )
                == 0
            ), field.name
        peaks.append(actual.logical_qubits)
    assert peaks == [expected_peak] * 4
