"""Ordinary interaction estimates follow the unmodified symbolic source graph."""

from dataclasses import fields
from math import pi

import pytest

from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_interaction_cost,
    source_hubbard_interaction_formula,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_hubbard_interaction_symbols import frozen_source as frozen_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("angle", (0.0, pi / 8, -pi / 4, 0.21, "symbolic"))
@pytest.mark.parametrize("length", (1, 2, 3, 8, "symbolic", "bare"))
def test_interaction_symbolic_categories_and_peak_match_source(angle, length):
    from qualtran.bloqs.chemistry.trotter.hubbard.interaction import Interaction
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol, simplify

    angle = Symbol("theta", real=True) if angle == "symbolic" else angle
    if length == "symbolic":
        length = Symbol("L", positive=True, integer=True)
    elif length == "bare":
        length = Symbol("L")
    source = Interaction(length, angle, 2)
    formula = source_hubbard_interaction_formula(length, angle, 2)
    gates = get_cost_value(source, QECGatesCost())
    for field in fields(formula):
        expected = (
            get_cost_value(source, QubitCount())
            if field.name == "logical_qubits"
            else getattr(gates, field.name)
        )
        assert simplify(getattr(formula, field.name) - expected) == 0
    if isinstance(length, int) and isinstance(angle, float):
        concrete = estimate_component_resources(
            source_hubbard_interaction_cost(length, angle, 2)
        )
        assert concrete.peak_logical_qubits == formula.logical_qubits
        for name in ("t", "clifford", "rotation"):
            assert getattr(concrete.gates, name) == getattr(formula, name)


def test_symbolic_hwp_source_failures_are_preserved():
    from qualtran.bloqs.chemistry.trotter.hubbard.interaction import InteractionHWP
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    source = InteractionHWP(Symbol("L", positive=True, integer=True), 0.21, 2)
    with pytest.raises(AttributeError, match="bit_length"):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(RuntimeError, match="Split cannot have a symbolic data type"):
        get_cost_value(source, QubitCount())


def test_one_site_hwp_source_failures_are_preserved():
    from qualtran.bloqs.chemistry.trotter.hubbard.interaction import InteractionHWP
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = InteractionHWP(1, 0.21, 2)
    with pytest.raises(ZeroDivisionError):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(RuntimeError, match="HammingWeightPhasing"):
        get_cost_value(source, QubitCount())


@pytest.mark.parametrize("length", (-2, 0))
def test_numeric_interaction_formula_retains_native_admission(length):
    with pytest.raises(ValueError, match="positive integer"):
        source_hubbard_interaction_formula(length, 0.21, 2)


@pytest.mark.parametrize("angle", (float("nan"), float("inf")))
def test_numeric_rotation_formula_rejects_nonfinite_angles(angle):
    with pytest.raises(ValueError, match="finite"):
        source_hubbard_interaction_formula(2, angle, 2)
