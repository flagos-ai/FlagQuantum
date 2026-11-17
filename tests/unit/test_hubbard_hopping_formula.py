"""Ordinary hopping estimates follow the unmodified symbolic source graph."""

from dataclasses import fields
from math import pi

import pytest

from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_hopping_plaquette_formula,
    source_hubbard_hopping_tile_cost,
    source_hubbard_hopping_tile_formula,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_hubbard_hopping_symbols import frozen_source as frozen_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("angle", (0.0, pi / 4, -pi / 2, 0.21, "symbolic"))
@pytest.mark.parametrize("length", (None, 2, 8, "symbolic"))
def test_plaquette_and_tile_symbolic_categories_and_peak_match_source(angle, length):
    from qualtran.bloqs.chemistry.trotter.hubbard.hopping import (
        HoppingPlaquette,
        HoppingTile,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol, simplify

    angle = Symbol("theta", real=True) if angle == "symbolic" else angle
    length = (
        Symbol("L", positive=True, integer=True) if length == "symbolic" else length
    )
    if length is None:
        source = HoppingPlaquette(angle)
        formula = source_hubbard_hopping_plaquette_formula(angle)
    else:
        source = HoppingTile(length, angle)
        formula = source_hubbard_hopping_tile_formula(length, angle)
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
            source_hubbard_hopping_tile_cost(length, angle)
        )
        assert concrete.peak_logical_qubits == formula.logical_qubits
        for name in ("t", "clifford", "rotation"):
            assert getattr(concrete.gates, name) == getattr(formula, name)


def test_symbolic_hwp_source_failures_are_preserved():
    from qualtran.bloqs.chemistry.trotter.hubbard.hopping import HoppingTileHWP
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    source = HoppingTileHWP(Symbol("L", positive=True, integer=True), 0.21)
    with pytest.raises(AttributeError, match="bit_length"):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(RuntimeError, match="Split cannot have a symbolic data type"):
        get_cost_value(source, QubitCount())


@pytest.mark.parametrize("length", (-2, 0, 1, 3))
def test_numeric_tile_formula_retains_native_admission(length):
    with pytest.raises(ValueError, match="positive even"):
        source_hubbard_hopping_tile_formula(length, 0.21)


@pytest.mark.parametrize("angle", (float("nan"), float("inf")))
def test_numeric_rotation_formula_rejects_nonfinite_angles(angle):
    with pytest.raises(ValueError, match="finite"):
        source_hubbard_hopping_plaquette_formula(angle)
