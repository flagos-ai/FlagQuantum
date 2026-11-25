"""Compare complete symbolic potential gate and connected-width source queries."""

from dataclasses import fields
from typing import Any

import pytest
import sympy

from flagquantum.algorithms.primitives.grid_potential_symbolic_resources import (
    source_grid_pair_potential_logical_width,
    source_grid_pair_potential_resource_formula,
)
from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from tests.unit.test_grid_potential_source import _pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_N, _P, _T = sympy.symbols("n p t")


def _arguments(width: Any, polynomial: Any, target: Any) -> dict[str, Any]:
    return {
        "bitsize": width,
        "poly_bitsize": polynomial,
        "inv_sqrt_bitsize": target,
        "qrom_data": _pair().qrom_data,
    }


@pytest.mark.parametrize(
    "width,polynomial,target",
    (
        (1, 3, 5),
        (2, 4, 6),
        (_N, 3, 5),
        (1, _P, 5),
        (1, 3, _T),
        (_N, _P, _T),
        (2 * _N + 1, _P + 1, _T + 2),
    ),
)
def test_complete_symbolic_peak_matches_source_connection_schedule(
    width: Any, polynomial: Any, target: Any
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential
    from qualtran.resource_counting import QubitCount, get_cost_value

    arguments = _arguments(width, polynomial, target)
    source = PairPotential(**arguments)
    expected = get_cost_value(source, QubitCount())
    actual = source_grid_pair_potential_logical_width(arguments)
    assert sympy.simplify(actual - expected) == 0


@pytest.mark.parametrize("legacy_shims", (False, True))
@pytest.mark.parametrize("polynomial,target", ((3, 5), (_P, 5), (3, _T), (_P, _T)))
def test_complete_symbolic_gate_categories_and_peak_both_profiles(
    polynomial: Any, target: Any, legacy_shims: bool
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    arguments = _arguments(1, polynomial, target)
    source = PairPotential(**arguments)
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy_shims))
    actual = source_grid_pair_potential_resource_formula(
        arguments, legacy_shims=legacy_shims
    )
    for field in fields(QECGateCost):
        assert (
            sympy.simplify(getattr(actual, field.name) - getattr(expected, field.name))
            == 0
        )
    assert (
        sympy.simplify(actual.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )


def test_symbolic_selection_gate_failure_does_not_remove_successful_peak() -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    arguments = _arguments(_N, 3, 5)
    with pytest.raises(TypeError, match="cannot determine truth value") as original:
        get_cost_value(PairPotential(**arguments), QECGatesCost())
    with pytest.raises(TypeError, match="cannot determine truth value") as native:
        source_grid_pair_potential_resource_formula(arguments)
    assert str(original.value) == str(native.value)
    assert sympy.sympify(source_grid_pair_potential_logical_width(arguments)).has(_N)


@pytest.mark.parametrize("rows,zero", ((5, False), (16, False), (16, True)))
@pytest.mark.parametrize("legacy_shims", (False, True))
def test_symbolic_precision_counts_actual_static_banks(
    rows: int, zero: bool, legacy_shims: bool
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    arguments = _arguments(1, _P, 5)
    arguments["qrom_data"] = tuple(
        tuple(0 if zero else (row + index) % 8 for index in range(rows))
        for row in range(4)
    )
    source = PairPotential(**arguments)
    if zero:
        with pytest.raises(
            DecomposeTypeError, match="Cannot decompose parameterized QROM"
        ):
            get_cost_value(source, QECGatesCost(legacy_shims=legacy_shims))
        with pytest.raises(NotImplementedError, match="parameterized uniform data"):
            source_grid_pair_potential_resource_formula(
                arguments, legacy_shims=legacy_shims
            )
        assert (
            sympy.simplify(
                source_grid_pair_potential_logical_width(arguments)
                - get_cost_value(source, QubitCount())
            )
            == 0
        )
        return
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy_shims))
    actual = source_grid_pair_potential_resource_formula(
        arguments, legacy_shims=legacy_shims
    )
    for field in fields(QECGateCost):
        assert (
            sympy.simplify(getattr(actual, field.name) - getattr(expected, field.name))
            == 0
        )


def test_models_use_owned_graph_with_reference_sdks_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    arguments = _arguments(1, _P, 5)
    expected = source_grid_pair_potential_resource_formula(arguments)
    original = builtins.__import__

    def no_reference(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_reference)
    assert source_grid_pair_potential_resource_formula(arguments) == expected
