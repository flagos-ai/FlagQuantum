"""Native Hamiltonian coefficients retain source finite and symbolic branches."""

from importlib import import_module

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np

    import_module("qualtran")
    import_module("cirq")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.simulation.numerics.hamiltonian_qsp_coefficients import (
    hamiltonian_qsp_coefficients,
)
from flagquantum.simulation.value_dimensions import SymbolicArrayShape
from tests.unit.test_controlled_hamiltonian_gqsp_source import (
    locked_hamiltonian_source as locked_hamiltonian_source,
)
from tests.unit.test_controlled_su2_source import locked_su2_source as locked_su2_source
from tests.unit.test_hamiltonian_gqsp import _source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "time,precision", ((0.0, 0.05), (0.15, 0.05), (-0.15, 0.05), (0.75, 1e-5))
)
def test_independent_coefficients_equal_source_and_reuse_cached_degree(
    time: float,
    precision: float,
) -> None:
    source = _source(time, precision)
    actual = hamiltonian_qsp_coefficients(source.alpha, time, precision)
    assert isinstance(actual, tuple)
    np.testing.assert_array_equal(np.asarray(actual), source.approx_cos)
    assert actual == hamiltonian_qsp_coefficients(
        source.alpha, time, precision, degree=int(source.degree)
    )


@pytest.mark.parametrize("argument", ("time", "precision", "both"))
def test_symbolic_metadata_keeps_original_shape_and_degree_expression(
    argument: str,
) -> None:
    import attrs
    import sympy

    source = _source(0.15, 0.05)
    t, eps = sympy.symbols("t eps", positive=True)
    source = attrs.evolve(
        source,
        t=t if argument != "precision" else 0.15,
        precision=eps if argument != "time" else 0.05,
    )
    actual = hamiltonian_qsp_coefficients(source.alpha, source.t, source.precision)
    assert isinstance(actual, SymbolicArrayShape)
    assert actual.shape == source.approx_cos.shape


def test_actual_symbolic_hubbard_shape_is_owned_without_source_numerics() -> None:
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        _symbolic_hamsim_by_gqsp,
    )

    source = _symbolic_hamsim_by_gqsp.make()
    actual = hamiltonian_qsp_coefficients(source.alpha, source.t, source.precision)
    assert isinstance(actual, SymbolicArrayShape)
    assert actual.shape == source.approx_cos.shape
