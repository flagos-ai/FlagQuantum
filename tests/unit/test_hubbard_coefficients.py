"""Shared coefficient arithmetic matches locked source gates without early rounding."""

import sys
from itertools import product

import pytest

from flagquantum.algorithms.primitives import hubbard_prepare as algorithm
from flagquantum.simulation.numerics import hubbard_coefficients as numeric

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dimension", (2, 3, 9))
def test_symbolic_normalization_retains_exact_source_floor(dimension: int) -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard,
    )

    t, u = sympy.symbols("t u", real=True)
    expression = numeric.hubbard_normalization(dimension, t, u)
    assert expression == PrepareHubbard(dimension, dimension, t, u).l1_norm_of_coeffs
    large = 2**60 + 1
    assert expression.subs({t: large, u: large}) == numeric.hubbard_normalization(
        dimension, large, large
    )


@pytest.mark.parametrize(
    "dimension,weights",
    tuple(
        product(
            (2, 3, 9),
            (
                (1, 4),
                (0.37, 0.27),
                (0.0, 4.0),
                (1.0, 0.0),
                (-1.0, -4.0),
                (2**54 + 1, 2**54 + 3),
            ),
        )
    ),
)
def test_exact_source_normalization_and_first_two_rotation_parameters(
    dimension: int, weights: tuple[float, float]
) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard,
    )

    t, u = weights
    source = PrepareHubbard(dimension, dimension, t, u)
    assert numeric.hubbard_normalization(dimension, t, u) == source.l1_norm_of_coeffs
    assert type(numeric.hubbard_normalization(dimension, t, u)) is type(
        source.l1_norm_of_coeffs
    )
    ports = {}
    cursor = 0
    for register in source.signature:
        width = int(register.total_bits())
        ports[register.name] = np.array(
            cirq.LineQubit.range(cursor, cursor + width), dtype=object
        )
        cursor += width
    operations = source.decompose_from_registers(
        context=cirq.DecompositionContext(cirq.SimpleQubitManager()), **ports
    )
    first, second = next(operations), next(operations)
    assert isinstance(first, cirq.GateOperation) and isinstance(first.gate, cirq.Ry)
    assert isinstance(second, cirq.Operation) and isinstance(
        second.gate, cirq.ControlledGate
    )
    assert isinstance(second.gate.sub_gate, cirq.Ry)
    actual = numeric.hubbard_preparation_angles(dimension, t, u)
    assert actual == (float(first.gate._rads), float(second.gate.sub_gate._rads))
    assert algorithm._hubbard_preparation_angles(dimension, t, u) == actual


@pytest.mark.parametrize(
    "weights,error", (((0.0, 0.0), ZeroDivisionError), ((-1.0, 8.0), ValueError))
)
def test_existing_private_arithmetic_failures_are_preserved(
    weights: tuple[float, float], error: type[Exception]
) -> None:
    with pytest.raises(error):
        numeric.hubbard_preparation_angles(2, *weights)
    with pytest.raises(error):
        algorithm._hubbard_preparation_angles(2, *weights)


def test_normalization_is_numpy_free_and_algorithm_accepts_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "numpy", None)
        assert numeric.hubbard_normalization(2, 1, 4) == 32
        with pytest.raises(ModuleNotFoundError):
            numeric.hubbard_preparation_angles(2, 1, 4)
    monkeypatch.setattr(
        numeric, "hubbard_preparation_angles", lambda n, t, u: (0.25, 0.5)
    )
    assert algorithm._hubbard_preparation_angles(2, 1, 4) == (0.25, 0.5)


def test_ecosystem_conversion_no_longer_imports_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from flagquantum.ecosystem import _packed_oracle_component_library as adapter

    expected = numeric.hubbard_preparation_angles(2, 0.37, 0.27)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "flagquantum.algorithms", None)
        importlib.reload(adapter)
        assert adapter._prepare_angles(2, 0.37, 0.27) == expected
