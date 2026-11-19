"""Original discrete-log observations and independent complex-state acceptance."""

import math
from fractions import Fraction

import pytest
import torch

import flagquantum as fq

try:
    import numpy as np

    from examples.synthesis.algorithms.number_theory_and_cryptography.discrete_log.overview import (
        workflow as dl,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "matplotlib"}:
        raise
    pytest.skip(str(error), allow_module_level=True)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "modulus,seed,logarithm,qubits,operations,depth",
    [(5, 1605, 3, 8, 65, 50), (13, 1613, 8, 16, 1068, 872)],
)
def test_original_shots_recovery_full_state_and_resources(
    modulus: int, seed: int, logarithm: int, qubits: int, operations: int, depth: int
) -> None:
    result = dl.run_estimate(modulus, seed=seed)
    assert result.shots == (4000 if modulus == 5 else 10000)
    assert result.logarithm == logarithm
    assert result.maximum_state_error < 2e-11
    assert result.dirty_workspace_probability < 1e-24
    assert result.resources.n_qubits == qubits
    assert result.resources.n_operations == operations
    assert result.resources.depth == depth
    assert pow(result.case.generator, logarithm, modulus) == result.case.element
    assert all(isinstance(row.alpha, Fraction) for row in result.observations)
    if modulus == 13:
        assert any(row.logarithm != 8 for row in result.candidates[10:])
    else:
        assert all(row.alpha.denominator == 1 for row in result.observations)


@pytest.mark.parametrize(
    "modulus,element,generator", [(5, 3, 2), (5, 4, 3), (13, 2, 6), (13, 5, 11)]
)
def test_nonzero_bindings_complete_state_and_workspace(
    modulus: int, element: int, generator: int
) -> None:
    program = dl.build_program(modulus, element=element, generator=generator)
    result = fq.run(program)
    case = dl.source_case(modulus)
    actual = result.statevector().reshape(1 << case.logical_bits, -1)
    expected = (
        fq.run(dl.reference_circuit(modulus, element=element, generator=generator))
        .statevector()
        .flatten()
    )
    torch.testing.assert_close(actual[:, 0], expected, atol=2e-11, rtol=0)
    torch.testing.assert_close(
        expected,
        dl.analytic_state(modulus, element=element, generator=generator),
        atol=2e-11,
        rtol=0,
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24
    assert sum(
        program.probabilities(result, registers=("alpha", "beta", "func_res")).values()
    ) == pytest.approx(1)


def test_source_rounding_filters_and_unfiltered_approximation_tail() -> None:
    rows = (
        dl.Observation(Fraction(1, 8), Fraction(3, 32), 10),
        dl.Observation(Fraction(3, 8), Fraction(3, 32), 9),
        dl.Observation(Fraction(1, 4), Fraction(1, 2), 8),
    )
    selected = dl.select_candidates(13, rows)
    assert tuple(row.alpha_rounded for row in selected) == (2, 4)
    assert tuple(row.logarithm for row in selected) == (2, 4)
    assert dl.select_candidates(13, ()) == ()
    assert dl.select_candidates(5, (dl.Observation(Fraction(1), Fraction(2), 1),)) == ()
    # Preserve the source's missing final modulo in its mod-five formula.
    assert (
        dl.select_candidates(5, (dl.Observation(Fraction(3), Fraction(3), 1),))[
            0
        ].logarithm
        == 9
    )


@pytest.mark.parametrize(
    "row",
    [
        dl.Observation(Fraction(-1), Fraction(1), 1),
        dl.Observation(Fraction(1, 64), Fraction(1, 32), 1),
        dl.Observation(Fraction(1), Fraction(0), 1),
        dl.Observation(Fraction(0), Fraction(0), 0),
        dl.Observation(Fraction(0), Fraction(0), True),
    ],
)
def test_invalid_observation_refused(row: dl.Observation) -> None:
    with pytest.raises(ValueError):
        dl.select_candidates(13, (row,))


def test_noninvertible_selected_mod_five_alpha_refused() -> None:
    with pytest.raises(ValueError, match="inverse"):
        dl.select_candidates(5, (dl.Observation(Fraction(2), Fraction(1), 1),))


@pytest.mark.parametrize("modulus", [0, 7, True])
def test_unsupported_source_case_refused(modulus: int) -> None:
    with pytest.raises(ValueError):
        dl.build_program(modulus)


@pytest.mark.parametrize("base", [0, 13, -1, True])
def test_invalid_base_refused(base: int) -> None:
    with pytest.raises(ValueError):
        dl.build_program(13, element=base)


@pytest.mark.parametrize(
    "shots,seed", [(0, 1), (100001, 1), (True, 1), (1, -1), (1, 2**32), (1, True)]
)
def test_sampling_bounds_refused(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        dl.run_estimate(5, shots=shots, seed=seed)


@pytest.mark.parametrize("bits", [5, 8])
def test_source_illustrative_curves_against_sine_formula(bits: int) -> None:
    curves = dl.illustrative_curves(bits)
    assert curves.real_log_ratio == math.log(3) / math.log(7)
    assert curves.real_log_ratio != 8
    size = 1 << bits
    for values, center in (
        (curves.alpha, 5 * size / 12),
        (curves.beta, curves.real_log_ratio * 5 * size / 12),
    ):
        delta = np.arange(size) - center
        expected = (
            np.sin(math.pi * delta) / (size * np.sin(math.pi * delta / size))
        ) ** 2
        expected /= expected.sum()
        np.testing.assert_allclose(values, expected, atol=2e-14, rtol=0)
        assert sum(values) == pytest.approx(1)
    figure = dl.plot_curves(curves)
    for line, values in zip(
        figure.axes[0].lines, (curves.alpha, curves.beta), strict=True
    ):
        np.testing.assert_array_equal(line.get_ydata(), values)
    figure.clear()


@pytest.mark.parametrize("bits", [0, 4, True])
def test_unsupported_illustration_size_refused(bits: int) -> None:
    with pytest.raises(ValueError):
        dl.illustrative_curves(bits)
