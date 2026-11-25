"""Complete source-size Hidden Shift and independent binary reconstruction."""

from itertools import product

import pytest
import torch

from examples.synthesis.algorithms.number_theory_and_cryptography.hidden_shift.overview import (
    workflow as hs,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("route", ["simple", "dual", "no_dual"])
def test_original_source_dimensions_complete_states_and_2048_samples(
    route: hs.Route,
) -> None:
    result = hs.run_estimate(
        route, seed=1500 + ({"simple": 0, "dual": 1, "no_dual": 2}[route])
    )
    bits = 4 if route == "simple" else 16
    expected_shift = tuple(
        int(bit in ((0,) if route == "simple" else hs.SHIFT)) for bit in range(bits)
    )
    assert result.recovered == expected_shift
    assert sum(count for _, count in result.counts) == 2048
    assert result.maximum_state_error < 2e-11
    assert result.dirty_workspace_probability < 1e-24
    assert result.resources.n_qubits == (
        {"simple": 4, "dual": 16, "no_dual": 21}[route]
    )
    if route == "no_dual":
        assert len(result.counts) > 1000
        assert all(
            (word & 1) == sum(((word // 2) >> bit) & 1 for bit in hs.SHIFT) % 2
            for word, _ in result.counts
        )
    else:
        assert result.counts == ((1 if route == "simple" else 522, 2048),)


def test_source_permutation_dual_and_full_boolean_identity() -> None:
    assert hs.PERMUTATION == (3, 6, 1, 5, 7, 0, 4, 2)
    assert tuple(hs.INVERSE[i] for i in hs.PERMUTATION) == tuple(range(8))
    # Independently evaluate the complete Walsh spectrum of the original bent f.
    values = []
    for word in range(65536):
        x = [word >> i & 1 for i in range(8)]
        y = [word >> (8 + i) & 1 for i in range(8)]
        value = sum(x[i] * y[hs.PERMUTATION[i]] for i in range(8)) % 2
        value ^= int(all(y))
        values.append(1.0 if value == 0 else -1.0)
    spectrum = torch.tensor(values, dtype=torch.float64)
    for bit in range(16):
        blocks = spectrum.reshape(-1, 2, 1 << bit)
        first = blocks[:, 0].clone()
        second = blocks[:, 1].clone()
        blocks[:, 0] = first + second
        blocks[:, 1] = first - second
    for word, coefficient in enumerate(spectrum.tolist()):
        x = [word >> i & 1 for i in range(8)]
        y = [word >> (8 + i) & 1 for i in range(8)]
        dual = sum(x[hs.INVERSE[i]] * y[i] for i in range(8)) % 2
        dual ^= int(all(x))
        assert coefficient == 256 * (-1 if dual else 1)


@pytest.mark.parametrize("shift", list(product(range(2), repeat=3)))
def test_exact_gf2_reconstruction_not_real_rank(shift: tuple[int, ...]) -> None:
    # First three vectors are dependent in GF(2), but independent over the reals.
    vectors = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 1)]
    equations = [
        (row, sum(a * b for a, b in zip(row, shift, strict=True)) % 2)
        for row in vectors
    ]
    assert hs.recover_shift(equations, bits=3) == shift


@pytest.mark.parametrize(
    "samples,bits",
    [
        ([], 3),
        ([((1, 0, 0), 1)], 3),
        ([((1, 2, 0), 0)], 3),
        ([((1, 0), 0)], 3),
        ([((True, 0, 0), 0)], 3),
        ([((1, 0, 0), 2)], 3),
    ],
)
def test_insufficient_or_invalid_equations_refused(
    samples: list[tuple[tuple[int, ...], int]], bits: int
) -> None:
    with pytest.raises(ValueError):
        hs.recover_shift(samples, bits=bits)


def test_inconsistent_extra_equation_refused() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        hs.recover_shift([((1, 0), 1), ((0, 1), 0), ((1, 1), 0)], bits=2)


@pytest.mark.parametrize(
    "shots,seed", [(0, 1), (True, 1), (100001, 1), (1, -1), (1, 2**32), (1, True)]
)
def test_sampling_bounds_fail_before_compile(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        hs.run_estimate("simple", shots=shots, seed=seed)


def test_route_refuses_unrecognized_value() -> None:
    from typing import cast

    with pytest.raises(ValueError):
        hs.build_program(cast(hs.Route, "unknown"))
    with pytest.raises(ValueError):
        hs.reference_circuit(cast(hs.Route, "unknown"))
