from __future__ import annotations

import itertools

import pytest
import torch

from flagquantum.ecosystem.kaiwu import (
    KaiwuMatrixValidationError,
    KaiwuPrecisionError,
    canonicalize_ising_matrix,
    decode_qubo_spins,
    encode_qubo_as_ising,
    ising_energy,
    prepare_integer_precision,
)

pytestmark = pytest.mark.unit


def _reference_energy(matrix: list[list[float]], spins: tuple[int, ...]) -> float:
    quadratic = 0.0
    for row, spin_i in zip(matrix, spins, strict=True):
        for coefficient, spin_j in zip(row, spins, strict=True):
            quadratic += spin_i * coefficient * spin_j
    return -quadratic


def _all_spins(width: int) -> list[tuple[int, ...]]:
    return list(itertools.product((-1, 1), repeat=width))


def test_canonicalize_returns_detached_cpu_float64_copy() -> None:
    source = torch.tensor([[1.0, -0.25], [-0.25, 2.0]], requires_grad=True)

    result = canonicalize_ising_matrix(source)
    source.detach()[0, 0] = 99.0

    assert result.device.type == "cpu"
    assert result.dtype == torch.float64
    assert not result.requires_grad
    assert result[0, 0].item() == 1.0


@pytest.mark.parametrize(
    "matrix",
    (
        [],
        [1.0, 2.0],
        [[1.0, 2.0]],
        [[0.0, 1.0], [2.0, 0.0]],
        [[0.0, float("nan")], [float("nan"), 0.0]],
        [[0.0, complex(1.0, 2.0)], [complex(1.0, 2.0), 0.0]],
    ),
)
def test_canonicalize_rejects_invalid_matrices(matrix: object) -> None:
    with pytest.raises(KaiwuMatrixValidationError):
        canonicalize_ising_matrix(matrix)  # type: ignore[arg-type]


def test_canonicalize_accepts_only_explicit_symmetry_tolerance() -> None:
    matrix = [[0.0, 1.0], [1.0001, 0.0]]

    with pytest.raises(KaiwuMatrixValidationError):
        canonicalize_ising_matrix(matrix)

    result = canonicalize_ising_matrix(matrix, symmetry_tolerance=0.001)
    assert result.tolist() == [
        [0.0, pytest.approx(1.00005)],
        [pytest.approx(1.00005), 0.0],
    ]
    assert torch.equal(result, result.T)


@pytest.mark.parametrize(
    "tolerance",
    (True, "0.1", 1 + 0j, float("nan"), float("inf"), -0.1),
)
def test_canonicalize_rejects_invalid_symmetry_tolerance(tolerance: object) -> None:
    with pytest.raises(KaiwuMatrixValidationError):
        canonicalize_ising_matrix(
            [[0.0, 1.0], [1.0, 0.0]],
            symmetry_tolerance=tolerance,  # type: ignore[arg-type]
        )


def test_canonicalize_symmetry_normalization_avoids_finite_input_overflow() -> None:
    largest = torch.finfo(torch.float64).max

    result = canonicalize_ising_matrix([[largest, largest], [largest, largest]])

    assert bool(torch.isfinite(result).all())
    assert result.tolist() == [[largest, largest], [largest, largest]]


def test_ising_energy_matches_independent_reference_and_bias() -> None:
    matrix = [[0.5, 1.25, -0.5], [1.25, -1.0, 0.75], [-0.5, 0.75, 2.0]]
    spins = _all_spins(3)

    actual = ising_energy(matrix, spins, bias=3.5)
    expected = torch.tensor(
        [_reference_energy(matrix, state) + 3.5 for state in spins],
        dtype=torch.float64,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert ising_energy(matrix, spins[0]).ndim == 0


@pytest.mark.parametrize(
    "bias",
    (True, "1.0", 1 + 0j, float("nan"), float("inf")),
)
def test_ising_energy_rejects_invalid_bias(bias: object) -> None:
    with pytest.raises(KaiwuMatrixValidationError):
        ising_energy([[0.0]], [1], bias=bias)  # type: ignore[arg-type]


def test_ising_energy_rejects_finite_coefficients_that_overflow_evaluation() -> None:
    largest = torch.finfo(torch.float64).max

    with pytest.raises(KaiwuMatrixValidationError, match="overflowed"):
        ising_energy([[largest, largest], [largest, largest]], [1, 1])


def test_qubo_auxiliary_encoding_has_exhaustive_energy_parity() -> None:
    qubo = [[1.5, -2.0, 0.25], [-2.0, 3.0, 1.0], [0.25, 1.0, -0.5]]
    offset = 4.25
    encoding = encode_qubo_as_ising(qubo, offset=offset)

    for binary in itertools.product((0, 1), repeat=3):
        source = offset
        for row, value_i in zip(qubo, binary, strict=True):
            for coefficient, value_j in zip(row, binary, strict=True):
                source += value_i * coefficient * value_j

        problem_spins = tuple(2 * value - 1 for value in binary)
        for auxiliary in (-1, 1):
            encoded_spins = tuple(value * auxiliary for value in problem_spins) + (
                auxiliary,
            )
            actual = ising_energy(
                encoding.matrix,
                encoded_spins,
                bias=encoding.bias,
            )

            assert actual.item() == pytest.approx(source)
            assert decode_qubo_spins(encoded_spins).tolist() == list(binary)


@pytest.mark.parametrize(
    "offset",
    (True, "1.0", 1 + 0j, float("nan"), float("inf")),
)
def test_qubo_encoding_rejects_invalid_offset(offset: object) -> None:
    with pytest.raises(KaiwuMatrixValidationError):
        encode_qubo_as_ising([[0.0]], offset=offset)  # type: ignore[arg-type]


def test_qubo_encoding_rejects_finite_coefficients_that_overflow_conversion() -> None:
    largest = torch.finfo(torch.float64).max

    with pytest.raises(KaiwuMatrixValidationError, match="overflowed"):
        encode_qubo_as_ising([[largest, largest], [largest, largest]])


def test_decode_qubo_spins_is_batch_safe_and_gauge_invariant() -> None:
    result = decode_qubo_spins([[1, -1, 1], [-1, 1, -1]])

    assert result.dtype == torch.int64
    assert result.tolist() == [[1, 0], [1, 0]]


@pytest.mark.parametrize("spins", ([1], [1, 0], [[[1, -1]]]))
def test_decode_qubo_spins_rejects_invalid_input(spins: object) -> None:
    with pytest.raises(KaiwuMatrixValidationError):
        decode_qubo_spins(spins)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "spins",
    (
        [1.0, 0.0],
        [1.0, float("inf")],
        [1.0],
        [[[1.0, -1.0]]],
    ),
)
def test_ising_energy_rejects_invalid_spins(spins: object) -> None:
    with pytest.raises(KaiwuMatrixValidationError):
        ising_energy([[0.0, 1.0], [1.0, 0.0]], spins)  # type: ignore[arg-type]


def test_integer_precision_is_explicit_symmetric_and_auditable() -> None:
    matrix = [[0.0, 4.0, -2.0], [4.0, 0.0, 1.0], [-2.0, 1.0, 0.0]]

    report = prepare_integer_precision(matrix, target_min=-7, target_max=7)

    assert report.quantized.dtype == torch.int64
    assert report.quantized.min().item() >= -7
    assert report.quantized.max().item() <= 7
    assert torch.equal(report.quantized, report.quantized.T)
    assert report.scale_factor == pytest.approx(1.75)
    assert report.max_abs_error >= 0.0
    assert report.mean_abs_error >= 0.0


def test_integer_precision_preserves_fixture_minimizers() -> None:
    matrix = [[0.0, 3.0, -1.0], [3.0, 0.0, 2.0], [-1.0, 2.0, 0.0]]
    spins = _all_spins(3)
    report = prepare_integer_precision(matrix, target_min=-31, target_max=31)

    original = ising_energy(matrix, spins)
    prepared = ising_energy(report.quantized, spins)

    assert (
        torch.where(original == original.min())[0].tolist()
        == torch.where(prepared == prepared.min())[0].tolist()
    )


def test_integer_precision_handles_zero_matrix_without_division_by_zero() -> None:
    report = prepare_integer_precision([[0.0, 0.0], [0.0, 0.0]])

    assert report.scale_factor == 1.0
    assert report.max_abs_error == 0.0
    assert report.quantized.tolist() == [[0, 0], [0, 0]]


@pytest.mark.parametrize(
    ("target_min", "target_max"),
    ((0, 127), (-127, 0), (1, -1), (-1.5, 1), (-1, True)),
)
def test_integer_precision_rejects_invalid_ranges(
    target_min: object, target_max: object
) -> None:
    with pytest.raises(KaiwuPrecisionError):
        prepare_integer_precision(
            [[0.0]],
            target_min=target_min,  # type: ignore[arg-type]
            target_max=target_max,  # type: ignore[arg-type]
        )
