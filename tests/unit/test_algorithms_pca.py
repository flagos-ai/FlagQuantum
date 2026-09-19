from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch

from flagquantum.algorithms.pca import PcaResult, principal_components

pytestmark = pytest.mark.unit

# Every expected value in this file was produced by a real run of the construction
# it checks, at these widths: the counter register resolves a phase to 1/2**6, and
# 20000 shots puts the sampling error on a share of 0.8 at about 0.003.
_COUNTING_WIRES = 6
_SHOTS = 20000
_SEED = 11


def _data_matrix(spectrum: Sequence[float], rows: int) -> torch.Tensor:
    """Build a data matrix whose density matrix has exactly ``spectrum`` eigenvalues.

    The spectrum is placed on the diagonal and the trace is already one, so
    ``A A^T / tr(A A^T)`` is the diagonal matrix of that spectrum. ``rows`` may
    exceed the number of eigenvalues, which is how a rank-deficient data matrix,
    and so a density matrix with zero eigenvalues, is built here. The matrix is
    real and held in double precision, so its spectrum is exact up to the square
    root and back.
    """
    matrix = torch.zeros((rows, len(spectrum)), dtype=torch.float64)
    for index, value in enumerate(spectrum):
        matrix[index, index] = math.sqrt(value)
    return matrix


def _exact_eigenvalues(data: torch.Tensor) -> torch.Tensor:
    """Return the spectrum of ``A``'s density matrix, computed here rather than read out.

    This is the independent reference path: it forms ``A A^T / tr(A A^T)`` from the
    data matrix directly and diagonalises it, so it shares no code with the module
    under test and no value with it.
    """
    gram = data @ data.T
    return torch.linalg.eigvalsh(gram / torch.trace(gram))


def _half_step(n_counting_wires: int) -> float:
    """Return half a counter step, in eigenvalue units.

    One counter value is ``2**-n`` of a turn and ``lambda = 1 - phi`` carries a phase
    step to an eigenvalue step of the same size, so this is the accuracy the readout
    claims. It is a property of the register width, not of any particular run.
    """
    return 0.5 / 2**n_counting_wires


def _counter_of(eigenvalue: float, n_counting_wires: int) -> str:
    """Return the counter value whose readout is ``eigenvalue``, as a bit string."""
    value = round((1.0 - eigenvalue) * 2**n_counting_wires)
    return format(value, f"0{n_counting_wires}b")


def test_the_mode_resolves_the_largest_eigenvalue_on_one_wire_per_register() -> None:
    """One data wire and one purification wire: the mode is the largest eigenvalue.

    The largest eigenvalue is 0.9962, whose phase ``1 - 0.9962`` is a small fraction
    of a turn, so it lands on the counter value 0 and reads back as ``1 - 0 = 1``.
    Half a counter step is 0.5/64 = 0.0078, which is what the two paths are required
    to agree to.
    """
    data = _data_matrix([0.9962, 0.0038], 2)
    exact = _exact_eigenvalues(data)
    assert float(exact[0]) == pytest.approx(0.0038, abs=1e-4)
    assert float(exact[-1]) == pytest.approx(0.9962, abs=1e-4)

    result = principal_components(
        data, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )

    assert result.n_counting_wires == _COUNTING_WIRES
    assert abs(result.dominant_eigenvalue - float(exact[-1])) <= _half_step(
        _COUNTING_WIRES
    )


def test_the_mode_resolves_the_largest_eigenvalue_on_two_data_wires() -> None:
    """Two data wires and one purification wire, so the density matrix is rank deficient.

    The two zero eigenvalues have phase ``1 - 0 = 0`` as well and would share the
    counter value 0, but they carry no weight, so the dominant eigenvalue 0.7374 is
    still what the mode reads. Its phase is 0.2626, which is 0.0030 away from the
    counter value 17/64 = 0.2656, well inside half a step.
    """
    data = _data_matrix([0.7374, 0.2626], 4)
    exact = _exact_eigenvalues(data)
    assert [round(float(value), 4) for value in exact] == [0.0, 0.0, 0.2626, 0.7374]

    result = principal_components(
        data, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )

    assert abs(result.dominant_eigenvalue - float(exact[-1])) <= _half_step(
        _COUNTING_WIRES
    )


def test_the_dominant_counter_value_carries_the_largest_share() -> None:
    """The mode's share beats every other counter value's, and that is the whole ordering.

    The share is not the eigenvalue -- see the tail test below -- so what the readout
    supports is a ranking of counter values, not a weight per eigenvalue. This is the
    assertion that survives the tail leak, and the one a caller may build on.
    """
    data = _data_matrix([0.9962, 0.0038], 2)

    result = principal_components(
        data, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )

    mode = max(result.distribution, key=result.distribution.__getitem__)
    assert result.distribution[mode] == result.dominant_probability
    assert all(
        result.dominant_probability > share
        for key, share in result.distribution.items()
        if key != mode
    )


def test_the_distribution_is_the_counter_registers_marginal() -> None:
    """The readout is over the counter register alone, so its keys carry only its wires.

    Sample keys span every wire, so a distribution that kept the full keys would not
    even be indexable by a counter value; the one below is keyed by the leading
    ``n_counting_wires`` characters and its shares sum to one.
    """
    data = _data_matrix([0.7374, 0.2626], 4)

    result = principal_components(
        data, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )

    assert result.distribution
    assert all(
        len(key) == _COUNTING_WIRES and key.strip("01") == ""
        for key in result.distribution
    )
    assert sum(result.distribution.values()) == pytest.approx(1.0)


def test_the_phase_to_eigenvalue_round_trip_is_exact_on_the_counter_grid() -> None:
    """Eigenvalues that sit on counter values come back as themselves.

    0.75 and 0.25 are three quarters and one quarter of a turn, so their phases are
    counter values exactly at this width and no tail is spread away from them. The
    readout reports both within half a step, and -- only in this alignment -- the
    shares are the eigenvalues, which is the case the tail test below contrasts with.
    """
    data = _data_matrix([0.75, 0.25], 2)

    result = principal_components(
        data, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )

    assert result.dominant_eigenvalue == pytest.approx(
        0.75, abs=_half_step(_COUNTING_WIRES)
    )
    assert result.within(0.75)
    assert not result.within(0.25)
    assert result.distribution[_counter_of(0.75, _COUNTING_WIRES)] == (
        pytest.approx(0.75, abs=0.02)
    )
    assert result.distribution[_counter_of(0.25, _COUNTING_WIRES)] == (
        pytest.approx(0.25, abs=0.02)
    )


def test_the_counter_value_nearest_the_small_eigenvalue_carries_its_tail() -> None:
    """The share where a small eigenvalue would be read is the dominant peak's tail.

    Measured: the counter value nearest the small eigenvalue 0.0038 carries 0.0331 of
    the sample, about nine times that eigenvalue. Phase estimation spreads the
    dominant peak's tail into the neighbouring counter value, and for a small
    eigenvalue that neighbour is where the share lands, so the share belongs to the
    dominant peak rather than to the small eigenvalue. Nothing in this file may
    therefore assert that a measured share equals an eigenvalue, and the bound below
    is deliberately loose: the ratio is the point, not its exact value.
    """
    data = _data_matrix([0.9962, 0.0038], 2)
    small = float(_exact_eigenvalues(data)[0])

    result = principal_components(
        data, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )

    nearest = min(
        result.distribution,
        key=lambda key: abs(1.0 - int(key, 2) / 2**_COUNTING_WIRES - small),
    )
    assert result.distribution[nearest] > 4 * small
    assert result.distribution[nearest] < result.dominant_probability


def test_the_resolution_is_the_counter_step_in_eigenvalue_units() -> None:
    """``lambda = 1 - phi`` carries the register's step straight into eigenvalue units.

    A wider counting register resolves a finer phase, and the eigenvalue step narrows
    with it, so the readout's accuracy is a property of the register that the result
    reports rather than of the sample it happened to draw. ``within`` is checked at
    its boundary on both sides: half a step holds and a full step does not. At these
    widths the step and the readout are both exact binary fractions, so the boundary
    itself is exact and needs no tolerance.
    """
    data = _data_matrix([0.9962, 0.0038], 2)

    for n_counting_wires in (3, 4, 5):
        result = principal_components(
            data, n_counting_wires=n_counting_wires, shots=512, seed=0
        )
        assert result.resolution == pytest.approx(1 / 2**n_counting_wires)
        assert result.within(result.dominant_eigenvalue - result.resolution / 2)
        assert not result.within(result.dominant_eigenvalue - result.resolution)


def test_a_zero_data_matrix_is_refused() -> None:
    """The zero matrix is refused rather than normalised into NaN eigenvalues."""
    with pytest.raises(ValueError, match="trace"):
        principal_components(torch.zeros(2, 2), n_counting_wires=4)


def test_a_non_finite_data_matrix_is_refused() -> None:
    """One non-finite entry makes every eigenvalue undefined, so it is refused."""
    data = torch.tensor([[1.0, 0.0], [0.0, float("nan")]])
    with pytest.raises(ValueError, match="finite"):
        principal_components(data, n_counting_wires=4)


def test_the_data_matrix_is_validated() -> None:
    """A non-tensor, a 1-D tensor, an integer matrix and a non-power-of-two shape go."""

    with pytest.raises(ValueError, match="torch.Tensor"):
        principal_components([1.0, 0.0, 0.0, 1.0], n_counting_wires=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two-dimensional"):
        principal_components(torch.ones(4), n_counting_wires=4)
    with pytest.raises(ValueError, match="floating-point"):
        principal_components(torch.eye(2, dtype=torch.int64), n_counting_wires=4)
    with pytest.raises(ValueError, match="power of two"):
        principal_components(torch.eye(3), n_counting_wires=4)


def test_the_run_validates_its_counting_width_and_shot_count() -> None:
    """A register with no wires and a sample with no shots are both refused."""
    data = _data_matrix([0.9962, 0.0038], 2)

    with pytest.raises(ValueError, match="counting wire"):
        principal_components(data, n_counting_wires=0)
    with pytest.raises(ValueError, match="at least one shot"):
        principal_components(data, n_counting_wires=3, shots=0)


def test_the_result_validates_its_own_fields() -> None:
    """A readout that could not come from a density matrix is refused at construction."""
    fields = {
        "distribution": {"0000": 1.0},
        "n_counting_wires": 4,
        "resolution": 0.0625,
    }

    with pytest.raises(ValueError, match="eigenvalue readout"):
        PcaResult(dominant_eigenvalue=0.0, dominant_probability=1.0, **fields)
    with pytest.raises(ValueError, match="sample share"):
        PcaResult(dominant_eigenvalue=0.5, dominant_probability=1.5, **fields)
    with pytest.raises(ValueError, match="must not be empty"):
        PcaResult(
            dominant_eigenvalue=0.5,
            dominant_probability=1.0,
            distribution={},
            n_counting_wires=4,
            resolution=0.0625,
        )
    with pytest.raises(ValueError, match="at least one wire"):
        PcaResult(
            dominant_eigenvalue=0.5,
            dominant_probability=1.0,
            distribution={"0": 1.0},
            n_counting_wires=0,
            resolution=0.0625,
        )
    with pytest.raises(ValueError, match="step must be positive"):
        PcaResult(
            dominant_eigenvalue=0.5,
            dominant_probability=1.0,
            distribution={"0": 1.0},
            n_counting_wires=1,
            resolution=0.0,
        )


def test_the_same_seed_replays_the_same_result_field_for_field() -> None:
    """The seed decides the sample, so the same seed rebuilds the same result."""
    data = _data_matrix([0.7374, 0.2626], 4)

    first = principal_components(data, n_counting_wires=5, shots=2048, seed=3)
    second = principal_components(data, n_counting_wires=5, shots=2048, seed=3)

    assert first == second
    assert first.distribution == second.distribution
    assert dict(first.distribution) == dict(second.distribution)
    assert first.dominant_eigenvalue == second.dominant_eigenvalue
    assert first.dominant_probability == second.dominant_probability
