from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch

from flagquantum.algorithms.pca import (
    PcaResult,
    _PhaseFromDensityMatrix,
    principal_components,
)
from flagquantum.algorithms.primitives.types import ControlledUnitary
from flagquantum.circuit import Circuit

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


def test_the_mode_can_resolve_a_smaller_eigenvalue_when_the_peak_splits() -> None:
    """A readout that is not the largest eigenvalue: this spectrum's observed outcome.

    This case is in the file because it is the counterexample to a claim the module
    once made, that the readout is always within half a step of the largest
    eigenvalue. It is not, and this spectrum is one where it is not.

    At six counting wires the counter grid is 1/64 of a turn. The largest eigenvalue
    here is 0.5078125, whose phase ``1 - 0.5078125 = 0.4921875`` is exactly the
    midpoint between the counter values 31/64 = 0.484375 and 32/64 = 0.5. The second
    largest is 0.375, whose phase 0.625 is exactly counter value 40/64. Measured at
    8000 shots: counter 40 carries 0.3619 to 0.3910 depending on the sampling seed,
    and counters 31 and 32 carry 0.1959 to 0.2164 and 0.1971 to 0.2188 -- so the mode
    is counter 40, the second largest eigenvalue, and the largest is not within half
    a step of it. That readout and that answer held for every one of the 300 sampling
    seeds checked at each of 4096, 8000, 20000 and 50000 shots.

    The assertions are orderings and floors rather than a seed's third decimal,
    because the shares move with the sample: the only pinned value is the readout,
    which is the mode's counter value and does not. The two halves of the largest
    eigenvalue's spread are close enough to each other that their own order is a coin
    flip across seeds, so it is not asserted, and neither half is pinned to a value.

    What this pins is the outcome of a spectrum whose largest eigenvalue lost the
    mode. It is not a claim about spectra in general, and not a claim that a peak
    between two counter values always loses: the companion test below has a largest
    eigenvalue whose phase also sits between two counter values, and a mode that is
    that eigenvalue. A change to the readout or to the register layout moves every
    assertion here.
    """
    data = _data_matrix([0.5078125, 0.375, 0.097, 0.0201875], 4)
    exact = _exact_eigenvalues(data)
    assert [round(float(value), 6) for value in exact] == [
        0.020188,
        0.097,
        0.375,
        0.507812,
    ]
    largest = float(exact[-1])

    for seed in range(4):
        result = principal_components(
            data, n_counting_wires=_COUNTING_WIRES, shots=8000, seed=seed
        )
        reader = _counter_of(0.375, _COUNTING_WIRES)

        assert result.dominant_eigenvalue == pytest.approx(0.375, abs=1e-9), seed
        assert result.within(0.375), seed
        assert not result.within(largest), seed
        # The mode is counter 40, and both of the largest eigenvalue's counter values
        # carry less than it does -- measured minima 0.3619 against maximum 0.2188.
        assert (
            result.dominant_probability
            > result.distribution[_counter_of(0.5, _COUNTING_WIRES)]
        ), seed
        assert (
            result.dominant_probability
            > result.distribution[_counter_of(0.515625, _COUNTING_WIRES)]
        ), seed
        # The largest eigenvalue's weight is spread over both neighbours rather than
        # sitting on one: measured, each carries at least 0.1959 over 400 runs.
        for half in (0.5, 0.515625):
            assert result.distribution[_counter_of(half, _COUNTING_WIRES)] > 0.05, seed
        assert reader in result.distribution, seed


def test_the_readout_is_invariant_in_the_scale_of_the_data_matrix() -> None:
    """Doubling ``A`` leaves the readout identical, and only a normalised Gram does.

    Every other spectrum in this file is built by ``_data_matrix``, whose trace is
    exactly one, so ``A A^T`` and ``A A^T / tr(A A^T)`` are the same matrix there and
    none of them can tell the two apart. This one scales ``A`` by two, so the Gram
    matrix's trace is four and the division is the only thing that keeps the density
    matrix's spectrum -- and therefore the readout -- unchanged.

    The reference path is computed here from the scaled matrix, independently of the
    module: ``eigvalsh(A A^T / tr(A A^T))``. With the normalisation removed, the
    unitary becomes ``exp(-2 pi i * 4 rho)`` and one extra turn of phase folds every
    eigenvalue onto ``-4 lambda mod 1``: measured, the readout moves from 1.0 to
    0.984375, ``within`` the largest eigenvalue goes from ``True`` to ``False``, and
    the two distributions stop being equal.
    """
    data = _data_matrix([0.9962, 0.0038], 2)
    scaled = data * 2
    gram = scaled @ scaled.T
    assert float(torch.trace(gram)) == pytest.approx(4.0)
    largest = float(_exact_eigenvalues(scaled)[-1])

    reference = principal_components(
        data, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )
    result = principal_components(
        scaled, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )

    assert dict(result.distribution) == dict(reference.distribution)
    assert result.dominant_eigenvalue == reference.dominant_eigenvalue
    assert result.dominant_probability == reference.dominant_probability
    assert result.within(largest)


def test_the_sample_size_is_visible_in_the_distribution() -> None:
    """Every share is a count over exactly ``shots`` samples, so ``shots`` reaches it.

    ``shots`` is a public argument with a documented meaning -- the number of samples
    the distribution is drawn from -- so the relation has to be checkable from the
    result alone. It is: a share is ``count / shots`` for an integer count, so
    multiplying every share back by ``shots`` recovers whole samples, and those
    samples sum to ``shots``. Both hold at every width tried.

    The check is what makes the argument load-bearing rather than decorative. A run
    that ignored the caller's ``shots`` and fell back on the module's default 4096
    would return shares that are multiples of ``1 / 4096``, which are not multiples
    of ``1 / 20000``: measured under that change, the worst share times ``shots``
    lands 0.48 away from an integer and the recovered samples sum to 20008 instead of
    20000.
    """
    data = _data_matrix([0.9962, 0.0038], 2)

    for shots in (100, 250, 20000):
        result = principal_components(
            data, n_counting_wires=_COUNTING_WIRES, shots=shots, seed=_SEED
        )
        samples = [share * shots for share in result.distribution.values()]
        assert all(abs(sample - round(sample)) < 1e-9 for sample in samples), shots
        assert sum(round(sample) for sample in samples) == shots


def test_the_adapter_satisfies_the_controlled_unitary_protocol() -> None:
    """The power form is what phase estimation consumes; the other two are its contract.

    ``append_phase_estimation`` only ever calls ``apply_power_controlled``, so the
    bare and single-power forms on ``_PhaseFromDensityMatrix`` are never reached by
    ``principal_components``. They are not dead code to delete: they are two thirds
    of the ``ControlledUnitary`` protocol the adapter is passed as, and a protocol
    method that is never executed is a method whose block order can silently be
    wrong. So they are exercised here, against a density matrix whose exponential is
    known by hand.

    ``rho = diag(0.75, 0.25)``, so ``U = exp(-2 pi i rho)`` multiplies an eigenvector
    with eigenvalue 0.75 by ``exp(-2 pi i * 0.75) = exp(-1.5 pi i) = i``. The circuit
    starts every wire in ``|0>``, so the data register holds the eigenvector of
    0.75 and ``U`` turns its amplitude into that phase. The controlled case pins the
    block order: ``diag(I, U)`` means the control being ``|0>`` leaves the register
    alone, and the control being ``|1>`` is the branch that carries the phase. The
    opposite order is the same matrix written the other way round and would pass a
    test that only checked a phase appeared somewhere.
    """
    rho = torch.diag(torch.tensor([0.75, 0.25], dtype=torch.float64))
    adapter = _PhaseFromDensityMatrix(rho)

    assert isinstance(adapter, ControlledUnitary)
    assert adapter.n_wires == 1

    # ``complex(...)`` on both sides: ``pytest.approx`` only understands a tensor
    # when numpy is importable, and this repository does not declare numpy.
    phase = complex(torch.exp(torch.tensor(-1.5j * math.pi)))

    bare = Circuit(1)
    adapter.apply(bare, [0])
    assert complex(bare.state().reshape(-1)[0]) == pytest.approx(phase, abs=1e-6)

    control_off = Circuit(2)
    adapter.apply_controlled(control_off, 0, [1])
    assert [complex(value) for value in control_off.state().reshape(-1)] == (
        pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    )

    control_on = Circuit(2)
    control_on.gate("x", 0)
    adapter.apply_controlled(control_on, 0, [1])
    assert [complex(value) for value in control_on.state().reshape(-1)] == (
        pytest.approx([0.0, 0.0, phase, 0.0], abs=1e-6)
    )


def test_a_split_peak_wins_the_mode_when_its_half_beats_the_competitors() -> None:
    """A readout that is the largest eigenvalue, on a spectrum where its peak splits.

    The mode is the counter value with the largest share and nothing else, so which
    eigenvalue it reports is an outcome of the shares and not a rule about the shape
    of the peak. This test is one of the two outcomes: here the readout *is* the
    largest eigenvalue, on a spectrum whose largest eigenvalue's phase sits between
    two counter values. Without it, a readout that assumed "a peak between two
    counter values means the eigenvalue lost" -- the opposite overclaim to the one
    this file already corrected -- would pass every other test here.

    The largest eigenvalue is 0.51015625, at phase ``31.35/64``, so its counter
    values are 31 and 32. Measured at 8000 shots: over seeds 0 through 199, counter
    31 carries 0.3215 to 0.3464, counter 32 carries 0.0881 to 0.1045, and counter 45
    -- where the second largest eigenvalue, 0.296875, sits -- carries 0.2845 to
    0.3111. The mode is counter 31, the readout is 0.515625, and ``within`` the
    largest eigenvalue is ``True``: that readout and that answer held for every one
    of the 300 sampling seeds checked at each of 4096, 8000, 20000 and 50000 shots.

    The assertions are orderings and floors rather than a seed's third decimal: the
    shares move with the sample and the mode's counter value does not, so the readout
    is the only pinned value. The narrowest margin among them is counter 31's share
    against counter 45's -- measured, at 4096 shots over 200 seeds, a minimum of
    0.3179 against a maximum of 0.3167 -- and that comparison is the claim this test
    exists to make, so it is asserted as an ordering rather than as a value.
    """
    # The largest eigenvalue's phase is 31.35/64, so lambda = 1 - 31.35/64.
    split = [0.51015625, 0.296875, 0.096484375, 0.096484375]
    assert sum(split) == pytest.approx(1.0)

    data = _data_matrix(split, 4)
    exact = _exact_eigenvalues(data)
    largest = float(exact[-1])
    assert largest == pytest.approx(1.0 - 31.35 / 64)

    for seed in range(4):
        result = principal_components(
            data, n_counting_wires=_COUNTING_WIRES, shots=8000, seed=seed
        )
        near = result.distribution[_counter_of(0.515625, _COUNTING_WIRES)]
        far = result.distribution[_counter_of(0.5, _COUNTING_WIRES)]
        competitor = result.distribution[_counter_of(0.296875, _COUNTING_WIRES)]

        # The readout is counter 31, and it resolves the largest eigenvalue.
        assert result.dominant_eigenvalue == pytest.approx(0.515625, abs=1e-9), seed
        assert result.within(largest), seed
        assert result.dominant_probability == near, seed
        # The mode's share beats the competitor's, and the competitor's beats the
        # other half of the largest eigenvalue's spread.
        assert near > competitor, seed
        assert competitor > far, seed
        # Both counter values of the largest eigenvalue carry a real share: measured,
        # the smaller carries at least 0.0835 over 600 runs.
        assert far > 0.05, seed


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
    """A non-tensor, a 1-D tensor, an integer matrix and a non-power-of-two shape go.

    The shape messages are matched on this module's own wording rather than on the
    phrase "power of two", which the state-preparation primitive raises too: a
    regression that deleted the module's own check would let the amplitudes reach
    ``append_arbitrary_state`` and come back reporting a length, and a loose match
    would not notice.
    """
    with pytest.raises(ValueError, match="torch.Tensor"):
        principal_components([1.0, 0.0, 0.0, 1.0], n_counting_wires=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two-dimensional"):
        principal_components(torch.ones(4), n_counting_wires=4)
    with pytest.raises(ValueError, match="floating-point"):
        principal_components(torch.eye(2, dtype=torch.int64), n_counting_wires=4)
    with pytest.raises(ValueError, match=r"A's rows must be a power of two"):
        principal_components(torch.eye(3), n_counting_wires=4)
    with pytest.raises(ValueError, match=r"A's columns must be a power of two"):
        principal_components(torch.zeros(4, 3), n_counting_wires=4)
    # A power of two with one row or one column is still not a register: a single
    # wire carries two amplitudes, and a one-amplitude register has no density
    # matrix with a meaningful spectrum. Without this case the ``size < 2`` half of
    # the check can be deleted and the matrix is accepted instead, returning a
    # degenerate readout rather than refusing.
    with pytest.raises(
        ValueError, match=r"A's rows must be a power of two of at least two"
    ):
        principal_components(torch.ones(1, 2), n_counting_wires=4)
    with pytest.raises(
        ValueError, match=r"A's columns must be a power of two of at least two"
    ):
        principal_components(torch.ones(2, 1), n_counting_wires=4)


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
