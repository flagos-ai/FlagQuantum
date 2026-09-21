"""Unit tests for the marginal reduction in :mod:`flagquantum.runtime.measurements`.

The module used to reach a joint marginal through ``2 ** k - 1`` parity
expectations, each of which contracted the whole state. A dense result now gives
it by reducing the distribution it already carries, and the parity reconstruction
remains for a result that is not dense.

Four things are pinned here, and the last three are what make the first worth
anything:

* the reduction reproduces the parity reconstruction, which is still reachable
  and still tested rather than deleted;
* the reduction is the path actually taken, observed by counting the
  ``expectation_ps`` calls rather than by reading the output, since both paths
  produce the same numbers;
* a request that names its wires out of order is restored to wire order, which
  would otherwise silently transpose the marginal instead of failing;
* an MPS result keeps the parity path, because for it the reduction would mean
  materialising a dense state that the representation exists to avoid. That
  boundary was found by measurement, not assumed: at 24 wires the reduction took
  330 ms against the parity path's 0.5 ms.
"""

from __future__ import annotations

import time

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.measurements as measurements
from flagquantum.core.ir import MeasurementNode
from flagquantum.runtime.execution import run as run_internal

pytestmark = pytest.mark.unit


def _bell() -> fq.Circuit:
    """A four-wire program with correlations across the requested wires."""

    return fq.Circuit(4).x(0).h(1).ry(2, 0.7).cx(1, 3)


def _shipped(circuit: fq.Circuit, wires: tuple[int, ...]) -> torch.Tensor:
    return (
        fq.run(
            circuit,
            options=fq.ExecutionOptions(mode="statevector"),
            outputs=fq.probabilities(wires),
        )
        .measurements[0]
        .value
    )


def _parity(circuit: fq.Circuit, wires: tuple[int, ...]) -> torch.Tensor:
    """The reconstruction the reduction replaced, reached through its own door."""

    target = measurements._statevector_target(circuit.state(), circuit.n_wires)
    return measurements._marginal_probabilities(target, wires)


def _dense(circuit: fq.Circuit, wires: tuple[int, ...]) -> torch.Tensor | None:
    """The reduction, driven by the dense state rather than by a target."""

    return measurements._joint_marginal_probabilities(
        circuit.state(),
        wires,
        n_wires=circuit.n_wires,
        noise_model=None,
    )


class _CountingTarget:
    """A target whose expectations are counted, to prove they are not consulted."""

    def __init__(self, n_wires: int) -> None:
        self.n_wires = n_wires
        self.expectation_calls: list[dict[str, tuple[int, ...]]] = []

    def expectation_ps(self, **axes: tuple[int, ...]) -> torch.Tensor:
        self.expectation_calls.append(axes)
        raise AssertionError("the parity path must not run for this case")

    def sample(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("sampling is not part of this case")


@pytest.mark.parametrize(
    "wires",
    [(0,), (1, 2), (0, 1), (2, 0), (3, 1, 0), (0, 1, 2, 3)],
)
def test_the_reduction_agrees_with_the_parity_reconstruction(
    wires: tuple[int, ...],
) -> None:
    circuit = _bell()
    reduced = _shipped(circuit, wires)
    reconstructed = _parity(circuit, wires)

    assert reduced.shape == (1, 2 ** len(wires))
    torch.testing.assert_close(reduced, reconstructed, atol=1e-6, rtol=0)
    assert float(reduced.sum()) == pytest.approx(1.0, abs=1e-6)


def test_a_full_width_request_is_the_distribution_itself() -> None:
    circuit = _bell()
    every_wire = _shipped(circuit, (0, 1, 2, 3))

    torch.testing.assert_close(every_wire, torch.abs(circuit.state()) ** 2)


@pytest.mark.parametrize("mode", ("statevector", "density_matrix"))
def test_both_dense_results_reduce_to_the_same_marginal(mode: str) -> None:
    """A statevector and a density matrix carry the same distribution."""

    circuit = _bell()
    value = (
        fq.run(
            circuit,
            options=fq.ExecutionOptions(mode=mode),
            outputs=fq.probabilities((2, 0)),
        )
        .measurements[0]
        .value
    )

    torch.testing.assert_close(
        value,
        _shipped(circuit, (2, 0)),
        atol=1e-6,
        rtol=0,
    )


def test_a_noisy_density_matrix_marginal_applies_readout_confusion() -> None:
    """The readout step stays in the shared path, so both arms keep it."""

    import flagquantum.noise as fqn

    noise = fqn.NoiseModel().add_readout(
        0,
        fqn.ReadoutError(((0.75, 0.25), (0.25, 0.75))),
    )
    circuit = fq.Circuit(2).x(0).h(1)
    options = fq.ExecutionOptions(mode="density_matrix")
    observed = (
        fq.run(
            circuit,
            options=options,
            noise_model=noise,
            outputs=fq.probabilities((0, 1)),
        )
        .measurements[0]
        .value
    )
    ideal = (
        fq.run(
            circuit,
            options=options,
            outputs=fq.probabilities((0, 1)),
        )
        .measurements[0]
        .value
    )

    torch.testing.assert_close(
        ideal,
        torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        observed,
        torch.tensor([[0.125, 0.125, 0.375, 0.375]]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(observed.sum(dim=-1), torch.ones(1), atol=1e-6, rtol=0)


def test_the_marginal_is_taken_from_the_dense_result_and_not_from_contractions() -> (
    None
):
    """Both paths agree, so the routing is observed rather than inferred.

    A test that only compared numbers would keep passing if the reduction were
    never reached, which is exactly the regression this change could suffer.
    """

    circuit = _bell()
    state = circuit.state()
    target = _CountingTarget(circuit.n_wires)

    reduced = measurements._joint_marginal_probabilities(
        state,
        (1,),
        n_wires=circuit.n_wires,
        noise_model=None,
    )
    assert reduced is not None
    torch.testing.assert_close(reduced, _parity(circuit, (1,)), atol=1e-6, rtol=0)
    assert target.expectation_calls == []

    # A result that is not dense takes the other branch, which does consult the
    # target -- so the absence of calls above is about the routing and not about
    # the counter.
    assert (
        measurements._joint_marginal_probabilities(
            object(), (1,), n_wires=circuit.n_wires, noise_model=None
        )
        is None
    )


def test_a_result_that_is_not_a_dense_distribution_is_not_reduced() -> None:
    """Reducing the wrong tensor would return a plausible marginal of nothing.

    A two-wire result has four basis states, so ``(batch, 4)`` complex is a
    statevector and ``(batch, 4, 4)`` complex is a density matrix. Everything
    else -- a real tensor, a wrong width, a square matrix of the wrong width, or
    something that is not a tensor at all -- must be declined instead of
    reshaped into whatever shape happened to fit.
    """

    declined = (
        torch.ones(1, 4),
        torch.ones(1, 4, dtype=torch.complex64).reshape(1, 2, 2),
        torch.ones(1, 4, 4, 4, dtype=torch.complex64),
        torch.ones(1, 3, dtype=torch.complex64),
        torch.ones(1, 4, 8, dtype=torch.complex64),
        [[0.5, 0.5]],
        object(),
    )
    for candidate in declined:
        assert (
            measurements._ideal_probabilities(candidate, 2) is None
        ), f"{type(candidate).__name__} must not be read as a two-wire distribution"

    accepted = torch.ones(1, 4, dtype=torch.complex64)
    assert measurements._ideal_probabilities(accepted, 2) is not None


def test_the_reduction_normalises_a_dense_state_that_was_never_normalised() -> None:
    """A marginal of a non-distribution is not a distribution.

    The parity arm normalised as a side effect of ``expectation_ps``; the
    reduction reads amplitudes directly, so it has to say what it does about a
    state that was handed in unnormalised. It divides the total out, which is
    what keeps the old statevector contract intact. Nothing the engine produces
    reaches this branch, and a normalised state is left bit-for-bit alone.
    """

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    unnormalised = circuit.state() * 3.0
    target = measurements._statevector_target(unnormalised, 2)

    reduction = measurements._joint_marginal_probabilities(
        unnormalised, (0,), n_wires=2, noise_model=None
    )
    assert reduction is not None
    torch.testing.assert_close(
        reduction,
        measurements._marginal_probabilities(target, (0,)),
        atol=1e-6,
        rtol=0,
    )
    assert float(reduction.sum()) == pytest.approx(1.0, abs=1e-6)

    normalised = circuit.state()
    untouched = measurements._joint_marginal_probabilities(
        normalised, (0, 1), n_wires=2, noise_model=None
    )
    torch.testing.assert_close(untouched, torch.abs(normalised) ** 2, rtol=0, atol=0)


def test_an_all_zero_dense_state_is_returned_rather_than_divided() -> None:
    """A zero state has no total to divide by, and ``nan`` would hide that."""

    zero = torch.zeros(1, 4, dtype=torch.complex64)
    marginal = measurements._joint_marginal_probabilities(
        zero, (0,), n_wires=2, noise_model=None
    )

    torch.testing.assert_close(marginal, torch.zeros(1, 2), rtol=0, atol=0)
    assert not bool(torch.isnan(marginal).any())


def test_an_unsorted_request_keeps_the_order_it_asked_for() -> None:
    """A transposed marginal is a wrong answer that no shape check would catch."""

    circuit = _bell()
    requested = _shipped(circuit, (2, 0))
    ascending = _shipped(circuit, (0, 2))

    torch.testing.assert_close(
        requested,
        ascending.reshape(2, 2).transpose(0, 1).reshape(1, 4),
        atol=1e-6,
        rtol=0,
    )
    assert not torch.allclose(requested, ascending)


def test_an_all_z_pauli_sample_reads_the_dense_distribution() -> None:
    """A Z-basis readout is the computational-basis marginal, so it can be reduced."""

    circuit = _bell()
    node = MeasurementNode(
        "sample_ps",
        (0, 1, 2),
        shots=4096,
        metadata={"z": (0, 1, 2), "seed": 5},
    )
    result = run_internal(
        circuit,
        options=fq.ExecutionOptions(mode="statevector"),
        measurements=(node,),
    )

    samples = result.measurements[0].value[0]
    exact = _shipped(circuit, (0, 1, 2))[0]
    observed = torch.zeros(8)
    for row in samples:
        index = int(row[0]) * 4 + int(row[1]) * 2 + int(row[2])
        observed[index] += 1
    observed = observed / observed.sum()

    # Every outcome the state forbids must be absent, not merely unlikely: the
    # parity reconstruction can return a small negative entry for those, and a
    # negative probability makes multinomial raise rather than sample.
    for index, probability in enumerate(exact):
        if float(probability) == 0.0:
            assert float(observed[index]) == 0.0
        else:
            assert float(observed[index]) == pytest.approx(float(probability), abs=0.03)


def test_a_mixed_basis_pauli_sample_keeps_the_parity_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading any wire in X or Y is a different measurement, not a basis change."""

    circuit = _bell()
    calls: list[int] = []
    original = measurements._probabilities_from_parity_expectations

    def recording(subset_expectations: object, *, n_wires: int) -> torch.Tensor:
        calls.append(n_wires)
        return original(subset_expectations, n_wires=n_wires)

    monkeypatch.setattr(
        measurements, "_probabilities_from_parity_expectations", recording
    )
    node = MeasurementNode(
        "sample_ps",
        (0, 1),
        shots=4,
        metadata={"z": (0,), "x": (1,), "seed": 5},
    )
    run_internal(
        circuit,
        options=fq.ExecutionOptions(mode="statevector"),
        measurements=(node,),
    )

    assert calls == [2]


def test_an_mps_marginal_keeps_the_parity_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MPS result is not dense, so it must not be reduced.

    This is the boundary the change has to respect rather than a preference: for
    an MPS target the distribution is reached by materialising a dense state, so
    reducing it would trade a handful of contractions for something that grows
    without bound. The assertion is on the routing, and the timing is recorded
    only to keep the reason visible.
    """

    circuit = fq.Circuit(4).x(0).h(1).ry(2, 0.7).cx(1, 3)
    observed: list[object] = []
    original = measurements._joint_marginal_probabilities

    def recording(output: object, wires: tuple[int, ...], **kwargs: object):
        observed.append(output)
        return original(output, wires, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(measurements, "_joint_marginal_probabilities", recording)
    value = (
        fq.run(
            circuit,
            options=fq.ExecutionOptions(mode="mps"),
            outputs=fq.probabilities((1, 2)),
        )
        .measurements[0]
        .value
    )

    assert len(observed) == 1
    assert not isinstance(observed[0], torch.Tensor)
    torch.testing.assert_close(value, _shipped(circuit, (1, 2)), atol=1e-6, rtol=0)


def test_the_two_arms_differ_by_orders_of_magnitude_on_a_wide_dense_state() -> None:
    """The change is a cost change, so it is measured rather than argued.

    Sixteen wires keeps the runtime bounded; the exponent is what decides the
    gap, and the timing is a bound assertion with a factor of five of slack so
    that a loaded host cannot fail the suite.
    """

    circuit = fq.Circuit(16)
    for wire in range(16):
        circuit.h(wire)
    for wire in range(15):
        circuit.cx(wire, wire + 1)

    state = circuit.state()
    target = measurements._statevector_target(state, 16)

    started = time.perf_counter()
    reduction = measurements._joint_marginal_probabilities(
        state, (0, 1, 2, 3), n_wires=16, noise_model=None
    )
    reduction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reconstructed = measurements._marginal_probabilities(target, (0, 1, 2, 3))
    parity_seconds = time.perf_counter() - started

    assert reduction is not None
    torch.testing.assert_close(reduction, reconstructed, atol=1e-5, rtol=0)
    assert reduction_seconds * 5 < parity_seconds


def test_the_retained_width_limit_still_refuses_a_wider_marginal() -> None:
    """The reduction is cheap; the limit is policy, and it is unchanged here."""

    circuit = fq.Circuit(9)

    with pytest.raises(ValueError, match="increase max_marginal_wires"):
        run_internal(
            circuit,
            options=fq.ExecutionOptions(mode="statevector"),
            measurements=(MeasurementNode("probabilities", tuple(range(9))),),
        )
    with pytest.raises(ValueError, match="exceed the supported limit of 8"):
        measurements.validate_measurements(
            (MeasurementNode("probabilities", tuple(range(9))),),
            n_wires=9,
        )
