"""The ``sample`` request contract shared by circuits, MPS, and TN states.

Two request properties are checked here, both before any sampling happens.

**The shot budget.** ``shots`` must be a positive integer, which is the repository
convention for this option (``_api.py``, ``runtime/dynamic/execution.py``,
``runtime/dynamic/deployment.py``, ``runtime/dynamic/hybrid_session.py``,
``algorithms/svd.py``). ``MPSState`` was the outlier: ``sample(0)`` returned a
zero-width tensor and ``counts(0)`` returned ``[{}]`` — a single empty histogram —
where the circuit and tensor-network states refuse the same request, while a float
or a string reached the tensor allocation as a raw torch error. Reading ``[{}]`` as
"no data" rather than as a refused request is exactly the silent degradation the
repository's fail-closed principle forbids.

The circuit and tensor-network states already refuse a zero budget, so this file
only pins MPS for it. Their ``RuntimeError`` is deliberately left alone: changing
the exception type raised by ``Circuit`` would alter documented behavior of a
Stable Core API, which this change does not authorize.

**The format.** All three entry points accept ``format="bits"`` and
``format="index"`` and refuse anything else with
``sample format must be 'bits' or 'index'.`` That refusal also has to happen
before sampling: the format is a property of the request, not of the samples, so
checking it afterwards spends the whole shot budget and then refuses a request that
was always going to be refused. ``Circuit.sample`` checked first from the start;
the MPS and tensor-network states sampled first and checked after, and the
tensor-network path built its full probability vector before refusing. The tests
below pin the earlier check by replacing the one call each entry point makes to
produce samples, so a regression cannot pass by moving the check back after it.

The exception type and message are asserted in full: both states raised
``ValueError`` with this exact text before the change, and a reworded message
would be a user-visible behavior change.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.mps as fqmps
import flagquantum.simulation.tensor_network as fqtn

pytestmark = pytest.mark.unit

State = object

MESSAGE = "sample format must be 'bits' or 'index'"


def _circuit() -> State:
    circuit = fq.Circuit(3)
    circuit.x(0).x(2)
    return circuit


def _mps_state() -> State:
    circuit = fq.Circuit(3)
    circuit.x(0).x(2)
    return fqmps.run_mps(circuit)


def _tensor_network_state() -> State:
    circuit = fq.Circuit(3)
    circuit.x(0).x(2)
    return fqtn.run_tensor_network(circuit)


STATES: tuple[tuple[str, Callable[[], State]], ...] = (
    ("circuit", _circuit),
    ("mps", _mps_state),
    ("tensor_network", _tensor_network_state),
)


@pytest.fixture(params=STATES, ids=[name for name, _ in STATES])
def state(request: pytest.FixtureRequest) -> State:
    return request.param[1]()


def _refuse_sampling() -> Callable[..., torch.Tensor]:
    def unexpected_sample(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("sample must refuse an unsupported format first")

    return unexpected_sample


def test_sample_rejects_an_unsupported_format(state: State) -> None:
    with pytest.raises(ValueError, match=MESSAGE):
        state.sample(4, format="hex")  # type: ignore[attr-defined]


def test_sample_keeps_the_documented_result_for_supported_formats(
    state: State,
) -> None:
    """Both formats stay supported, and the values stay consistent."""

    bits = state.sample(  # type: ignore[attr-defined]
        8, generator=torch.Generator().manual_seed(1234), format="bits"
    )
    index = state.sample(  # type: ignore[attr-defined]
        8, generator=torch.Generator().manual_seed(1234), format="index"
    )
    assert bits.shape == (1, 8, 3)
    assert index.shape == (1, 8)
    # Wire 0 and wire 2 are set, so the only sampled index is 0b101 = 5.
    assert index.tolist() == [[5] * 8]
    assert bits.tolist() == [[[1, 0, 1]] * 8]


def test_circuit_sample_rejects_an_unsupported_format_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flagquantum.simulation.statevector.local._sample_statevector",
        _refuse_sampling(),
    )
    with pytest.raises(ValueError, match=MESSAGE):
        fq.Circuit(2).sample(4, format="hex")


def test_mps_sample_rejects_an_unsupported_format_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _mps_state()
    monkeypatch.setattr(state, "_sample_indices", _refuse_sampling())
    with pytest.raises(ValueError, match=MESSAGE):
        state.sample(4, format="hex")  # type: ignore[attr-defined]


def test_tensor_network_sample_rejects_an_unsupported_format_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _tensor_network_state()
    monkeypatch.setattr(state, "probabilities", _refuse_sampling())
    with pytest.raises(ValueError, match=MESSAGE):
        state.sample(4, format="hex")  # type: ignore[attr-defined]


SHOTS_MESSAGE = "sample shots must be a positive integer"


@pytest.mark.parametrize("shots", [0, -1, -1000])
def test_mps_sample_rejects_a_non_positive_shot_budget(shots: int) -> None:
    """A zero budget is a refused request, not an empty sample tensor."""

    with pytest.raises(ValueError, match=SHOTS_MESSAGE):
        _mps_state().sample(shots)


@pytest.mark.parametrize("shots", [0, -1, -1000])
def test_mps_counts_rejects_a_non_positive_shot_budget(shots: int) -> None:
    """``counts`` must not report an empty histogram for a refused budget."""

    with pytest.raises(ValueError, match=SHOTS_MESSAGE):
        _mps_state().counts(shots)


@pytest.mark.parametrize("shots", [2.7, "4", None, True])
def test_mps_sample_rejects_a_non_integer_shot_budget(shots: object) -> None:
    """A non-integer budget is refused here, not inside a tensor allocation."""

    with pytest.raises(ValueError, match=SHOTS_MESSAGE):
        _mps_state().sample(shots)  # type: ignore[arg-type]


def test_mps_sample_still_accepts_a_valid_budget() -> None:
    """The guard must not narrow the supported requests."""

    state = _mps_state()
    assert state.sample(4, generator=torch.Generator().manual_seed(7)).shape == (
        1,
        4,
        3,
    )
    assert (
        sum(state.counts(4, generator=torch.Generator().manual_seed(7))[0].values())
        == 4
    )


def test_mps_counts_checks_the_format_before_the_shot_budget() -> None:
    """Both are request properties; the format refusal stays the one users see."""

    with pytest.raises(ValueError, match="counts format must be 'bin' or 'int'"):
        _mps_state().counts(0, format="hex")
