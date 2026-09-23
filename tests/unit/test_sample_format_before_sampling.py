"""The ``sample`` format contract shared by circuits, MPS, and TN states.

All three entry points accept ``format="bits"`` and ``format="index"`` and refuse
anything else with ``sample format must be 'bits' or 'index'.`` The refusal has to
happen before sampling: the format is a property of the request, not of the
samples, so checking it afterwards spends the whole shot budget and then refuses a
request that was always going to be refused.

``Circuit.sample`` checked first from the start. The MPS and tensor-network states
sampled first and checked after, so the cost of the refusal grew with the shot
budget — the tensor-network path built its full probability vector before
refusing. The tests below pin the earlier check by replacing the one call each
entry point makes to produce samples, so a regression cannot pass by moving the
check back after it.

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
