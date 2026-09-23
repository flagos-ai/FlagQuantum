"""Out-of-range observable wires must be refused by every representation.

`MPSState` validated observable wire indices; `TensorNetworkState` did not.
Because a wire outside `range(n_wires)` addresses no open leg, the contraction
still returned a finite number, and the dense sign-weight path reinterpreted a
negative bit shift, so a bogus wire produced a plausible expectation instead of
an error. These tests hold both representations to the same refusal.
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
OUT_OF_RANGE = (3, 9, -1, -2)


def _mps_state() -> State:
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1)
    return fqmps.run_mps(circuit)


def _tensor_network_state() -> State:
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1)
    return fqtn.run_tensor_network(circuit)


STATES: tuple[tuple[str, Callable[[], State]], ...] = (
    ("mps", _mps_state),
    ("tensor_network", _tensor_network_state),
)


@pytest.fixture(params=STATES, ids=[name for name, _ in STATES])
def state(request: pytest.FixtureRequest) -> State:
    return request.param[1]()


@pytest.mark.parametrize("wire", OUT_OF_RANGE)
def test_expectation_z_refuses_a_wire_outside_the_state(
    state: State, wire: int
) -> None:
    with pytest.raises(ValueError, match="observable wire index out of range"):
        state.expectation_z(wires=[wire])  # type: ignore[attr-defined]


@pytest.mark.parametrize("wire", OUT_OF_RANGE)
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_expectation_ps_refuses_a_wire_outside_the_state(
    state: State, wire: int, axis: str
) -> None:
    with pytest.raises(ValueError, match="observable wire index out of range"):
        state.expectation_ps(**{axis: [wire]})  # type: ignore[attr-defined]


def test_a_refused_wire_does_not_return_a_plausible_expectation(state: State) -> None:
    """The pre-change tensor-network path answered `wire=9` with a real number.

    A returned expectation must never be silently attributed to a wire the state
    does not have, so this asserts the refusal rather than inspecting a value.
    """
    for wire in OUT_OF_RANGE:
        with pytest.raises(ValueError):
            state.expectation_z(wires=[wire])  # type: ignore[attr-defined]


def test_in_range_observable_wires_still_evaluate(state: State) -> None:
    """The guard must not reject a wire the state does have."""

    z = state.expectation_z(wires=[0, 1, 2])  # type: ignore[attr-defined]
    assert z.shape[-1] == 3

    single = state.expectation_z(wires=[2])  # type: ignore[attr-defined]
    assert torch.allclose(single.reshape(-1), z.reshape(-1)[-1:], atol=1e-6)

    for axis in ("x", "y", "z"):
        values = state.expectation_ps(**{axis: [0]})  # type: ignore[attr-defined]
        assert values.numel() == 1
