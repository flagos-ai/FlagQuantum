"""The ``counts`` histogram contract shared by the MPS and TN states.

Both states accept ``format="bin"`` and ``format="int"`` and refuse anything
else. The refusal has to happen before sampling: the reference implementation
validated the format while formatting each sampled row, so a request that
sampled no rows returned an empty histogram instead of an error.
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


def _mps_state() -> State:
    circuit = fq.Circuit(3)
    circuit.x(0).x(2)
    return fqmps.run_mps(circuit)


def _tensor_network_state() -> State:
    circuit = fq.Circuit(3)
    circuit.x(0).x(2)
    return fqtn.run_tensor_network(circuit)


STATES: tuple[tuple[str, Callable[[], State]], ...] = (
    ("mps", _mps_state),
    ("tensor_network", _tensor_network_state),
)


@pytest.fixture(params=STATES, ids=[name for name, _ in STATES])
def state(request: pytest.FixtureRequest) -> State:
    return request.param[1]()


def test_counts_rejects_an_unsupported_format_with_shots(state: State) -> None:
    with pytest.raises(ValueError, match="counts format must be 'bin' or 'int'"):
        state.counts(4, format="hex")  # type: ignore[attr-defined]


def test_counts_rejects_an_unsupported_format_before_sampling(state: State) -> None:
    """An empty shot budget must not turn a bad format into an empty histogram."""

    def unexpected_sample(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("counts must refuse an unsupported format first")

    state.sample = unexpected_sample  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="counts format must be 'bin' or 'int'"):
        state.counts(0, format="hex")  # type: ignore[attr-defined]


def test_counts_keeps_the_documented_histogram_for_supported_formats(
    state: State,
) -> None:
    generator = torch.Generator().manual_seed(1234)

    assert state.counts(8, generator=generator, format="bin") == [  # type: ignore[attr-defined]
        {"101": 8}
    ]
    assert state.counts(  # type: ignore[attr-defined]
        8, generator=torch.Generator().manual_seed(1234), format="int"
    ) == [{5: 8}]
