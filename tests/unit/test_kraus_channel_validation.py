"""A Kraus channel must contain at least one operator, on every entry path.

`KrausChannel.__post_init__` already refuses an empty operator tuple. The module
function accepts a raw sequence as well, and on that path it bypassed the check:
`out` starts as `torch.zeros_like(rho)`, so an empty channel returned the zero
map -- a trace-0 object that is not a state and that silently zeroed every later
expectation computed from it.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch

import flagquantum.simulation.density_matrix as density_matrix_module
from flagquantum.noise import KrausChannel

pytestmark = pytest.mark.unit

N_WIRES = 2
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
EMPTY_CHANNELS: tuple[tuple[str, object], ...] = (
    ("empty-list", []),
    ("empty-tuple", ()),
    ("exhausted-iterator", iter(())),
)


def _ground_state() -> torch.Tensor:
    rho = torch.zeros(2**N_WIRES, 2**N_WIRES, dtype=torch.complex64)
    rho[0, 0] = 1
    return rho


@pytest.mark.parametrize(
    ("channel", "kraus"),
    EMPTY_CHANNELS,
    ids=[name for name, _ in EMPTY_CHANNELS],
)
def test_an_empty_channel_is_refused(
    channel: str, kraus: Iterable[torch.Tensor]
) -> None:
    with pytest.raises(ValueError, match="must contain at least one operator"):
        density_matrix_module.apply_kraus_density(_ground_state(), kraus, [0], N_WIRES)


def test_the_dataclass_and_the_function_agree_on_an_empty_channel() -> None:
    """Both entry paths refuse, and say the same thing.

    The dataclass was already fail-closed; only the raw-sequence path was not, so
    which of the two a caller happened to use decided whether a malformed channel
    was reported or silently turned into the zero map.
    """

    with pytest.raises(ValueError, match="must contain at least one operator"):
        KrausChannel(name="empty", kraus=())

    with pytest.raises(ValueError, match="must contain at least one operator"):
        density_matrix_module.apply_kraus_density(_ground_state(), [], [0], N_WIRES)


def test_an_empty_channel_no_longer_yields_the_zero_map() -> None:
    """The refused call must not leave behind a silent all-zero result."""

    zero_map = torch.zeros_like(_ground_state()).reshape(1, 2**N_WIRES, 2**N_WIRES)
    assert zero_map[0].trace().real.item() == 0.0

    with pytest.raises(ValueError):
        density_matrix_module.apply_kraus_density(_ground_state(), [], [0], N_WIRES)


def test_a_valid_channel_still_applies() -> None:
    """The guard must not reject a channel that does have operators."""

    for kraus in ([X], (X,), [X, Z]):
        applied = density_matrix_module.apply_kraus_density(
            _ground_state(), kraus, [0], N_WIRES
        )
        assert applied.shape == (1, 2**N_WIRES, 2**N_WIRES)

    # {X, Z} / sqrt(2) is trace preserving on one wire, so the trace must be 1.
    # Every accepted form of the sequence is exercised above; this checks one of
    # them end to end against the property the channel is defined by.
    normalized = [X / 2**0.5, Z / 2**0.5]
    pair = density_matrix_module.apply_kraus_density(
        _ground_state(), normalized, [0], N_WIRES
    )
    assert torch.allclose(pair[0].trace().real, torch.tensor(1.0), atol=1e-6)

    from_dataclass = density_matrix_module.apply_kraus_density(
        _ground_state(), KrausChannel(name="x", kraus=(X,)), [0], N_WIRES
    )
    assert torch.allclose(from_dataclass[0].trace().real, torch.tensor(1.0), atol=1e-6)
