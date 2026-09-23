"""JAX executor results refuse an observable wire outside the executed state."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from flagquantum.runtime.executors.jax.statevector import records as statevector_records
from flagquantum.runtime.executors.jax.tensor_network import records as tn_records

pytestmark = pytest.mark.unit

# ``|100>`` on three wires: the true Z expectations are ``-1, +1, +1``, so a
# wire that reports the wrong basis bit cannot be mistaken for a correct answer.
_BASIS_INDEX = 0b100
_REFERENCE = (-1.0, 1.0, 1.0)


def _state(*, complex_bytes: int) -> torch.Tensor:
    dtype = torch.complex64 if complex_bytes == 8 else torch.complex128
    state = torch.zeros((1, 8), dtype=dtype)
    state[0, _BASIS_INDEX] = 1.0
    return state


def _statevector_result(
    monkeypatch: pytest.MonkeyPatch, *, complex_bytes: int
) -> statevector_records.JAXShardedStatevectorResult:
    """Build a real executor result whose shard reconstruction is deterministic."""

    state = _state(complex_bytes=complex_bytes)
    monkeypatch.setattr(
        statevector_records,
        "_reconstruct_torch_state_from_jax_shards",
        lambda shards, plan: state,
        raising=True,
    )
    plan = cast(
        Any,
        type(
            "StatevectorPlan",
            (),
            {
                "n_wires": 3,
                "bsz": 1,
                "total_amplitudes": 8,
                "complex_bytes": complex_bytes,
            },
        )(),
    )
    return statevector_records.JAXShardedStatevectorResult(
        shards=(),
        plan=plan,
        jax_plan=cast(Any, None),
        backend_policy=cast(Any, None),
        local_gate_count=0,
        distributed_gate_count=0,
        simulated_communication_count=0,
        simulated_communication_bytes=0,
    )


def _tensor_network_result(
    monkeypatch: pytest.MonkeyPatch, *, complex_bytes: int
) -> tn_records.JAXShardedTensorNetworkResult:
    """Build a real tensor-network result over the same three-wire state."""

    state = _state(complex_bytes=complex_bytes)
    monkeypatch.setattr(
        tn_records,
        "_jax_reduced_tn_output_to_torch_state",
        lambda reduced_output, *, n_wires, bsz, complex_bytes: state,
        raising=True,
    )
    return tn_records.JAXShardedTensorNetworkResult(
        rank_partials=(),
        reduced_output=object(),
        slicing=None,
        jax_plan=cast(Any, None),
        backend_policy=cast(Any, None),
        n_wires=3,
        bsz=1,
        complex_bytes=complex_bytes,
        local_world_size=1,
        node_count=1,
    )


_RESULTS = ("statevector", "tensor_network")


def _result(monkeypatch: pytest.MonkeyPatch, kind: str, complex_bytes: int) -> Any:
    if kind == "statevector":
        return _statevector_result(monkeypatch, complex_bytes=complex_bytes)
    return _tensor_network_result(monkeypatch, complex_bytes=complex_bytes)


@pytest.mark.parametrize("kind", _RESULTS)
@pytest.mark.parametrize("complex_bytes", (8, 16))
def test_in_range_wires_keep_their_values(
    monkeypatch: pytest.MonkeyPatch, kind: str, complex_bytes: int
) -> None:
    result = _result(monkeypatch, kind, complex_bytes)
    expected_dtype = torch.float32 if complex_bytes == 8 else torch.float64

    for wire, expected in enumerate(_REFERENCE):
        value = result.expectation_z(wire)
        assert value.shape == (1, 1)
        assert value.dtype == expected_dtype
        assert torch.allclose(
            value.to(torch.float64), torch.tensor([[expected]], dtype=torch.float64)
        )

    assert torch.allclose(
        result.expectation_z().to(torch.float64).reshape(-1),
        torch.tensor(_REFERENCE, dtype=torch.float64),
    )


@pytest.mark.parametrize("kind", _RESULTS)
def test_negative_wire_is_refused_instead_of_reporting_plus_one(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A negative wire used to select bit 0 of the basis index and return +1.0."""

    result = _result(monkeypatch, kind, complex_bytes=16)

    for wire in (-1, -2, -3, -4):
        with pytest.raises(ValueError, match="observable wire index out of range"):
            result.expectation_z(wire)


@pytest.mark.parametrize("kind", _RESULTS)
def test_wire_at_or_beyond_the_state_is_refused(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    result = _result(monkeypatch, kind, complex_bytes=16)

    for wire in (3, 4, 9):
        with pytest.raises(ValueError, match="observable wire index out of range"):
            result.expectation_z(wire)


@pytest.mark.parametrize("kind", _RESULTS)
def test_refused_wire_does_not_build_the_state_facade(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The request fails before the reconstruction facade is materialized."""

    result = _result(monkeypatch, kind, complex_bytes=16)

    with pytest.raises(ValueError, match="observable wire index out of range"):
        result.expectation_z(-1)

    facade_count = (
        result.full_state_reconstruction_count
        if kind == "statevector"
        else result.final_state_facade_count
    )
    assert facade_count == 0
