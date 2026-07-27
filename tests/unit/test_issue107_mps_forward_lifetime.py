from __future__ import annotations

import inspect

import pytest
import torch

import flagquantum.runtime.backends.mps.forward as forward

pytestmark = pytest.mark.unit


def test_layer_cache_drain_invariant_rejects_retained_tensor() -> None:
    forward._require_layer_cache_drained(
        {}, layer_sequence=0, layer_start=2, layer_end=5
    )
    with pytest.raises(
        forward.MPSForwardLifetimeError,
        match=r"layer=3.*instructions=10:12.*retained=\(11,\)",
    ):
        forward._require_layer_cache_drained(
            {11: (torch.ones(1),)},
            layer_sequence=3,
            layer_start=10,
            layer_end=12,
        )


def test_forward_does_not_eagerly_materialize_all_gate_matrices() -> None:
    source = inspect.getsource(forward.execute_torch_distributed_mps_forward)
    assert "matrices.append" not in source
    assert "zip(ir.instructions, matrices)" not in source
    assert 'gate_matrix_materialization="per_instruction_last_use"' in source


def test_cuda_memory_fields_are_json_metadata_on_cpu() -> None:
    fields = forward._cuda_memory_fields(torch.device("cpu"))
    assert fields == {
        "allocated_memory_bytes": None,
        "reserved_memory_bytes": None,
        "peak_allocated_memory_bytes": None,
    }
    assert not any(isinstance(value, torch.Tensor) for value in fields.values())
