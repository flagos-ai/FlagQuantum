from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.executors.mps.forward as forward
from flagquantum.runtime.executors.mps import compiled_layers

pytestmark = pytest.mark.unit


def test_forward_validates_policy_before_tensor_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_allocation(*args, **kwargs):
        raise AssertionError("validation must precede tensor allocation")

    monkeypatch.setattr(forward.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(forward.torch, "zeros", reject_allocation)

    with pytest.raises(ValueError, match="global_error_budget"):
        forward.execute_torch_distributed_mps_forward(
            fq.Circuit(1), global_error_budget=-1.0
        )


@pytest.mark.parametrize("two_site", [False, True])
def test_forward_rejects_cache_entry_for_wrong_gate_width(
    monkeypatch: pytest.MonkeyPatch, two_site: bool
) -> None:
    circuit = fq.Circuit(2)
    if two_site:
        circuit.rxx(0, 1, theta=0.2)
        malformed_output = (torch.ones(1),)
    else:
        circuit.ry(0, theta=0.2)
        malformed_output = (torch.ones(1), torch.ones(1), {})
    ir = circuit.to_ir()
    layer = ((0, ir.instructions[0], ir.instructions[0].wires),)
    monkeypatch.setattr(forward.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(forward.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(forward.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(forward.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        forward,
        "_prepare_compiled_layer",
        lambda *args, **kwargs: (layer, {0: malformed_output}, ()),
    )
    with pytest.raises(
        forward.MPSForwardLifetimeError, match="compiled MPS instruction"
    ):
        forward.execute_torch_distributed_mps_forward(
            ir, compile_site_kernels=True, rebalance_threshold=float("inf")
        )


def test_layer_cache_drain_invariant_rejects_retained_tensor() -> None:
    compiled_layers.require_layer_cache_drained(
        {}, layer_sequence=0, layer_start=2, layer_end=5
    )
    with pytest.raises(
        forward.MPSForwardLifetimeError,
        match=r"layer=3.*instructions=10:12.*retained=\(11,\)",
    ):
        compiled_layers.require_layer_cache_drained(
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
    fields = compiled_layers.device_memory_metadata(torch.device("cpu"))
    assert fields == {
        "allocated_memory_bytes": None,
        "reserved_memory_bytes": None,
        "peak_allocated_memory_bytes": None,
    }
    assert not any(isinstance(value, torch.Tensor) for value in fields.values())


def test_cuda_memory_fields_use_platform_snapshot(monkeypatch) -> None:
    memory = SimpleNamespace(allocated_bytes=10, reserved_bytes=20)
    platform = SimpleNamespace(memory_snapshot=lambda device: memory)
    monkeypatch.setattr(compiled_layers, "get_platform_runtime", lambda kind: platform)
    monkeypatch.setattr(
        compiled_layers.torch.cuda, "max_memory_allocated", lambda device: 30
    )

    assert compiled_layers.device_memory_metadata(torch.device("cuda:0")) == {
        "allocated_memory_bytes": 10,
        "reserved_memory_bytes": 20,
        "peak_allocated_memory_bytes": 30,
    }
