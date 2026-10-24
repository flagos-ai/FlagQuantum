"""CPU control-flow reference for the CUDA benchmark's warmup boundary."""

from __future__ import annotations

import itertools

import pytest
import torch

from benchmarks import continuous_performance as benchmark

pytestmark = [pytest.mark.unit, pytest.mark.benchmark_contract]


@pytest.mark.parametrize("warmup", [0, 2])
def test_warmup_matches_sample_work_and_is_excluded_from_totals(
    monkeypatch: pytest.MonkeyPatch, warmup: int
) -> None:
    # CPU tensors and one identity collective isolate the scheduling boundary;
    # this test does not validate CUDA/NCCL numerics or hardware performance.
    events: list[str] = []
    full, zeros = torch.full, torch.zeros

    def cpu_full(
        size: tuple[int, ...],
        value: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return full(size, value, dtype=dtype)

    def cpu_zeros(
        size: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return zeros(size, dtype=dtype)

    def synchronize(device: torch.device) -> None:
        events.append("synchronize")

    def all_reduce(tensor: torch.Tensor) -> None:
        events.append("collective")

    def gather(
        output: list[dict[str, object] | None], local: dict[str, object]
    ) -> None:
        output[0] = local

    def reset_peak(device: torch.device) -> None:
        events.append("reset_peak")

    def allocated(device: torch.device) -> int:
        events.append("sample")
        return 64

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(benchmark.torch, "full", cpu_full)
    monkeypatch.setattr(benchmark.torch, "zeros", cpu_zeros)
    monkeypatch.setattr(benchmark.dist, "init_process_group", lambda backend: None)
    monkeypatch.setattr(benchmark.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(benchmark.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(benchmark.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(benchmark.dist, "all_gather_object", gather)
    monkeypatch.setattr(benchmark.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(benchmark.torch.cuda, "reset_peak_memory_stats", reset_peak)
    monkeypatch.setattr(benchmark.torch.cuda, "memory_allocated", allocated)
    monkeypatch.setattr(benchmark.torch.cuda, "memory_reserved", lambda device: 128)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_allocated", lambda device: 64)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_reserved", lambda device: 128)
    clock = itertools.count()
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: float(next(clock)))

    repetitions, inner_loops = 3, 4
    record, _ = benchmark._measure_distributed(warmup, repetitions, inner_loops)

    block = ["synchronize", "collective", "synchronize"] * inner_loops + ["synchronize"]
    assert events == (
        block * warmup
        + ["synchronize", "reset_peak"]
        + (block + ["sample"]) * repetitions
    )
    assert record.warmup == warmup
    assert record.initialization_seconds == 1 + 2 * warmup * inner_loops
    assert record.samples_seconds == (2 * inner_loops + 1,) * repetitions
    assert record.communication_seconds == repetitions * inner_loops
    assert record.completed_work_units == repetitions * inner_loops
    assert record.correctness_passed
