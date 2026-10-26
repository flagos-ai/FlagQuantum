"""Verify count conservation and failure cleanup at the optional SDK boundary."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from flagquantum.benchmarking import qec_memory_sampling as benchmark
from flagquantum.qec import DemSample

pytestmark = pytest.mark.benchmark_contract


@pytest.fixture
def stream(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(requests=[], closed=False, offset=0)

    def sample(*, shots: int) -> DemSample:
        state.requests.append(shots)
        observed = (torch.arange(state.offset, state.offset + shots) % 2).to(torch.int8)
        state.offset += shots
        return DemSample(torch.zeros((shots, 24), dtype=torch.int8), observed[:, None])

    @contextmanager
    def compile_sampler(*args: object, **kwargs: object) -> Iterator[SimpleNamespace]:
        try:
            yield SimpleNamespace(sample=sample)
        finally:
            state.closed = True

    monkeypatch.setattr(benchmark, "compile_memory_sampler", compile_sampler)
    monkeypatch.setattr(
        benchmark.PyMatchingDecoder,
        "from_detector_error_model",
        lambda model: SimpleNamespace(
            decode_batch=lambda events: torch.zeros(
                (events.shape[0], 1), dtype=torch.bool
            )
        ),
    )
    monkeypatch.setattr(benchmark, "version", lambda name: "SDK-boundary-test-double")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return state


def test_tail_batch_counts_and_non_release_classification(
    stream: SimpleNamespace,
) -> None:
    result = benchmark.run_benchmark(shots=7, batch_size=3, warmup=0, iterations=1)
    row = result["observations"][0]
    assert stream.requests == [3, 3, 1]
    assert stream.closed
    assert row["completed_shots"] == 7
    assert row["logical_failures"] == 3  # Odd-labelled shots; no batch is dropped.
    assert row["visible_batch_tensor_bytes_max"] == 3 * (24 + 1 + 1)
    assert row["pipeline_seconds_excluding_imports_and_report_io"] >= sum(
        row["phases_seconds"].values()
    )
    assert result["non_release_evidence"] is True
    assert result["scalability_claim_allowed"] is False
    assert result["release_gate_allowed"] is False


@pytest.mark.parametrize("malformed", [False, True])
def test_decoder_failure_or_malformed_output_closes_stream(
    monkeypatch: pytest.MonkeyPatch, stream: SimpleNamespace, malformed: bool
) -> None:
    def decode(events: torch.Tensor) -> torch.Tensor:
        if malformed:
            return torch.zeros((events.shape[0], 2), dtype=torch.bool)
        raise RuntimeError("SDK prediction failed")

    monkeypatch.setattr(
        benchmark.PyMatchingDecoder,
        "from_detector_error_model",
        lambda model: SimpleNamespace(decode_batch=decode),
    )
    with pytest.raises(
        (ValueError, RuntimeError), match="logical batch|SDK prediction"
    ):
        benchmark.run_benchmark(shots=7, batch_size=3, warmup=0, iterations=1)
    assert stream.closed
    assert stream.requests == [3]


@pytest.mark.parametrize(
    "arguments",
    (
        {"shots": 0},
        {"shots": 65537},
        {"batch_size": True},
        {"iterations": 0},
        {"warmup": 4},
    ),
)
def test_outside_approved_measurement_envelope_is_refused(
    arguments: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        benchmark.run_benchmark(**arguments)


@pytest.mark.parametrize("shots", [16384, 65535, 65536])
def test_approved_stage_a_counts_do_not_change_batch_storage(
    stream: SimpleNamespace, shots: int
) -> None:
    result = benchmark.run_benchmark(shots=shots, batch_size=32, warmup=0, iterations=1)
    row = result["observations"][0]
    assert row["completed_shots"] == shots
    assert row["logical_failures"] == shots // 2
    assert sum(stream.requests) == shots
    assert max(stream.requests) == 32
    assert len(stream.requests) == (shots + 31) // 32
    assert row["visible_batch_tensor_bytes_max"] == 32 * 26
    assert stream.closed
