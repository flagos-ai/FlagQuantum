"""Protect original-source observation, tail batches and failed-run cleanup."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmarks import qec_compatibility_matrix as matrix

pytestmark = pytest.mark.benchmark_contract


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("entrance", ["native", "exported_stim"])
def test_real_two_entrances_complete_the_tail(basis: str, entrance: str) -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    row = matrix._observe(
        distance=3, basis=basis, entrance=entrance, shots=17, batch_size=8
    )
    assert row["completed_shots"] == 17
    assert row["batches"] == 3
    assert 0 <= row["logical_failures"] <= 17
    assert row["pipeline_seconds"] >= sum(row["phases_seconds"].values())
    assert row["visible_batch_tensor_bytes_max"] == 8 * (24 + 1 + 1)


@pytest.mark.parametrize("malformed", [False, True])
def test_prediction_failure_closes_original_sampler(
    monkeypatch: pytest.MonkeyPatch, malformed: bool
) -> None:
    pytest.importorskip("stim")
    state = SimpleNamespace(closed=False, requested=[])

    def sample(*, shots: int) -> SimpleNamespace:
        state.requested.append(shots)
        return SimpleNamespace(
            shots=shots,
            detectors=torch.zeros((shots, 24), dtype=torch.int8),
            observables=torch.zeros((shots, 1), dtype=torch.int8),
        )

    @contextmanager
    def compile_sampler(*args: object, **kwargs: object) -> Iterator[SimpleNamespace]:
        try:
            yield SimpleNamespace(sample=sample)
        finally:
            state.closed = True

    def decode(events: torch.Tensor) -> torch.Tensor:
        if malformed:
            return torch.zeros((events.shape[0], 2), dtype=torch.bool)
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(matrix, "compile_memory_sampler", compile_sampler)
    monkeypatch.setattr(
        matrix.PyMatchingDecoder,
        "from_detector_error_model",
        lambda model: SimpleNamespace(decode_batch=decode),
    )
    with pytest.raises(
        (RuntimeError, ValueError), match="prediction failed|incomplete"
    ):
        matrix._observe(
            distance=3, basis="z", entrance="native", shots=17, batch_size=8
        )
    assert state.closed
    assert state.requested == [8]


@pytest.mark.parametrize(
    "delta",
    [
        {"distance": 19},
        {"batch_size": True},
        {"shots": 4098},
        {"warmup": 4},
        {"iterations": 0},
        {"basis": "y"},
        {"entrance": "graph"},
    ],
)
def test_outside_budget_fails_before_collection(delta: dict[str, object]) -> None:
    args = dict(
        distance=3,
        basis="z",
        entrance="native",
        shots=17,
        batch_size=8,
        warmup=0,
        iterations=1,
    )
    args.update(delta)
    with pytest.raises(ValueError):
        matrix._run_case(**args)
