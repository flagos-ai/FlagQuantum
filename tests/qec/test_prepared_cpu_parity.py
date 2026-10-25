"""Prepared parity state belongs to one live sampler, never a sample record."""

import weakref

import pytest
import torch

from flagquantum.qec import (
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
    compile_memory_sampler,
)
from flagquantum.qec.execution import sampling

pytestmark = pytest.mark.unit


class _Source:
    def __init__(self, record: torch.Tensor) -> None:
        self.record = record
        self.closed = False

    def sample(self, *, shots: int) -> torch.Tensor:
        return self.record[:shots]

    def close(self) -> None:
        self.closed = True


def test_draws_do_not_rebuild_index_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    record = torch.tensor([[0, 1, 1, 0, 1, 1, 0, 1, 0]]).expand(5, -1)
    source = _Source(record)
    monkeypatch.setattr(sampling, "_prepare_noisy_measurements", lambda *a, **k: source)
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)
    with compile_memory_sampler(memory, noise=PhenomenologicalNoise()) as handle:
        expected = handle.sample(shots=3)

        def rebuild(*args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("index tensor rebuilt during draw")

        monkeypatch.setattr(torch, "tensor", rebuild)
        actual = handle.sample(shots=3)
        assert torch.equal(actual.detectors, expected.detectors)
        assert torch.equal(actual.observables, expected.observables)
        actual.detectors.fill_(1)
        assert torch.equal(handle.sample(shots=3).detectors, expected.detectors)
    assert source.closed


@pytest.mark.parametrize("width", [255, 256, 257, 513])
def test_prepared_ragged_parity_matches_independent_xor(width: int) -> None:
    generator = torch.Generator().manual_seed(53)
    record = torch.randint(2, (7, width * 2), generator=generator)[:, ::2]
    original = record.clone()
    columns = ((), (0, 0), (width - 1,), tuple(range(width)), (0, 1, 0))
    source = _Source(record)
    handle = sampling._memory_sampler(source, columns, columns[:2])
    expected = torch.tensor(
        [
            [sum(int(row[i]) for i in parity) % 2 for parity in columns]
            for row in record
        ],
        dtype=torch.int8,
    )
    with handle:
        actual = handle.sample(shots=7)
        assert torch.equal(actual.detectors, expected)
        assert torch.equal(actual.observables, expected[:, :2])
        groups = tuple(
            weakref.ref(tensor) for group in handle._detector_groups for tensor in group
        )
        actual.detectors.zero_()
        assert torch.equal(handle.sample(shots=7).detectors, expected)
        assert torch.equal(record, original)
    assert all(reference() is None for reference in groups)
    assert source.closed


def test_native_prepare_failure_releases_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _Source(torch.zeros((1, 9), dtype=torch.int8))
    monkeypatch.setattr(sampling, "_prepare_noisy_measurements", lambda *a, **k: source)
    original = sampling._prepare_parity_groups
    prepared = 0
    sentinel = RuntimeError("index allocation failed")

    def fail_second(
        columns: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        nonlocal prepared
        prepared += 1
        if prepared == 2:
            raise sentinel
        return original(columns)

    monkeypatch.setattr(sampling, "_prepare_parity_groups", fail_second)
    with pytest.raises(RuntimeError) as caught:
        compile_memory_sampler(
            build_memory_circuit(RepetitionCode(3), rounds=3),
            noise=PhenomenologicalNoise(),
        )
    assert caught.value is sentinel and source.closed


def test_one_byte_copy_serves_both_output_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Callable
    from typing import Any

    from torch.utils._python_dispatch import TorchDispatchMode

    class ByteCopies(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.copies = 0

        def __torch_dispatch__(
            self,
            func: Callable[..., Any],
            types: tuple[type, ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            options = kwargs or {}
            if (
                str(func) == "aten._to_copy.default"
                and options.get("dtype") == torch.uint8
            ):
                self.copies += 1
            return func(*args, **options)

    source = _Source(torch.zeros((4, 9), dtype=torch.int64))
    monkeypatch.setattr(sampling, "_prepare_noisy_measurements", lambda *a, **k: source)
    with compile_memory_sampler(
        build_memory_circuit(RepetitionCode(3), rounds=3),
        noise=PhenomenologicalNoise(),
    ) as handle:
        counter = ByteCopies()
        with counter:
            result = handle.sample(shots=4)
        assert counter.copies == 1
        assert result.detectors.dtype == result.observables.dtype == torch.int8


def test_draw_failure_releases_prepared_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _Source(torch.zeros((1, 3), dtype=torch.int8))
    handle = sampling._memory_sampler(source, ((0, 1),), ((2,),))
    refs = tuple(
        weakref.ref(tensor)
        for group in handle._detector_groups + handle._observable_groups
        for tensor in group
    )
    sentinel = RuntimeError("engine draw failed")

    def fail(*, shots: int) -> torch.Tensor:
        raise sentinel

    monkeypatch.setattr(source, "sample", fail)
    with pytest.raises(RuntimeError) as caught:
        handle.sample(shots=1)
    assert caught.value is sentinel and source.closed
    assert all(reference() is None for reference in refs)
    with pytest.raises(RuntimeError, match="closed"):
        handle.sample(shots=1)
