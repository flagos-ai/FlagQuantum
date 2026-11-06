"""Core-only contract tests for bounded prepared measurement sampling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.simulation.stabilizer import measurement_sampler as implementation

pytestmark = pytest.mark.unit


class _FakeSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool | str]] = []
        self.short_write = False

    def sample(self, shots: int, *, bit_packed: bool) -> torch.Tensor:
        self.calls.append((shots, bit_packed))
        if bit_packed:
            return torch.ones((shots, 1), dtype=torch.uint8)
        return torch.tensor([[True, False]]).expand(shots, 2).clone()

    def sample_write(self, shots: int, *, filepath: str, format: str) -> None:
        self.calls.append((shots, format))
        Path(filepath).write_bytes(b"\x01" * (shots - self.short_write))


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeSampler:
    sdk = _FakeSampler()

    def prepare(operations: object, seed: int | None) -> _FakeSampler:
        assert operations == (
            ("X", (0,), None),
            ("M", (0,), None),
            ("R", (0,), None),
            ("M", (0,), None),
        )
        assert seed == 7
        return sdk

    monkeypatch.setattr(implementation, "_prepare_sdk_sampler", prepare)
    monkeypatch.setattr(implementation, "_as_torch_sample", lambda data: data)
    return sdk


def _program() -> DynamicCircuit:
    return DynamicCircuit(16).x(0).measure(0).reset(0).measure(0)


def test_one_preparation_and_owned_outputs(fake: _FakeSampler, tmp_path: Path) -> None:
    program = _program()
    sampler = program.compile_sampler(seed=7)
    assert isinstance(sampler, implementation.CompiledMeasurementSampler)
    assert not hasattr(fq, "CompiledMeasurementSampler")
    program.x(0)
    first = sampler.sample(1)
    second = sampler.sample(1)
    first.zero_()
    assert second.tolist() == [[True, False]]
    assert sampler.sample(1, bit_packed=True).tolist() == [[1]]
    destination = tmp_path / "new.b8"
    sampler.sample_write(3, filepath=destination, format="b8")
    assert destination.read_bytes() == b"\x01" * 3
    assert fake.calls == [(1, False), (1, False), (1, True), (3, "b8")]


def test_rejects_payload_frame_and_disk_excess_before_sdk(
    fake: _FakeSampler, tmp_path: Path
) -> None:
    sampler = _program().compile_sampler(seed=7)
    with pytest.raises(CapabilityError, match="payload"):
        sampler.sample(64 * 1024**2 // 2 + 1)
    with pytest.raises(CapabilityError, match="frame"):
        sampler.sample(50_000_000, bit_packed=True)
    with pytest.raises(CapabilityError, match="file"):
        sampler.sample_write(1024**3 + 1, filepath=tmp_path / "new", format="b8")
    assert fake.calls == []
    assert list(tmp_path.iterdir()) == []


def test_no_overwrite_and_incomplete_file_remains_unpublished(
    fake: _FakeSampler, tmp_path: Path
) -> None:
    sampler = _program().compile_sampler(seed=7)
    existing = tmp_path / "existing.b8"
    existing.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        sampler.sample_write(2, filepath=existing, format="b8")
    assert existing.read_bytes() == b"preserved"
    assert fake.calls == []
    destination = tmp_path / "new.b8"
    fake.short_write = True
    with pytest.raises(OSError, match="size"):
        sampler.sample_write(3, filepath=destination, format="b8")
    assert sorted(tmp_path.iterdir()) == [existing]
    with pytest.raises(RuntimeError, match="invalidated"):
        sampler.sample(1)


def test_unsupported_programs_refuse_before_preparation() -> None:
    for program in (
        fq.Circuit(1).h(0),
        DynamicCircuit(1).measure(0).t(0),
        DynamicCircuit(2).measure(0).conditional("z", 1, classical_bit=0),
        DynamicCircuit(2)
        .measure(0)
        .measure(1)
        .conditional("x", 0, conditions={0: 1, 1: 1}),
    ):
        with pytest.raises(CapabilityError):
            program.compile_sampler()


def test_exact_budget_boundaries_without_large_allocations(
    fake: _FakeSampler, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert implementation._check_sample_budget(33_554_432, 2, 1, False) == 2
    assert implementation._check_sample_budget(65_536, 4096, 2048, True) == 512
    with pytest.raises(CapabilityError, match="frame estimate"):
        implementation._check_sample_budget(65_537, 4096, 2048, True)
    monkeypatch.setattr(implementation, "_MAX_FILE_BYTES", 3)
    sampler = _program().compile_sampler(seed=7)
    destination = tmp_path / "at-limit.b8"
    with pytest.raises(CapabilityError, match="file"):
        sampler.sample_write(4, filepath=destination, format="b8")
    assert fake.calls == []
    sampler.sample_write(3, filepath=destination, format="b8")
    assert destination.read_bytes() == b"\x01" * 3


def test_symlink_and_publication_race_never_replace_destination(
    fake: _FakeSampler, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sampler = _program().compile_sampler(seed=7)
    missing = tmp_path / "missing"
    link = tmp_path / "symlink.b8"
    link.symlink_to(missing)
    with pytest.raises(FileExistsError):
        sampler.sample_write(1, filepath=link, format="b8")
    assert link.is_symlink() and not missing.exists()
    assert fake.calls == []

    path = tmp_path / "concurrent.b8"
    original = fake.sample_write

    def write_with_collision(shots: int, *, filepath: str, format: str) -> None:
        original(shots, filepath=filepath, format=format)
        path.write_bytes(b"concurrent writer")

    monkeypatch.setattr(fake, "sample_write", write_with_collision)
    with pytest.raises(FileExistsError):
        sampler.sample_write(1, filepath=path, format="b8")
    assert path.read_bytes() == b"concurrent writer"
    assert set(tmp_path.iterdir()) == {link, path}
