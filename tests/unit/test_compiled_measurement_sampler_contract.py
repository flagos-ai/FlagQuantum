"""Core-only contract tests for bounded prepared measurement sampling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    ("program", "reason"),
    [
        (
            DynamicCircuit(2)
            .h(0)
            .t(0)
            .h(0)
            .measure(0, classical_bit=0)
            .conditional("x", 1, classical_bit=0),
            "outside the exact Clifford slice",
        ),
        (
            DynamicCircuit(2)
            .h(0)
            .measure(0, classical_bit=0)
            .conditional("h", 1, classical_bit=0),
            "supports one classical test on X",
        ),
    ],
)
def test_route_boundaries_refuse_before_engine_preparation(
    program: DynamicCircuit, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_prepare(*args: object) -> None:
        pytest.fail("rejected circuit entered engine preparation")

    monkeypatch.setattr(implementation, "_prepare_sdk_sampler", unexpected_prepare)
    monkeypatch.setattr(implementation, "_prepare_embedded_sampler", unexpected_prepare)
    with pytest.raises(CapabilityError, match=reason):
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


def test_embedded_profile_uses_record_targets_and_owns_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "_embedded_stim_profile.txt"
    marker.write_text("stim-1.16.0\n")
    monkeypatch.setattr(
        implementation, "__file__", str(tmp_path / "measurement_sampler.py")
    )
    handle = object()
    calls = []

    def prepare(text: str, seed: int | None) -> object:
        assert text == "X 0\nM 0\nCX rec[-1] 1\nR 0\nM 0\nCX rec[-1] 2\nM 1\nM 2"
        assert seed == 7
        calls.append("prepare")
        return handle

    def sample(native_handle: object, shots: int, packed: bool) -> bytearray:
        assert native_handle is handle and shots == 2
        calls.append("packed" if packed else "unpacked")
        return bytearray([5, 5] if packed else [1, 0, 1, 0] * 2)

    def write_b8(native_handle: object, shots: int, filepath: str) -> None:
        assert native_handle is handle and shots == 2
        calls.append("write")
        Path(filepath).write_bytes(b"\x05" * shots)

    native = SimpleNamespace(prepare=prepare, sample=sample, write_b8=write_b8)
    monkeypatch.setitem(
        sys.modules, "flagquantum.simulation.stabilizer._embedded_stim", native
    )
    circuit = (
        DynamicCircuit(3)
        .x(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .reset(0)
        .measure(0, classical_bit=0)
        .conditional("x", 2, classical_bit=0)
        .measure(1)
        .measure(2)
    )
    sampler = circuit.compile_sampler(seed=7)
    first = sampler.sample(2)
    second = sampler.sample(2)
    assert first.dtype == second.dtype == torch.bool
    first.zero_()
    assert second.tolist() == [[True, False, True, False]] * 2
    packed = sampler.sample(2, bit_packed=True)
    assert packed.dtype == torch.uint8
    assert packed.tolist() == [[5], [5]]
    output = tmp_path / "records.b8"
    sampler.sample_write(2, filepath=output, format="b8")
    assert output.read_bytes() == b"\x05\x05"
    assert calls == ["prepare", "unpacked", "unpacked", "packed", "write"]


def test_embedded_profile_does_not_fall_back_when_extension_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "_embedded_stim_profile.txt").write_text("stim-1.16.0\n")
    monkeypatch.setattr(
        implementation, "__file__", str(tmp_path / "measurement_sampler.py")
    )
    monkeypatch.setitem(
        sys.modules, "flagquantum.simulation.stabilizer._embedded_stim", None
    )

    def unexpected_sdk(*args: object) -> None:
        pytest.fail("embedded profile fell back to the Stim SDK")

    monkeypatch.setattr(implementation, "_prepare_sdk_sampler", unexpected_sdk)
    with pytest.raises(ModuleNotFoundError, match="_embedded_stim"):
        _program().compile_sampler(seed=7)


def test_embedded_converter_preserves_buffers_and_missing_extension_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from flagquantum.qec.execution.context import MeasurementMap
    from flagquantum.simulation.stabilizer import _record_conversion as conversion

    (tmp_path / "_embedded_stim_profile.txt").write_text("stim-1.16.0\n")
    monkeypatch.setattr(conversion, "__file__", str(tmp_path / "_record_conversion.py"))
    prepared: list[str] = []
    calls: list[bytes] = []

    def prepare(text: str) -> str:
        prepared.append(text)
        return "handle"

    def convert(handle: str, raw: bytes, shots: int) -> tuple[bytearray, bytearray]:
        assert handle == "handle" and shots == 1
        calls.append(raw)
        return bytearray([0]), bytearray()

    native = SimpleNamespace(prepare_converter=prepare, convert_records=convert)
    monkeypatch.setitem(
        sys.modules, "flagquantum.simulation.stabilizer._embedded_stim", native
    )
    handle = (
        DynamicCircuit(1)
        .x(0)
        .measure(0)
        .compile_m2d_converter(
            detector_map=MeasurementMap(1, ((0,),)),
            observable_map=MeasurementMap(1, ()),
        )
    )
    first, obs = handle.convert(
        measurements=torch.tensor([[True]]), separate_observables=True
    )
    second, _ = handle.convert(
        measurements=torch.tensor([[True]]), separate_observables=True
    )
    first.fill_(True)
    assert not second.any() and obs.shape == (1, 0)
    assert calls == [b"\x01", b"\x01"]
    assert prepared == ["X 0\nM 0\nDETECTOR rec[-1]"]
    monkeypatch.setitem(
        sys.modules, "flagquantum.simulation.stabilizer._embedded_stim", None
    )
    with pytest.raises(ModuleNotFoundError):
        DynamicCircuit(1).measure(0).compile_m2d_converter(
            detector_map=MeasurementMap(1, ((0,),)),
            observable_map=MeasurementMap(1, ()),
        )


@pytest.mark.parametrize(
    "result", [None, (b"\x00", bytearray()), (bytearray(), bytearray())]
)
def test_embedded_converter_invalid_buffers_refused(
    result: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.simulation.stabilizer._embedded_sampler import (
        _EmbeddedRecordConverter,
    )

    native = SimpleNamespace(
        prepare_converter=lambda text: None, convert_records=lambda *args: result
    )
    monkeypatch.setitem(
        sys.modules, "flagquantum.simulation.stabilizer._embedded_stim", native
    )
    converter = _EmbeddedRecordConverter("M 0\nDETECTOR rec[-1]", 1, 0)
    with pytest.raises(ValueError, match="invalid output"):
        converter.convert(torch.tensor([[False]]))


def test_new_detector_noise_admission_keeps_old_sampler_refusal() -> None:
    program = DynamicCircuit(1).bit_flip(0, 0.1).measure(0)
    with pytest.raises(CapabilityError, match="exact Clifford slice"):
        program.compile_sampler(seed=7)
    from flagquantum.qec.execution.context import MeasurementMap

    with pytest.raises(CapabilityError, match="exact Clifford slice"):
        program.compile_m2d_converter(
            detector_map=MeasurementMap(1, ((0,),)),
            observable_map=MeasurementMap(1, ()),
        )
