"""Core-only contract tests for bounded prepared measurement sampling."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_type_hints

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
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeSampler:
    monkeypatch.setattr(
        implementation, "__file__", str(tmp_path / "measurement_sampler.py")
    )
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


_FileFormat = Literal["01", "b8", "r8", "hits", "dets", "ptb64"]
_FORMAT_BYTES: dict[str, bytes] = {
    "01": b"10\n" * 64,
    "b8": b"\x01" * 64,
    "r8": b"\x00\x01" * 64,
    "hits": b"0\n" * 64,
    "dets": b"shot M0\n" * 64,
    "ptb64": b"\xff" * 8 + b"\x00" * 8,
}


@pytest.mark.parametrize("format", tuple(_FORMAT_BYTES))
def test_all_formats_publish_complete_records_without_sampling_a_tensor(
    fake: _FakeSampler,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    format: _FileFormat,
) -> None:
    def write(shots: int, *, filepath: str, format: str) -> None:
        assert shots == 64
        fake.calls.append((shots, format))
        Path(filepath).write_bytes(_FORMAT_BYTES[format])

    monkeypatch.setattr(fake, "sample_write", write)
    output = tmp_path / format
    _program().compile_sampler(seed=7).sample_write(64, filepath=output, format=format)
    assert output.read_bytes() == _FORMAT_BYTES[format]
    assert fake.calls == [(64, format)]
    assert list(tmp_path.iterdir()) == [output]


def test_additive_format_contract_preserves_original_b8_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    historical = json.loads(
        (root / "contracts/compiled-measurement-sampler-v1.json").read_text()
    )
    contract = json.loads(
        (
            root / "contracts/compiled-measurement-file-formats-v1-candidate.json"
        ).read_text()
    )
    assert historical["write_format"] == "b8"
    assert contract["formats"] == list(_FORMAT_BYTES)
    assert (
        get_type_hints(implementation.CompiledMeasurementSampler.sample_write)["format"]
        == _FileFormat
    )
    assert contract["max_file_bytes"] == implementation._MAX_FILE_BYTES


@pytest.mark.parametrize("shots", (1, 63, 65))
def test_ptb64_alignment_refuses_without_writer_or_staging(
    fake: _FakeSampler,
    tmp_path: Path,
    shots: int,
) -> None:
    sampler = _program().compile_sampler(seed=7)
    with pytest.raises(ValueError, match="multiple of 64"):
        sampler.sample_write(shots, filepath=tmp_path / "output", format="ptb64")
    assert not fake.calls and not list(tmp_path.iterdir())
    assert sampler.sample(1).shape == (1, 2)


@pytest.mark.parametrize("format", tuple(_FORMAT_BYTES))
def test_format_bounds_refuse_before_writer_and_accept_exact_limit(
    fake: _FakeSampler,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    format: _FileFormat,
) -> None:
    from flagquantum.simulation.stabilizer._measurement_file_validation import (
        _file_size_bound,
    )

    bound = _file_size_bound(64, 2, format)
    sampler = _program().compile_sampler(seed=7)
    monkeypatch.setattr(implementation, "_MAX_FILE_BYTES", bound - 1)
    with pytest.raises(CapabilityError, match="file bytes"):
        sampler.sample_write(64, filepath=tmp_path / "refused", format=format)
    assert not fake.calls and not list(tmp_path.iterdir())
    monkeypatch.setattr(implementation, "_MAX_FILE_BYTES", bound)

    def write(shots: int, *, filepath: str, format: str) -> None:
        Path(filepath).write_bytes(_FORMAT_BYTES[format])

    monkeypatch.setattr(fake, "sample_write", write)
    sampler.sample_write(64, filepath=tmp_path / "accepted", format=format)
    assert (tmp_path / "accepted").read_bytes() == _FORMAT_BYTES[format]


@pytest.mark.parametrize(
    ("format", "data"),
    (
        ("r8", b"\x00"),
        ("r8", b"\x03"),
        ("r8", b"\x02\x02"),
        ("hits", b"0"),
        ("hits", b"2\n"),
        ("hits", b"0,0\n"),
        ("hits", b"0,\n"),
        ("hits", b"0\n\n"),
        ("hits", b"x\n"),
        ("dets", b"shot M0"),
        ("dets", b"shot D0\n"),
        ("dets", b"shot M2\n"),
        ("dets", b"shot M0 M0\n"),
        ("dets", b"shot \n"),
        ("dets", b"shot\nshot\n"),
        ("01", b"1x\n"),
        ("01", b"100"),
    ),
)
def test_incomplete_or_malformed_outputs_are_not_published(
    fake: _FakeSampler,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    format: _FileFormat,
    data: bytes,
) -> None:
    def write(shots: int, *, filepath: str, format: str) -> None:
        Path(filepath).write_bytes(data)

    monkeypatch.setattr(fake, "sample_write", write)
    sampler = _program().compile_sampler(seed=7)
    with pytest.raises(OSError):
        sampler.sample_write(1, filepath=tmp_path / "output", format=format)
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="invalidated"):
        sampler.sample(1)


@pytest.mark.parametrize(
    ("columns", "encoded"),
    (
        (254, b"\xfe"),
        (255, b"\xff\x00"),
        (256, b"\xff\x01"),
    ),
)
def test_r8_zero_runs_include_the_terminating_bit(
    tmp_path: Path,
    columns: int,
    encoded: bytes,
) -> None:
    from flagquantum.simulation.stabilizer._measurement_file_validation import (
        _validate_record_file,
    )

    output = tmp_path / "runs.r8"
    output.write_bytes(encoded * 2)
    _validate_record_file(output, 2, columns, "r8", 2 * (columns + 1))
    output.write_bytes(encoded[:-1])
    with pytest.raises(OSError, match="incomplete"):
        _validate_record_file(output, 1, columns, "r8", columns + 1)


def test_stream_validation_reads_bounded_chunks_across_record_boundaries() -> None:
    from flagquantum.simulation.stabilizer import (
        _measurement_file_validation as validation,
    )

    class BoundedReader(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size == 65536
            return super().read(size)

    validation._validate_sparse_text(
        BoundedReader(b"0,10\n" * 14000), 14000, 11, dets=False
    )
    validation._validate_sparse_text(
        BoundedReader(b"shot M0 M10\n" * 6000), 6000, 11, dets=True
    )
    validation._validate_sparse_text(BoundedReader(b"\n"), 1, 2, dets=False)
    validation._validate_sparse_text(BoundedReader(b"shot\n"), 1, 2, dets=True)
    validation._validate_r8(BoundedReader(b"\x00\x01" * 33000), 33000, 2)


@pytest.mark.parametrize("format", tuple(_FORMAT_BYTES))
def test_writer_failure_cleans_staging_and_invalidates_all_formats(
    fake: _FakeSampler,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    format: _FileFormat,
) -> None:
    def fail(shots: int, *, filepath: str, format: str) -> None:
        Path(filepath).write_bytes(b"partial")
        raise OSError("controlled disk or close failure")

    monkeypatch.setattr(fake, "sample_write", fail)
    sampler = _program().compile_sampler(seed=7)
    with pytest.raises(OSError, match="controlled"):
        sampler.sample_write(64, filepath=tmp_path / "output", format=format)
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="invalidated"):
        sampler.sample(1)


def test_integer_and_unknown_format_preflight() -> None:
    from flagquantum.simulation.stabilizer._measurement_file_validation import (
        _file_size_bound,
    )

    with pytest.raises(CapabilityError, match="integer range"):
        _file_size_bound(sys.maxsize + 1, 1, "01")
    with pytest.raises(ValueError, match="format"):
        _file_size_bound(1, 1, "unknown")


@pytest.mark.parametrize("encoding", ["01", "b8", "r8", "hits", "dets", "ptb64"])
def test_zero_width_file_framing(tmp_path: Path, encoding: str) -> None:
    from flagquantum.simulation.stabilizer._measurement_file_validation import (
        _file_size_bound,
        _validate_record_file,
    )

    content = {
        "01": b"\n",
        "b8": b"",
        "r8": b"\x00",
        "hits": b"\n",
        "dets": b"shot\n",
        "ptb64": b"",
    }[encoding] * 64
    path = tmp_path / encoding
    path.write_bytes(content)
    _validate_record_file(
        path, 64, 0, encoding, _file_size_bound(64, 0, encoding), sections=(("L", 0),)
    )
