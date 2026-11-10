"""File conversion scenarios shared by contract fakes and numerical producers."""

from __future__ import annotations

import inspect
import io
import json
import os
import shutil
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.qec.execution import measurement_conversion as public
from flagquantum.qec.execution.context import MeasurementMap
from flagquantum.simulation.stabilizer import _measurement_file_validation as files
from flagquantum.simulation.stabilizer import _record_conversion as engine

RecordFormat = Literal["01", "b8", "r8", "ptb64", "hits", "dets"]

pytestmark = pytest.mark.unit

INPUTS = {
    "01": b"010\n" * 64,
    "b8": b"\x02" * 64,
    "r8": b"\x01\x01" * 64,
    "hits": b"1\n" * 64,
    "dets": b"shot M1\n" * 64,
    "ptb64": b"\0" * 8 + b"\xff" * 8 + b"\0" * 8,
}
DETECTORS = {
    "01": b"10\n" * 64,
    "b8": b"\x01" * 64,
    "r8": b"\x00\x01" * 64,
    "hits": b"0\n" * 64,
    "dets": b"shot D0\n" * 64,
}
OBSERVABLES = {
    "01": b"1\n" * 64,
    "b8": b"\x01" * 64,
    "r8": b"\x00\x00" * 64,
    "hits": b"0\n" * 64,
    "dets": b"shot L0\n" * 64,
}
APPENDED = {
    "01": b"101\n" * 64,
    "b8": b"\x05" * 64,
    "r8": b"\x00\x01\x00" * 64,
    "hits": b"0,2\n" * 64,
    "dets": b"shot D0 L0\n" * 64,
}


def _program() -> fq.Circuit:
    source = (
        DynamicCircuit(1)
        .measure(0, classical_bit=0)
        .measure(0, classical_bit=0)
        .conditional("x", 0, classical_bit=0)
        .measure(0, classical_bit=0)
    )
    return fq.Circuit.from_ir(source.to_ir())


def _converter() -> public.CompiledMeasurementsToDetectionEventsConverter:
    handle = _program().compile_m2d_converter(
        detector_map=MeasurementMap(3, ((0, 1), (0, 2))),
        observable_map=MeasurementMap(3, ((1,),)),
    )
    assert isinstance(handle, public.CompiledMeasurementsToDetectionEventsConverter)
    return handle


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    calls: list[tuple[str, bool]] = []

    def dense(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.cat(
                (records[:, :1] ^ records[:, 1:2], records[:, :1] ^ records[:, 2:3]),
                dim=1,
            ),
            records[:, 1:2].clone(),
        )

    def write(
        source: str,
        input_format: str,
        target: str,
        output_format: str,
        append: bool,
        observable: str | None,
        obs_format: str,
    ) -> None:
        assert Path(source).read_bytes() == INPUTS[input_format]
        calls.append((input_format, append))
        Path(target).write_bytes((APPENDED if append else DETECTORS)[output_format])
        if observable is not None:
            Path(observable).write_bytes(OBSERVABLES[obs_format])

    def prepare(text: str, detectors: int, observables: int) -> engine._RecordBackend:
        assert "CX rec[-1] 0" in text and (detectors, observables) == (2, 1)
        return engine._RecordBackend(dense, write)

    monkeypatch.setattr(engine, "_prepare_explicit_backend", prepare)
    return calls


def _scenario(
    handle: public.CompiledMeasurementsToDetectionEventsConverter,
    root: Path,
    input_format: RecordFormat,
    output_format: RecordFormat,
    layout: str,
) -> None:
    source, target, logical = root / "input", root / "events", root / "logical"
    source.write_bytes(INPUTS[input_format])
    handle.convert_file(
        measurements_filepath=source,
        measurements_format=input_format,
        detection_events_filepath=target,
        detection_events_format=output_format,
        append_observables=layout != "separate",
        obs_out_filepath=logical,
        obs_out_format="b8" if layout == "mixed" else output_format,
    )
    assert (
        target.read_bytes()
        == (DETECTORS if layout == "separate" else APPENDED)[output_format]
    )
    assert (
        logical.read_bytes()
        == OBSERVABLES["b8" if layout == "mixed" else output_format]
    )
    assert source.read_bytes() == INPUTS[input_format]
    assert sorted(p.name for p in root.iterdir()) == ["events", "input", "logical"]


@pytest.mark.parametrize("input_format", INPUTS)
@pytest.mark.parametrize("output_format", DETECTORS)
@pytest.mark.parametrize("layout", ["separate", "appended", "mixed"])
def test_fake_consumer_replacement(
    fake: list[tuple[str, bool]],
    tmp_path: Path,
    input_format: RecordFormat,
    output_format: RecordFormat,
    layout: str,
) -> None:
    _scenario(_converter(), tmp_path, input_format, output_format, layout)
    assert fake == [(input_format, layout != "separate")]


def test_approved_signature_and_candidate() -> None:
    contract = json.loads(
        Path("contracts/qec-measurement-file-conversion-v1-candidate.json").read_text()
    )
    signature = inspect.signature(
        public.CompiledMeasurementsToDetectionEventsConverter.convert_file
    )
    assert list(signature.parameters) == contract["parameters"]
    assert {
        name: item.default
        for name, item in signature.parameters.items()
        if item.default is not inspect.Parameter.empty
    } == contract["defaults"]
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for name, item in signature.parameters.items()
        if name != "self"
    )


@pytest.mark.parametrize(
    "format,raw",
    [
        ("b8", b"\x82"),
        ("ptb64", b"\0" * 23),
        ("01", b"01\n"),
        ("01", b"010"),
        ("hits", b"1,1\n"),
        ("dets", b"shot D1\n"),
        ("dets", b"shot M3\n"),
        ("r8", b"\xff"),
        ("r8", b"\x04"),
    ],
)
def test_malformed_input_precedes_engine(
    fake: list[tuple[str, bool]], tmp_path: Path, format: RecordFormat, raw: bytes
) -> None:
    source, target = tmp_path / "input", tmp_path / "output"
    source.write_bytes(raw)
    with pytest.raises(OSError):
        _converter().convert_file(
            measurements_filepath=source,
            measurements_format=format,
            detection_events_filepath=target,
        )
    assert not fake and not target.exists()
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("format", INPUTS)
def test_empty_input_skips_numerical_conversion(
    fake: list[tuple[str, bool]], tmp_path: Path, format: RecordFormat
) -> None:
    source, target, logical = (
        tmp_path / "input",
        tmp_path / "output",
        tmp_path / "logical",
    )
    source.touch()
    _converter().convert_file(
        measurements_filepath=source,
        measurements_format=format,
        detection_events_filepath=target,
        obs_out_filepath=logical,
    )
    assert target.read_bytes() == logical.read_bytes() == b"" and not fake


@pytest.mark.parametrize(
    "option", ["sweep", "output_ptb64", "obs_ptb64", "flag", "format"]
)
def test_unsupported_requests_precede_file_io(
    fake: list[tuple[str, bool]], tmp_path: Path, option: str
) -> None:
    handle = _converter()
    source, target = tmp_path / "missing", tmp_path / "output"
    with pytest.raises((ValueError, TypeError, CapabilityError)):
        if option == "sweep":
            handle.convert_file(
                measurements_filepath=source,
                detection_events_filepath=target,
                sweep_bits_filepath=source,
            )
        elif option == "output_ptb64":
            handle.convert_file(
                measurements_filepath=source,
                detection_events_filepath=target,
                detection_events_format="ptb64",
            )
        elif option == "obs_ptb64":
            handle.convert_file(
                measurements_filepath=source,
                detection_events_filepath=target,
                obs_out_filepath=tmp_path / "obs",
                obs_out_format="ptb64",
            )
        elif option == "flag":
            getattr(handle, "convert_file")(
                measurements_filepath=source,
                detection_events_filepath=target,
                append_observables=1,
            )
        else:
            getattr(handle, "convert_file")(
                measurements_filepath=source,
                detection_events_filepath=target,
                measurements_format="invalid",
            )
    assert not fake and not target.exists()


@pytest.mark.parametrize("kind", ["symlink", "fifo", "existing", "duplicate", "alias"])
def test_file_admission(
    fake: list[tuple[str, bool]], tmp_path: Path, kind: str
) -> None:
    source, target = tmp_path / "input", tmp_path / "output"
    if kind == "symlink":
        source.symlink_to(tmp_path / "absent")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("this operating system does not support FIFO creation")
        os.mkfifo(source)
    else:
        source.write_bytes(INPUTS["b8"])
    if kind == "existing":
        target.write_bytes(b"preserved")
    if kind == "alias":
        target = source
    with pytest.raises((OSError, ValueError)):
        _converter().convert_file(
            measurements_filepath=source,
            measurements_format="b8",
            detection_events_filepath=target,
            obs_out_filepath=target if kind == "duplicate" else None,
        )
    assert not fake
    if kind == "existing":
        assert target.read_bytes() == b"preserved"


def test_output_failure_preserves_dense_reuse(
    fake: list[tuple[str, bool]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = _converter()
    source, target, logical = (
        tmp_path / "input",
        tmp_path / "events",
        tmp_path / "logical",
    )
    source.write_bytes(INPUTS["b8"])
    real_link = os.link

    def link(src: str | Path, dst: str | Path) -> None:
        if Path(dst) == logical:
            raise OSError("injected second publication failure")
        real_link(src, dst)

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(OSError, match="second publication"):
        handle.convert_file(
            measurements_filepath=source,
            measurements_format="b8",
            detection_events_filepath=target,
            detection_events_format="b8",
            obs_out_filepath=logical,
            obs_out_format="b8",
        )
    assert not target.exists() and not logical.exists()
    dets, obs = handle.convert(
        measurements=torch.tensor([[False, True, False]]), separate_observables=True
    )
    assert dets.tolist() == [[True, False]] and obs.tolist() == [[True]]


@pytest.mark.integration
@pytest.mark.parametrize("input_format", INPUTS)
@pytest.mark.parametrize("output_format", DETECTORS)
@pytest.mark.parametrize("layout", ["separate", "appended", "mixed"])
def test_sdk_fixed_independent_bytes(
    tmp_path: Path, input_format: RecordFormat, output_format: RecordFormat, layout: str
) -> None:
    pytest.importorskip("stim")
    _scenario(_converter(), tmp_path, input_format, output_format, layout)


@pytest.mark.parametrize(
    "limit",
    ["_MAX_SNAPSHOT_BYTES", "_MAX_FILE_OUTPUT_BYTES", "_MAX_FILE_PRINCIPAL_BYTES"],
)
def test_budgets_precede_numerical_conversion(
    fake: list[tuple[str, bool]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
) -> None:
    handle = _converter()
    source, target = tmp_path / "input", tmp_path / "output"
    source.write_bytes(INPUTS["b8"])
    monkeypatch.setattr(engine, limit, 1)
    with pytest.raises(CapabilityError, match="byte limit"):
        handle.convert_file(
            measurements_filepath=source,
            measurements_format="b8",
            detection_events_filepath=target,
        )
    assert not fake and list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("replace_with_symlink", [False, True])
def test_snapshot_without_unix_open_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_with_symlink: bool,
) -> None:
    source, snapshot = tmp_path / "input", tmp_path / "snapshot"
    source.write_bytes(b"010\n")
    original_open = os.open
    for name in ("O_NOFOLLOW", "O_NONBLOCK"):
        monkeypatch.delattr(os, name, raising=False)

    def open_input(path: str | Path, flags: int) -> int:
        descriptor = original_open(path, flags)
        if replace_with_symlink:
            saved = tmp_path / "saved"
            source.rename(saved)
            source.symlink_to(saved)
        return descriptor

    monkeypatch.setattr(os, "open", open_input)
    if replace_with_symlink:
        with pytest.raises(OSError, match="changed"):
            engine._snapshot(source, snapshot)
        assert not snapshot.exists()
    else:
        assert engine._snapshot(source, snapshot) == 4
        assert snapshot.read_bytes() == b"010\n"


def test_snapshot_detects_changed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, snapshot = tmp_path / "input", tmp_path / "snapshot"
    source.write_bytes(b"010\n")
    real_fstat = os.fstat
    calls = 0

    def fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            source.write_bytes(b"101\n")
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "fstat", fstat)
    with pytest.raises(OSError, match="changed"):
        engine._snapshot(source, snapshot)
    assert snapshot.read_bytes() == b"010\n"


def test_parser_read_bound_and_partial_tail(tmp_path: Path) -> None:
    class BoundedInput(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 < size <= 65536
            return super().read(size)

    stream = BoundedInput(b"010\n" * 20000)
    assert (
        files._validate_sparse_text(
            BoundedInput(b"shot M1\n" * 20000), None, 3, dets=True
        )
        == 20000
    )
    assert sum(1 for _ in files._file_bytes(stream)) == 80000
    source = tmp_path / "input"
    source.write_bytes(b"010\n" * 20000)
    assert files._inspect_measurement_file(source, 3, "01") == 20000
    source.write_bytes(b"010\n0")
    with pytest.raises(OSError):
        files._inspect_measurement_file(source, 3, "01")


def test_input_snapshot_survives_caller_replacement(tmp_path: Path) -> None:
    source, target = tmp_path / "input", tmp_path / "output"
    source.write_bytes(b"\x02" * 64)

    def dense(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("file conversion called the dense engine")

    def write(
        snapshot: str,
        input_format: str,
        output: str,
        output_format: str,
        append: bool,
        observable: str | None,
        obs_format: str,
    ) -> None:
        assert Path(snapshot) != source
        source.write_bytes(b"changed after the snapshot was accepted")
        assert Path(snapshot).read_bytes() == b"\x02" * 64
        Path(output).write_bytes(b"\x01" * 64)

    prepared = engine._PreparedConversion(
        engine._RecordBackend(dense, write), (3, 2, 1), 1
    )
    prepared.convert_file(source, "b8", target, "b8", False, None, "01")
    assert target.read_bytes() == b"\x01" * 64


def test_free_space_is_grouped_for_both_outputs(
    fake: list[tuple[str, bool]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target, logical = tmp_path / "input", tmp_path / "output", tmp_path / "obs"
    source.write_bytes(INPUTS["b8"])
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: type(usage)(1000, 900, 100))
    # Snapshot needs 64 bytes; each individual output also needs 64. Together they need 128.
    with pytest.raises(OSError, match="free space"):
        _converter().convert_file(
            measurements_filepath=source,
            measurements_format="b8",
            detection_events_filepath=target,
            detection_events_format="b8",
            obs_out_filepath=logical,
            obs_out_format="b8",
        )
    assert not fake and list(tmp_path.iterdir()) == [source]


def test_failed_publication_does_not_remove_caller_replacement(
    fake: list[tuple[str, bool]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target, logical = tmp_path / "input", tmp_path / "output", tmp_path / "obs"
    source.write_bytes(INPUTS["b8"])
    real_link = os.link

    def link(src: str | Path, dst: str | Path) -> None:
        if Path(dst) == logical:
            target.unlink()
            target.write_bytes(b"caller replacement")
            raise OSError("publication refused")
        real_link(src, dst)

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(OSError, match="publication refused"):
        _converter().convert_file(
            measurements_filepath=source,
            measurements_format="b8",
            detection_events_filepath=target,
            detection_events_format="b8",
            obs_out_filepath=logical,
            obs_out_format="b8",
        )
    assert target.read_bytes() == b"caller replacement" and not logical.exists()


def test_cleanup_residue_is_reported(
    fake: list[tuple[str, bool]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target, logical = tmp_path / "input", tmp_path / "output", tmp_path / "obs"
    source.write_bytes(INPUTS["b8"])
    real_link, real_unlink = os.link, Path.unlink

    def link(src: str | Path, dst: str | Path) -> None:
        if Path(dst) == logical:
            raise OSError("second publication failed")
        real_link(src, dst)

    def unlink(path: Path, missing_ok: bool = False) -> None:
        if path == target:
            raise OSError("rollback denied")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(
        OSError, match="cleanup failed.*rollback denied.*second publication failed"
    ):
        _converter().convert_file(
            measurements_filepath=source,
            measurements_format="b8",
            detection_events_filepath=target,
            detection_events_format="b8",
            obs_out_filepath=logical,
            obs_out_format="b8",
        )
    assert target.exists()
    monkeypatch.undo()
    target.unlink()


@pytest.mark.integration
@pytest.mark.parametrize("format", DETECTORS)
def test_terminal_factory_and_zero_observables(
    tmp_path: Path, format: RecordFormat
) -> None:
    pytest.importorskip("stim")
    handle = public.compile_m2d_converter(
        fq.Circuit(1).x(0),
        measurement_qubits=(0,),
        detector_map=MeasurementMap(1, ((0,),)),
        observable_map=MeasurementMap(1, ()),
    )
    source, target, obs = tmp_path / "input", tmp_path / "output", tmp_path / "obs"
    source.write_bytes(b"0\n" * 64)
    handle.convert_file(
        measurements_filepath=source,
        detection_events_filepath=target,
        detection_events_format=format,
        obs_out_filepath=obs,
        obs_out_format=format,
    )
    assert target.read_bytes() == OBSERVABLES[format].replace(b"L0", b"D0")
    empty = {"01": b"\n", "b8": b"", "r8": b"\0", "hits": b"\n", "dets": b"shot\n"}
    assert obs.read_bytes() == empty[format] * 64


@pytest.mark.integration
def test_multiple_upstream_batches_and_dense_agreement(tmp_path: Path) -> None:
    pytest.importorskip("stim")
    handle = _converter()
    source, target, logical = tmp_path / "input", tmp_path / "output", tmp_path / "obs"
    source.write_bytes(b"\x02" * 2051)
    handle.convert_file(
        measurements_filepath=source,
        measurements_format="b8",
        detection_events_filepath=target,
        detection_events_format="b8",
        append_observables=True,
        obs_out_filepath=logical,
        obs_out_format="01",
    )
    assert target.read_bytes() == b"\x05" * 2051
    assert logical.read_bytes() == b"1\n" * 2051
    dets, obs = handle.convert(
        measurements=torch.tensor([[False, True, False]]), separate_observables=True
    )
    assert dets.tolist() == [[True, False]] and obs.tolist() == [[True]]
