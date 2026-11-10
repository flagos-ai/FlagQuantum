"""Compare fixed prepared samplers to Stim record order and b8."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.stabilizer import measurement_sampler

pytestmark = pytest.mark.integration


def test_feedback_uses_latest_record_for_the_overwritten_slot(tmp_path: Path) -> None:
    stim = pytest.importorskip("stim")
    program = (
        DynamicCircuit(2)
        .x(0)
        .measure(0, classical_bit=0)
        .reset(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .measure(1, classical_bit=1)
    )
    expected = stim.Circuit("X 0\nM 0\nR 0\nM 0\nCX rec[-1] 1\nM 1")
    stale = stim.Circuit("X 0\nM 0\nR 0\nM 0\nCX rec[-2] 1\nM 1")
    sampler = program.compile_sampler(seed=7)
    records = sampler.sample(8)
    assert records.dtype == torch.bool
    assert records.tolist() == expected.compile_sampler(seed=7).sample(8).tolist()
    assert records.tolist() == [[True, False, False]] * 8
    native = fq.Circuit.from_ir(program.to_ir())
    assert type(native) is fq.Circuit
    assert torch.equal(native.compile_sampler(seed=7).sample(8), records)
    assert (
        stale.compile_sampler(seed=7).sample(8, bit_packed=True).tolist() == [[5]] * 8
    )
    path = tmp_path / "feedback.b8"
    sampler.sample_write(8, filepath=path, format="b8")
    assert path.read_bytes() == b"\x01" * 8
    actual = stim.read_shot_data_file(path=str(path), format="b8", num_measurements=3)
    assert actual.tolist() == records.tolist()


def test_inverted_feedback_and_cross_byte_output(tmp_path: Path) -> None:
    stim = pytest.importorskip("stim")
    complement = (
        DynamicCircuit(1)
        .h(0)
        .measure(0)
        .reset(0)
        .conditional("x", 0, classical_bit=0, equals=0)
        .measure(0)
    )
    dense = complement.compile_sampler(seed=7).sample(32)
    assert (dense[:, 0] ^ dense[:, 1]).all()
    assert set(map(tuple, dense.tolist())) == {(True, False), (False, True)}
    program = DynamicCircuit(1).x(0)
    for _ in range(9):
        program.measure(0)
    sampler = program.compile_sampler(seed=7)
    packed = sampler.sample(3, bit_packed=True)
    assert packed.dtype == torch.uint8 and packed.tolist() == [[255, 1]] * 3
    path = tmp_path / "cross-byte.b8"
    sampler.sample_write(3, filepath=path, format="b8")
    assert path.read_bytes() == b"\xff\x01" * 3
    rows = stim.read_shot_data_file(
        path=str(path), format="b8", num_measurements=9, bit_packed=True
    )
    assert rows.tolist() == packed.tolist()


def test_overwrite_feedback_targets_and_b8_invariants(tmp_path: Path) -> None:
    if (
        not Path(measurement_sampler.__file__)
        .with_name("_embedded_stim_profile.txt")
        .is_file()
    ):
        pytest.importorskip("stim")
    program = (
        DynamicCircuit(3)
        .h(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .reset(0)
        .conditional("x", 0, classical_bit=0, equals=0)
        .measure(0, classical_bit=0)
        .conditional("x", 2, classical_bit=0)
        .measure(1)
        .measure(2)
    )
    sampler = program.compile_sampler(seed=7)
    records = sampler.sample(32)
    assert records.shape == (32, 4)
    assert records.dtype == torch.bool
    assert set(map(tuple, records.tolist())) <= {
        (False, True, False, True),
        (True, False, True, False),
    }
    packed = sampler.sample(32, bit_packed=True)
    assert packed.shape == (32, 1)
    assert packed.dtype == torch.uint8
    assert set(packed[:, 0].tolist()) <= {0x05, 0x0A}
    output = tmp_path / "records.b8"
    sampler.sample_write(32, filepath=output, format="b8")
    data = output.read_bytes()
    assert len(data) == 32
    assert set(data) <= {0x05, 0x0A}


_FileFormat = Literal["01", "b8", "r8", "hits", "dets", "ptb64"]
_FORMATS: tuple[_FileFormat, ...] = ("01", "b8", "r8", "hits", "dets", "ptb64")


def _independent_read(data: bytes, format: _FileFormat, width: int) -> list[list[bool]]:
    if format == "01":
        return [[bit == 49 for bit in row] for row in data.splitlines()]
    if format == "b8":
        stride = (width + 7) // 8
        return [
            [bool(data[start + k // 8] & (1 << (k % 8))) for k in range(width)]
            for start in range(0, len(data), stride)
        ]
    if format == "ptb64":
        rows: list[list[bool]] = []
        for start in range(0, len(data), width * 8):
            words = [
                int.from_bytes(data[start + k * 8 : start + (k + 1) * 8], "little")
                for k in range(width)
            ]
            rows.extend(
                [[bool(word & (1 << shot)) for word in words] for shot in range(64)]
            )
        return rows
    if format == "r8":
        result: list[list[bool]] = []
        row: list[bool] = []
        for byte in data:
            row.extend([False] * byte)
            if byte != 255:
                row.append(True)
            if len(row) == width + 1:
                assert row.pop()
                result.append(row)
                row = []
        assert not row
        return result
    result = []
    for line in data.splitlines():
        row = [False] * width
        if format == "hits":
            indices = [int(token) for token in line.split(b",")] if line else []
        else:
            tokens = line.split()
            assert tokens[0] == b"shot"
            assert all(token.startswith(b"M") for token in tokens[1:])
            indices = [int(token[1:]) for token in tokens[1:]]
        for index in indices:
            row[index] = True
        result.append(row)
    return result


@pytest.mark.parametrize("format", _FORMATS)
def test_six_formats_match_handwritten_bytes_and_pinned_sdk(
    tmp_path: Path,
    format: _FileFormat,
) -> None:
    stim = pytest.importorskip("stim")
    assert stim.__version__ == "1.16.0"
    program = (
        DynamicCircuit(1)
        .x(0)
        .measure(0, classical_bit=0)
        .reset(0)
        .measure(0, classical_bit=0)
    )
    native = fq.Circuit.from_ir(program.to_ir())
    expected = {
        "01": b"10\n" * 64,
        "b8": b"\x01" * 64,
        "r8": b"\x00\x01" * 64,
        "hits": b"0\n" * 64,
        "dets": b"shot M0\n" * 64,
        "ptb64": b"\xff" * 8 + b"\x00" * 8,
    }
    actual = tmp_path / "actual"
    reference = tmp_path / "reference"
    native.compile_sampler(seed=7).sample_write(64, filepath=actual, format=format)
    stim.Circuit("X 0\nM 0\nR 0\nM 0").compile_sampler(seed=7).sample_write(
        64, filepath=str(reference), format=format
    )
    assert actual.read_bytes() == reference.read_bytes() == expected[format]
    assert _independent_read(actual.read_bytes(), format, 2) == [[True, False]] * 64


@pytest.mark.parametrize("format", _FORMATS)
def test_format_records_preserve_latest_slot_and_complement_feedback(
    tmp_path: Path,
    format: _FileFormat,
) -> None:
    if (
        not Path(measurement_sampler.__file__)
        .with_name("_embedded_stim_profile.txt")
        .is_file()
    ):
        pytest.importorskip("stim")
    program = (
        DynamicCircuit(3)
        .h(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .reset(0)
        .conditional("x", 0, classical_bit=0, equals=0)
        .measure(0, classical_bit=0)
        .conditional("x", 2, classical_bit=0)
        .measure(1)
        .measure(2)
    )
    path = tmp_path / "feedback"
    fq.Circuit.from_ir(program.to_ir()).compile_sampler(seed=7).sample_write(
        128, filepath=path, format=format
    )
    rows = _independent_read(path.read_bytes(), format, 4)
    assert len(rows) == 128
    assert all(
        row in ([False, True, False, True], [True, False, True, False]) for row in rows
    )


@pytest.mark.parametrize("format", _FORMATS)
def test_formats_preserve_empty_sparse_shots_and_cross_byte_columns(
    tmp_path: Path,
    format: _FileFormat,
) -> None:
    if (
        not Path(measurement_sampler.__file__)
        .with_name("_embedded_stim_profile.txt")
        .is_file()
    ):
        pytest.importorskip("stim")
    for hit in (False, True):
        program = DynamicCircuit(1)
        if hit:
            program.x(0)
        for _ in range(12):
            program.measure(0, classical_bit=0)
        path = tmp_path / str(hit)
        program.compile_sampler(seed=7).sample_write(128, filepath=path, format=format)
        assert _independent_read(path.read_bytes(), format, 12) == [[hit] * 12] * 128
        if format == "b8" and hit:
            assert path.read_bytes() == b"\xff\x0f" * 128


@pytest.mark.parametrize("columns", (254, 255, 256))
def test_r8_long_zero_runs_match_independent_reference(
    tmp_path: Path, columns: int
) -> None:
    if (
        not Path(measurement_sampler.__file__)
        .with_name("_embedded_stim_profile.txt")
        .is_file()
    ):
        pytest.importorskip("stim")
    program = DynamicCircuit(1)
    for _ in range(columns):
        program.measure(0)
    path = tmp_path / "runs"
    program.compile_sampler(seed=7).sample_write(2, filepath=path, format="r8")
    assert (
        path.read_bytes()
        == {254: b"\xfe", 255: b"\xff\x00", 256: b"\xff\x01"}[columns] * 2
    )
    assert (
        _independent_read(path.read_bytes(), "r8", columns) == [[False] * columns] * 2
    )
