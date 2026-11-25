"""Public shot-data writes match fixed Stim 1.16.0 byte witnesses."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pytest
import torch

from flagquantum.qec.execution.shot_data import (
    read_shot_data_file,
    write_shot_data_file,
)
from flagquantum.simulation.stabilizer import _shot_data_write as owner

pytestmark = pytest.mark.integration

_Format = Literal["01", "b8", "r8", "ptb64", "hits", "dets"]


def _reference(
    rows: list[list[bool]], dims: tuple[int, int, int], encoding: _Format
) -> bytes:
    """Encode small fixtures by their documented bit positions and run lengths."""
    nm, nd, no = dims
    columns = sum(dims)
    if encoding == "ptb64":
        return b"".join(
            sum(int(rows[block + shot][column]) << shot for shot in range(64)).to_bytes(
                8, "little"
            )
            for block in range(0, len(rows), 64)
            for column in range(columns)
        )
    output = bytearray()
    for row in rows:
        ones = [bit for bit, value in enumerate(row) if value]
        if encoding == "01":
            output.extend(
                ("".join("1" if value else "0" for value in row) + "\n").encode()
            )
        elif encoding == "b8":
            output.extend(
                sum(1 << bit for bit in ones).to_bytes((columns + 7) // 8, "little")
            )
        elif encoding == "r8":
            previous = -1
            for bit in [*ones, columns]:
                run = bit - previous - 1
                output.extend(b"\xff" * (run // 255))
                output.append(run % 255)
                previous = bit
        elif encoding == "hits":
            output.extend((",".join(str(bit) for bit in ones) + "\n").encode())
        else:
            labels = (
                [f"M{bit}" for bit in range(nm)]
                if nm
                else [
                    *[f"D{bit}" for bit in range(nd)],
                    *[f"L{bit}" for bit in range(no)],
                ]
            )
            output.extend(
                ("shot" + "".join(" " + labels[bit] for bit in ones) + "\n").encode()
            )
    return bytes(output)


@pytest.mark.parametrize("encoding", ["01", "b8", "r8", "ptb64", "hits", "dets"])
@pytest.mark.parametrize("shots", [0, 1, 63, 64, 65, 128])
@pytest.mark.parametrize("dims", [(9, 0, 0), (0, 7, 2), (0, 8, 1), (0, 0, 9)])
@pytest.mark.parametrize("packed", [False, True])
def test_independent_patterns_sections_and_sdk_bytes(
    tmp_path: Path,
    encoding: _Format,
    shots: int,
    dims: tuple[int, int, int],
    packed: bool,
) -> None:
    columns = sum(dims)
    # Include all-zero, all-one and alternating rows across every block boundary.
    rows = [
        [
            (False if shot % 3 == 0 else True if shot % 3 == 1 else bit % 2 == shot % 2)
            for bit in range(columns)
        ]
        for shot in range(shots)
    ]
    dense = torch.tensor(rows, dtype=torch.bool).reshape(shots, columns)
    data = dense
    if packed:
        data = torch.tensor(
            [
                [
                    sum(
                        int(row[bit]) << (bit % 8)
                        for bit in range(byte * 8, min(columns, byte * 8 + 8))
                    )
                    for byte in range((columns + 7) // 8)
                ]
                for row in rows
            ],
            dtype=torch.uint8,
        ).reshape(shots, (columns + 7) // 8)
    nm, nd, no = dims
    path = tmp_path / "actual"
    if encoding == "ptb64" and shots % 64:
        with pytest.raises(ValueError, match="multiple of 64"):
            write_shot_data_file(
                data=data,
                path=path,
                format=encoding,
                num_measurements=nm,
                num_detectors=nd,
                num_observables=no,
            )
        assert not path.exists()
        return
    write_shot_data_file(
        data=data,
        path=path,
        format=encoding,
        num_measurements=nm,
        num_detectors=nd,
        num_observables=no,
    )
    assert path.read_bytes() == _reference(rows, dims, encoding)
    result = read_shot_data_file(
        path=path,
        format=encoding,
        num_measurements=nm,
        num_detectors=nd,
        num_observables=no,
    )
    assert isinstance(result, torch.Tensor) and torch.equal(result, dense)
    if find_spec("stim") is not None:
        import stim

        sdk_path = tmp_path / "sdk"
        stim.write_shot_data_file(
            data=data.numpy(),
            path=sdk_path,
            format=encoding,
            num_measurements=nm,
            num_detectors=nd,
            num_observables=no,
        )
        assert sdk_path.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("zeros", [254, 255, 256])
def test_r8_run_crosses_byte_boundary(tmp_path: Path, zeros: int) -> None:
    rows = [[False] * zeros, [False] * zeros + [True]]
    for index, row in enumerate(rows):
        path = tmp_path / str(index)
        write_shot_data_file(
            data=torch.tensor([row]), path=path, format="r8", num_measurements=len(row)
        )
        assert path.read_bytes() == _reference([row], (len(row), 0, 0), "r8")


def test_packed_file_batches_reach_the_existing_decoder(tmp_path: Path) -> None:
    from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder

    pytest.importorskip("pymatching")
    model = DetectorErrorModel(1, 1, (DemError(0.1, (0,), (0,)),))
    decoder = PyMatchingDecoder.from_detector_error_model(model)
    path = tmp_path / "events"
    write_shot_data_file(
        data=torch.tensor([[0], [3]] * 32, dtype=torch.uint8),
        path=path,
        format="b8",
        num_detectors=1,
        num_observables=1,
    )
    dets_and_obs = read_shot_data_file(
        path=path,
        format="b8",
        num_detectors=1,
        num_observables=1,
        bit_packed=True,
        separate_observables=True,
    )
    assert isinstance(dets_and_obs, tuple)
    events, logicals = dets_and_obs
    for start in range(0, 64, 17):
        assert torch.equal(
            decoder.decode_packed_batch(events[start : start + 17]),
            logicals[start : start + 17],
        )


@pytest.fixture(autouse=True)
def _require_writer() -> None:
    if (
        not Path(owner.__file__).with_name("_embedded_stim_profile.txt").is_file()
        and find_spec("stim") is None
    ):
        pytest.skip("requires Stim SDK or an embedded FlagQuantum installation")


@pytest.mark.parametrize(
    "encoding,expected",
    [
        ("01", b"111\n" * 64),
        ("b8", b"\x07" * 64),
        ("r8", b"\x00\x00\x00\x00" * 64),
        ("ptb64", b"\xff" * 24),
        ("hits", b"0,1,2\n" * 64),
        ("dets", b"shot D0 D1 D2\n" * 64),
    ],
)
def test_all_six_formats_match_independent_bytes(
    tmp_path: Path,
    encoding: Literal["01", "b8", "r8", "ptb64", "hits", "dets"],
    expected: bytes,
) -> None:
    data = torch.ones((64, 3), dtype=torch.bool)
    path = tmp_path / encoding
    write_shot_data_file(
        data=data,
        path=path,
        format=encoding,
        num_detectors=3,
    )
    assert path.read_bytes() == expected


def test_measurement_and_detector_sections_are_written_in_order(tmp_path: Path) -> None:
    rows = torch.tensor([[True, False], [False, True]])
    path = tmp_path / "measurements.01"
    write_shot_data_file(
        data=rows,
        path=path,
        format="01",
        num_measurements=2,
    )
    assert path.read_bytes() == b"10\n01\n"


def test_stim_sdk_can_replace_the_selected_codec_without_changing_the_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = pytest.importorskip("stim")
    monkeypatch.setattr(owner, "_backend", lambda: sdk)
    path = tmp_path / "sdk.b8"
    write_shot_data_file(
        data=torch.tensor([[True, False, True], [False, True, False]]),
        path=path,
        format="b8",
        num_detectors=3,
    )
    assert path.read_bytes() == b"\x05\x02"
