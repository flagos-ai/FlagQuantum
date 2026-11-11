"""Independent record bytes and interchangeable SDK/embedded eager consumers."""

from __future__ import annotations

from contextlib import closing
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pytest
import torch

from flagquantum.qec.execution.shot_data import iter_shot_data_file, read_shot_data_file
from flagquantum.simulation.stabilizer import _shot_data as owner

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _require_record_codec() -> None:
    if (
        not Path(owner.__file__).with_name("_embedded_stim_profile.txt").is_file()
        and find_spec("stim") is None
    ):
        pytest.skip("requires Stim SDK or an embedded FlagQuantum installation")


@pytest.mark.parametrize("encoding", ["01", "b8", "r8", "ptb64", "hits", "dets"])
@pytest.mark.parametrize("packed", [False, True])
def test_six_formats_and_independent_separation(
    tmp_path: Path,
    encoding: Literal["01", "b8", "r8", "ptb64", "hits", "dets"],
    packed: bool,
) -> None:
    # D6 and L1 share a byte in combined data, but not in separated output.
    data = {
        "01": b"000000101\n" * 64,
        "b8": b"\x40\x01" * 64,
        "r8": b"\x06\x01\x00" * 64,
        "ptb64": b"\0" * 48 + b"\xff" * 8 + b"\0" * 8 + b"\xff" * 8,
        "hits": b"6,8\n" * 64,
        "dets": b"shot D6 L1\n" * 64,
    }[encoding]
    path = tmp_path / encoding
    path.write_bytes(data)
    result = read_shot_data_file(
        path=path,
        format=encoding,
        num_detectors=7,
        num_observables=2,
        separate_observables=True,
        bit_packed=packed,
    )
    assert isinstance(result, tuple)
    dets, obs = result
    assert dets.dtype == obs.dtype == (torch.uint8 if packed else torch.bool)
    assert dets.tolist() == ([[64]] if packed else [[False] * 6 + [True]]) * 64
    assert obs.tolist() == ([[2]] if packed else [[False, True]]) * 64


@pytest.mark.parametrize(
    "encoding,data",
    [("01", b"10\r\n"), ("hits", b"0\n"), ("dets", b"\n\t  shot D0\n\n")],
)
def test_noncanonical_valid_upstream_text(
    tmp_path: Path, encoding: Literal["01", "hits", "dets"], data: bytes
) -> None:
    path = tmp_path / "data"
    path.write_bytes(data)
    actual = read_shot_data_file(path=path, format=encoding, num_detectors=2)
    assert isinstance(actual, torch.Tensor) and actual.tolist() == [[True, False]]


def test_typed_measurement_detector_logical_columns(tmp_path: Path) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"shot M1 D0 L1\n")
    actual = read_shot_data_file(
        path=path, format="dets", num_measurements=2, num_detectors=1, num_observables=2
    )
    assert isinstance(actual, torch.Tensor)
    assert actual.tolist() == [[False, True, True, False, True]]


def test_sdk_numerical_replacement_with_same_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = pytest.importorskip("stim")
    path = tmp_path / "data"
    path.write_bytes(b"\x00\x01")
    original = read_shot_data_file(path=path, format="b8", num_detectors=1)
    monkeypatch.setattr(owner, "_backend", lambda *, streaming: sdk)
    replaced = read_shot_data_file(path=path, format="b8", num_detectors=1)
    assert isinstance(original, torch.Tensor) and isinstance(replaced, torch.Tensor)
    assert torch.equal(original, replaced)


@pytest.mark.parametrize("encoding", ["b8", "01", "r8", "hits", "dets", "ptb64"])
def test_positive_width_empty_files(
    tmp_path: Path, encoding: Literal["b8", "01", "r8", "hits", "dets", "ptb64"]
) -> None:
    path = tmp_path / "empty"
    path.touch()
    result = read_shot_data_file(
        path=path,
        format=encoding,
        num_detectors=1,
        num_observables=0,
        separate_observables=True,
    )
    assert isinstance(result, tuple)
    assert result[0].shape == (0, 1) and result[1].shape == (0, 0)


@pytest.mark.parametrize(
    "data,encoding",
    [(b"\x80", "b8"), (b"\x00", "ptb64"), (b"1", "01"), (b"\xff", "r8")],
)
def test_malformed_framing_or_padding(
    tmp_path: Path, data: bytes, encoding: Literal["b8", "ptb64", "01", "r8"]
) -> None:
    path = tmp_path / "bad"
    path.write_bytes(data)
    with pytest.raises((ValueError, OSError)):
        read_shot_data_file(path=path, format=encoding, num_detectors=1)


@pytest.mark.parametrize("size", [1, 17, 63, 64, 65, 1024, 65536])
def test_embedded_ptb64_arbitrary_consumer_batches(tmp_path: Path, size: int) -> None:
    if not Path(owner.__file__).with_name("_embedded_stim_profile.txt").is_file():
        pytest.skip(
            "embedded installation required; covered by no-SDK installed acceptance"
        )
    path = tmp_path / "records"
    path.write_bytes((b"\xaa" * 8 + b"\xaa" * 8) * 2)
    outputs = []
    with closing(
        iter_shot_data_file(
            path=path,
            format="ptb64",
            num_detectors=1,
            num_observables=1,
            separate_observables=True,
            bit_packed=True,
            batch_size=size,
        )
    ) as batches:
        for result in batches:
            assert isinstance(result, tuple)
            assert torch.equal(result[0], result[1])
            outputs.extend(result[0].flatten().tolist())
    assert outputs == [0, 1] * 64
