"""Recorded mechanisms define an exact XOR oracle across replaceable samplers."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pytest
import torch

from flagquantum.qec import DemError, DetectorErrorModel
from flagquantum.simulation.stabilizer import _dem_sampling as adapter

pytestmark = pytest.mark.integration
_Format = Literal["01", "b8", "r8", "ptb64", "hits", "dets"]
FORMATS: tuple[_Format, ...] = ("01", "b8", "r8", "ptb64", "hits", "dets")


@pytest.fixture(autouse=True)
def _require_backend() -> None:
    if (
        not Path(adapter.__file__).with_name("_embedded_stim_profile.txt").is_file()
        and find_spec("stim") is None
    ):
        pytest.skip("requires the embedded installation or Stim 1.16.0")


def _bytes(rows: list[list[bool]], encoding: _Format, label: str) -> bytes:
    columns = len(rows[0])
    if encoding == "ptb64":
        return b"".join(
            sum(int(rows[start + s][c]) << s for s in range(64)).to_bytes(8, "little")
            for start in range(0, len(rows), 64)
            for c in range(columns)
        )
    output = bytearray()
    for row in rows:
        ones = [i for i, bit in enumerate(row) if bit]
        if encoding == "01":
            output.extend(("".join(str(int(bit)) for bit in row) + "\n").encode())
        elif encoding == "b8":
            output.extend(
                sum(1 << i for i in ones).to_bytes((columns + 7) // 8, "little")
            )
        elif encoding == "r8":
            previous = -1
            for i in [*ones, columns]:
                gap = i - previous - 1
                output.extend(b"\xff" * (gap // 255))
                output.append(gap % 255)
                previous = i
        elif encoding == "hits":
            output.extend((",".join(map(str, ones)) + "\n").encode())
        else:
            output.extend(
                ("shot" + "".join(f" {label}{i}" for i in ones) + "\n").encode()
            )
    return bytes(output)


def _oracle(
    model: DetectorErrorModel, errors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    det = torch.zeros((len(errors), model.num_detectors), dtype=torch.bool)
    obs = torch.zeros((len(errors), model.num_observables), dtype=torch.bool)
    for shot, row in enumerate(errors.tolist()):
        for bit, mechanism in zip(row, model.errors, strict=True):
            if bit:
                for d in mechanism.detectors:
                    det[shot, d] = not det[shot, d]
                for o in mechanism.observables:
                    obs[shot, o] = not obs[shot, o]
    return det, obs


def _pack(bits: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            [
                sum(
                    int(row[c]) << (c % 8)
                    for c in range(b * 8, min(len(row), b * 8 + 8))
                )
                for b in range((len(row) + 7) // 8)
            ]
            for row in bits.tolist()
        ],
        dtype=torch.uint8,
    ).reshape(len(bits), (bits.shape[1] + 7) // 8)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("replay_packed", [False, True])
def test_canonical_columns_duplicates_noncontiguous_replay_and_storage(
    packed: bool, replay_packed: bool
) -> None:
    model = DetectorErrorModel(
        9,
        9,
        (
            DemError(0.8, (8,), (8,)),
            DemError(0.7, (0,), (0,)),
            DemError(0.1, (0,), (0,)),
            DemError(0.0, (1,), (1,)),
        ),
    )
    bits = torch.tensor(
        [[bool((s * 3 + e) % 5 < 2) for e in range(4)] for s in range(1025)]
    )
    expected = _oracle(model, bits)
    replay = _pack(bits) if replay_packed else bits
    if replay_packed:
        replay |= 0xF0  # Unused replay padding is ignored, never echoed.
    backing = torch.zeros((1025, replay.shape[1] * 2), dtype=replay.dtype)
    backing[:, ::2] = replay
    view = backing[:, ::2]
    before = backing.clone()
    sampler = model.compile_sampler(seed=7)
    det, obs, err = sampler.sample(
        1025, bit_packed=packed, return_errors=True, recorded_errors_to_replay=view
    )
    assert torch.equal(det, _pack(expected[0]) if packed else expected[0])
    assert torch.equal(obs, _pack(expected[1]) if packed else expected[1])
    assert err is not None and torch.equal(err, _pack(bits) if packed else bits)
    assert torch.equal(backing, before)
    saved = det.clone()
    sampler.sample(1)
    assert torch.equal(det, saved)


@pytest.mark.parametrize("encoding", FORMATS)
@pytest.mark.parametrize("replay_format", FORMATS)
def test_six_output_and_replay_formats_match_independent_bytes(
    tmp_path: Path, encoding: _Format, replay_format: _Format
) -> None:
    model = DetectorErrorModel(3, 1, (DemError(1.0, (1, 2), (0,)), DemError(0.0, (0,))))
    # Canonical errors are (D0), then (D1,D2,L0).
    bits = [[bool(s % 2), bool(s % 3)] for s in range(64)]
    det, obs = _oracle(model, torch.tensor(bits))
    replay = tmp_path / "input"
    replay.write_bytes(_bytes(bits, replay_format, "M"))
    paths = [tmp_path / name for name in ("d", "o", "e")]
    model.compile_sampler(seed=7).sample_write(
        64,
        det_out_file=paths[0],
        det_out_format=encoding,
        obs_out_file=paths[1],
        obs_out_format=encoding,
        err_out_file=paths[2],
        err_out_format=encoding,
        replay_err_in_file=replay,
        replay_err_in_format=replay_format,
    )
    for path, rows, label in zip(
        paths, (det.tolist(), obs.tolist(), bits), ("D", "L", "M"), strict=True
    ):
        assert path.read_bytes() == _bytes(rows, encoding, label)


@pytest.mark.parametrize("encoding", FORMATS)
def test_deterministic_random_draw_bytes(tmp_path: Path, encoding: _Format) -> None:
    model = DetectorErrorModel(3, 1, (DemError(0.0, (0,)), DemError(1.0, (1, 2), (0,))))
    paths = [tmp_path / name for name in ("d", "o", "e")]
    sampler = model.compile_sampler(seed=17)
    sampler.sample_write(
        64,
        det_out_file=paths[0],
        det_out_format=encoding,
        obs_out_file=paths[1],
        obs_out_format=encoding,
        err_out_file=paths[2],
        err_out_format=encoding,
    )
    for path, row, label in zip(
        paths,
        ([False, True, True], [True], [False, True]),
        ("D", "L", "M"),
        strict=True,
    ):
        assert path.read_bytes() == _bytes([row] * 64, encoding, label)


@pytest.mark.parametrize("encoding", FORMATS)
def test_zero_error_and_observable_columns_with_explicit_shots(
    tmp_path: Path, encoding: _Format
) -> None:
    model = DetectorErrorModel(1, 0)
    sampler = model.compile_sampler(seed=7)
    d, o, e = sampler.sample(
        3,
        return_errors=True,
        recorded_errors_to_replay=torch.empty((3, 0), dtype=torch.bool),
    )
    assert (
        not bool(d.any()) and o.shape == (3, 0) and e is not None and e.shape == (3, 0)
    )
    replay = tmp_path / "input"
    replay.write_bytes(_bytes([[]] * 64, encoding, "M"))
    out = tmp_path / "output"
    sampler.sample_write(
        64,
        det_out_file=out,
        det_out_format=encoding,
        obs_out_file=None,
        replay_err_in_file=replay,
        replay_err_in_format=encoding,
    )
    assert out.read_bytes() == _bytes([[False]] * 64, encoding, "D")


def test_zero_shots_do_not_advance_rng() -> None:
    model = DetectorErrorModel(1, 1, (DemError(0.25, (0,), (0,)),))
    left, right = model.compile_sampler(seed=42), model.compile_sampler(seed=42)
    for packed in (False, True):
        assert left.sample(0, bit_packed=packed, return_errors=True)[0].shape == (0, 1)
    assert torch.equal(left.sample(128)[0], right.sample(128)[0])


@pytest.mark.parametrize("p,limit", [(0.125, 127), (0.25, 167), (0.5, 192)])
def test_fixed_seed_marginal_sanity(p: float, limit: int) -> None:
    model = DetectorErrorModel(1, 1, (DemError(p, (0,), (0,)),))
    d, o, e = model.compile_sampler(seed=1234).sample(4096, return_errors=True)
    assert e is not None and torch.equal(d, o) and torch.equal(d, e)
    assert abs(int(d.sum()) - 4096 * p) <= limit


def test_sdk_replaces_selected_backend_without_consumer_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("stim")
    model = DetectorErrorModel(1, 1, (DemError(1.0, (0,), (0,)),))
    reference = model.compile_sampler().sample(64, bit_packed=True, return_errors=True)
    monkeypatch.setattr(adapter, "prepare", adapter._SdkStream)
    actual = model.compile_sampler().sample(64, bit_packed=True, return_errors=True)
    for left, right in zip(reference, actual, strict=True):
        assert left is not None and right is not None and torch.equal(left, right)
