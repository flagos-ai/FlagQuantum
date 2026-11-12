"""A reusable independent DEM produces owned records and replays canonical columns."""

from __future__ import annotations

import inspect
import json
import os
import pickle
from pathlib import Path

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec import DemError, DetectorErrorModel
from flagquantum.qec.execution import dem_sampling as owner
from flagquantum.simulation.stabilizer import _dem_sampling as adapter

pytestmark = pytest.mark.unit


class FakeStream:
    """Apply independently enumerated XOR signatures without any Stim object."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.fail = False

    def sample(
        self,
        shots: int,
        *,
        packed: bool,
        errors: torch.Tensor | None,
        return_errors: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self.calls.append(shots)
        if self.fail:
            raise OSError("controlled sample failure")
        bits = torch.tensor([[False, True]] * shots, dtype=torch.bool)
        if errors is not None:
            bits = (
                errors.bool()
                if errors.dtype == torch.bool
                else torch.stack(
                    [(errors[:, 0] & 1).bool(), ((errors[:, 0] >> 1) & 1).bool()], dim=1
                )
            )
        det = torch.stack((bits[:, 0] ^ bits[:, 1], bits[:, 1]), dim=1)
        obs = bits[:, 1:2].clone()
        if packed:
            det = det[:, :1].to(torch.uint8) | (det[:, 1:2].to(torch.uint8) << 1)
            obs, bits = obs.to(torch.uint8), (
                bits[:, :1].to(torch.uint8) | (bits[:, 1:2].to(torch.uint8) << 1)
            )
        return det, obs, bits if return_errors else None

    def sample_write(
        self,
        shots: int,
        outputs: tuple[str | None, str | None, str | None],
        formats: tuple[str, str, str],
        replay: str | None,
        replay_format: str,
    ) -> None:
        self.calls.append(shots)
        if self.fail:
            raise OSError("controlled file failure")
        for index, path in enumerate(outputs):
            if path is not None:
                assert formats[index] == "b8"
                Path(path).write_bytes(
                    bytes([3 if index == 0 else 1 if index == 1 else 2]) * shots
                )


@pytest.fixture
def prepared(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[owner.CompiledDemSampler, FakeStream]:
    fake = FakeStream()
    monkeypatch.setattr(adapter, "prepare", lambda *args: fake)
    model = DetectorErrorModel(2, 1, (DemError(0.1, (0,)), DemError(0.2, (0, 1), (0,))))
    return model.compile_sampler(seed=7), fake


def test_contract_is_the_approved_addition() -> None:
    contract = json.loads(
        (
            Path(__file__).parents[2] / "contracts/qec-dem-sampling-v1-candidate.json"
        ).read_text()
    )
    assert (
        contract["approval"]["proposal_sha256"]
        == "cfd7b071023e7658501ea2419bbc513d9efdeced495a548036a45a11631aca73"
    )
    for name, method in (
        ("compile_sampler", DetectorErrorModel.compile_sampler),
        ("sample", owner.CompiledDemSampler.sample),
        ("sample_write", owner.CompiledDemSampler.sample_write),
    ):
        assert list(inspect.signature(method).parameters) == contract["methods"][name]
    assert owner.__all__ == ("CompiledDemSampler",)


def test_batching_and_owned_results(
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
) -> None:
    sampler, fake = prepared
    det, obs, errors = sampler.sample(1025, bit_packed=True, return_errors=True)
    assert fake.calls == [1024, 1]
    assert det.tolist() == [[3]] * 1025 and obs.tolist() == [[1]] * 1025
    assert errors is not None and errors.tolist() == [[2]] * 1025
    det.zero_()
    fresh, _, _ = sampler.sample(1, bit_packed=True)
    assert fresh.tolist() == [[3]]


@pytest.mark.parametrize("packed_input", [False, True])
@pytest.mark.parametrize("packed_output", [False, True])
def test_replay_has_independent_input_and_output_packing(
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
    packed_input: bool,
    packed_output: bool,
) -> None:
    sampler, _ = prepared
    source = (
        torch.tensor([[1], [2], [3]], dtype=torch.uint8)
        if packed_input
        else torch.tensor([[True, False], [False, True], [True, True]])
    )
    before = source.clone()
    det, obs, err = sampler.sample(
        3,
        bit_packed=packed_output,
        return_errors=True,
        recorded_errors_to_replay=source,
    )
    expected = (
        [[1], [3], [2]]
        if packed_output
        else [[True, False], [True, True], [False, True]]
    )
    assert det.tolist() == expected and obs.tolist() == [[0], [1], [1]]
    assert err is not None and torch.equal(source, before)


@pytest.mark.parametrize(
    "shots,error", [(True, TypeError), (-1, ValueError), (1.0, TypeError)]
)
def test_invalid_shots_do_not_consume_rng(
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
    shots: int,
    error: type[Exception],
) -> None:
    sampler, fake = prepared
    with pytest.raises(error):
        sampler.sample(shots)
    assert not fake.calls
    assert sampler.sample(0)[0].shape == (0, 2) and not fake.calls


def test_error_ids_refuse_before_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object) -> FakeStream:
        raise AssertionError("engine must not prepare")

    monkeypatch.setattr(adapter, "prepare", forbidden)
    model = DetectorErrorModel(1, 0, (DemError(0.1, (0,), error_id=7),))
    with pytest.raises(ValueError, match="error ids"):
        model.compile_sampler()
    assert model.dem_sampling(shots=1).detectors.shape == (1, 1)


def test_execution_failure_invalidates(
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
) -> None:
    sampler, fake = prepared
    fake.fail = True
    with pytest.raises(OSError, match="controlled"):
        sampler.sample(1)
    with pytest.raises(RuntimeError, match="invalidated"):
        sampler.sample(1)


def test_handle_cannot_be_constructed_or_pickled(
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
) -> None:
    with pytest.raises(TypeError, match="compile_sampler"):
        owner.CompiledDemSampler()
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(prepared[0])


def test_files_preserve_existing_destinations(
    prepared: tuple[owner.CompiledDemSampler, FakeStream], tmp_path: Path
) -> None:
    sampler, fake = prepared
    path = tmp_path / "events"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        sampler.sample_write(
            64, det_out_file=path, obs_out_file=None, det_out_format="b8"
        )
    assert path.read_bytes() == b"original" and not fake.calls


def test_three_files_publish_independent_streams(
    prepared: tuple[owner.CompiledDemSampler, FakeStream], tmp_path: Path
) -> None:
    sampler, _ = prepared
    paths = [tmp_path / name for name in ("d", "o", "e")]
    sampler.sample_write(
        64,
        det_out_file=paths[0],
        obs_out_file=paths[1],
        err_out_file=paths[2],
        det_out_format="b8",
        obs_out_format="b8",
        err_out_format="b8",
    )
    assert [p.read_bytes() for p in paths] == [b"\x03" * 64, b"\x01" * 64, b"\x02" * 64]


def test_budget_refuses_before_sampling(
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
) -> None:
    sampler, fake = prepared
    with pytest.raises(CapabilityError, match="64 MiB"):
        sampler.sample(100_000_000)
    assert not fake.calls


def test_publication_race_rolls_back_only_owned_inodes(
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler, _ = prepared
    d, o = tmp_path / "d", tmp_path / "o"
    link = os.link

    def race(source: Path, target: Path) -> None:
        if target == o:
            o.write_bytes(b"concurrent owner")
        link(source, target)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(FileExistsError):
        sampler.sample_write(
            64, det_out_file=d, det_out_format="b8", obs_out_file=o, obs_out_format="b8"
        )
    assert not d.exists() and o.read_bytes() == b"concurrent owner"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["o"]
    with pytest.raises(RuntimeError, match="invalidated"):
        sampler.sample(1)


def test_bad_replay_file_refuses_before_execution(
    prepared: tuple[owner.CompiledDemSampler, FakeStream], tmp_path: Path
) -> None:
    sampler, fake = prepared
    replay, target = tmp_path / "input", tmp_path / "d"
    replay.write_bytes(b"\x80" * 64)
    with pytest.raises(OSError):
        sampler.sample_write(
            64,
            det_out_file=target,
            det_out_format="b8",
            obs_out_file=None,
            replay_err_in_file=replay,
            replay_err_in_format="b8",
        )
    assert not fake.calls and not target.exists()
    sampler.sample(1)


def test_zero_shot_files_never_execute(
    prepared: tuple[owner.CompiledDemSampler, FakeStream], tmp_path: Path
) -> None:
    sampler, fake = prepared
    path = tmp_path / "d"
    sampler.sample_write(0, det_out_file=path, obs_out_file=None)
    assert path.read_bytes() == b"" and not fake.calls


def test_file_execution_failure_cleans_staging(
    prepared: tuple[owner.CompiledDemSampler, FakeStream], tmp_path: Path
) -> None:
    sampler, fake = prepared
    fake.fail = True
    with pytest.raises(OSError, match="controlled"):
        sampler.sample_write(
            64, det_out_file=tmp_path / "d", obs_out_file=None, det_out_format="b8"
        )
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="invalidated"):
        sampler.sample(1)


def test_owner_thread_and_process_are_enforced(
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    sampler, fake = prepared
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(sampler.sample, 1)
        with pytest.raises(RuntimeError, match="thread and process"):
            future.result(timeout=10)
    sampler._pid -= 1
    with pytest.raises(RuntimeError, match="thread and process"):
        sampler.sample(1)
    assert not fake.calls


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_seed_range_is_checked_before_preparation(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object) -> FakeStream:
        raise AssertionError("must validate seed before preparing")

    monkeypatch.setattr(adapter, "prepare", forbidden)
    with pytest.raises(ValueError, match="range"):
        DetectorErrorModel(1, 0).compile_sampler(seed=seed)


def test_dimensions_and_output_bytes_refuse_before_engine(
    monkeypatch: pytest.MonkeyPatch,
    prepared: tuple[owner.CompiledDemSampler, FakeStream],
    tmp_path: Path,
) -> None:
    def forbidden(*args: object) -> FakeStream:
        raise AssertionError("must validate dimensions before preparing")

    monkeypatch.setattr(adapter, "prepare", forbidden)
    with pytest.raises(CapabilityError, match="dimensions"):
        DetectorErrorModel(65537, 0).compile_sampler()
    sampler, fake = prepared
    with pytest.raises(CapabilityError, match="1 GiB"):
        sampler.sample_write(2**30, det_out_file=tmp_path / "d", obs_out_file=None)
    assert not fake.calls
