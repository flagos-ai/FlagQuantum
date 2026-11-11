"""Captured records reach eager and streaming consumers without an SDK."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec.execution.shot_data import iter_shot_data_file, read_shot_data_file
from flagquantum.simulation.stabilizer import _shot_data as owner

pytestmark = pytest.mark.unit


class FakeReader(ModuleType):
    def __init__(self) -> None:
        super().__init__("fake_shot_reader")
        self.closed = 0
        self.snapshots: list[Path] = []
        self.draws: list[int] = []
        self.rows = [b"\x00", b"\x03", b"\x00", b"\x03"]
        self.malformed = False

    def open_shot_reader(
        self, path: str, format: str, nm: int, nd: int, no: int
    ) -> dict[str, int]:
        self.snapshots.append(Path(path))
        assert Path(path).read_bytes() == b"\x00\x03" * 2
        return {"offset": 0}

    def scan_shot_reader(self, cursor: dict[str, int]) -> int:
        if self.malformed:
            raise ValueError("invalid final record")
        return len(self.rows)

    def read_shot_reader(
        self,
        cursor: dict[str, int],
        rows: int,
        packed: bool,
        separate: bool,
        first: int,
    ) -> tuple[bytearray, bytearray, int]:
        self.draws.append(rows)
        start = cursor["offset"]
        cursor["offset"] += rows
        data = self.rows[start : start + rows]
        if separate:
            return (
                bytearray(row[0] & 1 for row in data),
                bytearray((row[0] >> 1) & 1 for row in data),
                rows,
            )
        if packed:
            return bytearray(b"".join(data)), bytearray(), rows
        return (
            bytearray(bit for row in data for bit in (row[0] & 1, row[0] >> 1)),
            bytearray(),
            rows,
        )

    def close_shot_reader(self, cursor: dict[str, int]) -> None:
        self.closed += 1


@pytest.fixture
def captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, FakeReader]:
    path = tmp_path / "events.b8"
    path.write_bytes(b"\x00\x03" * 2)
    backend = FakeReader()
    monkeypatch.setattr(owner, "_backend", lambda *, streaming: backend)
    return path, backend


def test_same_eager_and_streaming_consumer(captured: tuple[Path, FakeReader]) -> None:
    path, backend = captured
    eager = read_shot_data_file(
        path=path,
        format="b8",
        num_detectors=1,
        num_observables=1,
        separate_observables=True,
        bit_packed=True,
    )
    assert isinstance(eager, tuple)
    with closing(
        iter_shot_data_file(
            path=path,
            format="b8",
            num_detectors=1,
            num_observables=1,
            separate_observables=True,
            bit_packed=True,
            batch_size=3,
        )
    ) as batches:
        parts = list(batches)
    pairs = []
    for part in parts:
        assert isinstance(part, tuple)
        pairs.append(part)
    assert torch.equal(torch.cat([part[0] for part in pairs]), eager[0])
    assert torch.equal(torch.cat([part[1] for part in pairs]), eager[1])
    assert backend.draws == [4, 3, 1]
    assert backend.closed == 4
    assert all(not snapshot.exists() for snapshot in backend.snapshots)


def test_consumer_break_releases_snapshot(captured: tuple[Path, FakeReader]) -> None:
    path, backend = captured
    with closing(
        iter_shot_data_file(path=path, format="b8", num_detectors=2, batch_size=1)
    ) as batches:
        first = next(batches)
        assert isinstance(first, torch.Tensor)
        first.fill_(True)
        second = next(batches)
        assert isinstance(second, torch.Tensor) and second.tolist() == [[True, True]]
    assert backend.closed == 2
    assert all(not snapshot.exists() for snapshot in backend.snapshots)


def test_bad_tail_refuses_before_first_batch(captured: tuple[Path, FakeReader]) -> None:
    path, backend = captured
    backend.malformed = True
    records = iter_shot_data_file(path=path, format="b8", num_detectors=2)
    with pytest.raises(ValueError, match="final record"):
        next(records)
    assert not backend.draws and backend.closed == 1
    assert all(not snapshot.exists() for snapshot in backend.snapshots)


@pytest.mark.parametrize(
    "dimensions,error",
    [
        ((None, None, None), ValueError),
        ((0, 0, 0), CapabilityError),
        ((True, 0, 0), TypeError),
        ((-1, 0, 0), ValueError),
    ],
)
def test_dimensions_refuse_before_io(
    dimensions: tuple[int | None, int | None, int | None], error: type[Exception]
) -> None:
    with pytest.raises(error):
        read_shot_data_file(
            path="missing",
            format="b8",
            num_measurements=dimensions[0],
            num_detectors=dimensions[1],
            num_observables=dimensions[2],
        )


@pytest.mark.parametrize("size", [0, -1, 65537])
def test_invalid_batch_before_io(size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        iter_shot_data_file(
            path="missing", format="b8", num_detectors=1, batch_size=size
        )


def test_budget_refuses_before_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*, streaming: bool) -> ModuleType:
        raise AssertionError("backend must not prepare")

    monkeypatch.setattr(owner, "_backend", forbidden)
    with pytest.raises(CapabilityError, match="byte limit"):
        iter_shot_data_file(
            path="missing", format="b8", num_detectors=2**32, batch_size=65536
        )


def test_sdk_iterator_refuses_before_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(CapabilityError, match="embedded"):
        iter_shot_data_file(path="missing", format="b8", num_detectors=1)


def test_alias_and_owned_dense(captured: tuple[Path, FakeReader]) -> None:
    path, _ = captured
    packed = read_shot_data_file(path=path, format="b8", num_detectors=2, bit_pack=True)
    dense = read_shot_data_file(path=path, format="b8", num_detectors=2)
    assert isinstance(packed, torch.Tensor) and packed.tolist() == [[0], [3], [0], [3]]
    assert (
        isinstance(dense, torch.Tensor)
        and dense.tolist() == [[False, False], [True, True]] * 2
    )


def test_snapshot_unchanged_and_symlink_refused(tmp_path: Path) -> None:
    from flagquantum.simulation.stabilizer._measurement_file_validation import _snapshot

    source, target = tmp_path / "input", tmp_path / "snapshot"
    source.write_bytes(b"records")
    assert _snapshot(source, target) == 7
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        _snapshot(link, tmp_path / "refused")
    assert source.read_bytes() == b"records"


@pytest.mark.parametrize(
    "dimensions,packed,separate,last_allowed",
    [((0, 1, 0), False, False, 8_378_368), ((0, 7, 1), True, True, 5_585_578)],
)
def test_frozen_payload_boundary(
    dimensions: tuple[int, int, int],
    packed: bool,
    separate: bool,
    last_allowed: int,
) -> None:
    # Fixed acceptance vectors: 81,920 scratch bytes, then 8 or 12 bytes/shot.
    # No record table is allocated while checking either side of the limit.
    owner._budget(last_allowed, dimensions, packed, separate)
    with pytest.raises(CapabilityError, match="byte limit"):
        owner._budget(last_allowed + 1, dimensions, packed, separate)


def test_eager_refusal_precedes_output_allocation(
    captured: tuple[Path, FakeReader], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, backend = captured
    monkeypatch.setattr(backend, "scan_shot_reader", lambda cursor: 10_000_000)
    with pytest.raises(CapabilityError, match="byte limit"):
        read_shot_data_file(path=path, format="b8", num_detectors=2)
    assert not backend.draws and backend.closed == 1
    assert all(not snapshot.exists() for snapshot in backend.snapshots)


def test_large_batch_request_uses_actual_remaining_rows(
    captured: tuple[Path, FakeReader], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, backend = captured
    backend.rows = [b"\x00"]

    def zero_rows(
        cursor: object, rows: int, packed: bool, separate: bool, first: int
    ) -> tuple[bytearray, bytearray, int]:
        backend.draws.append(rows)
        return bytearray(rows * 256), bytearray(), rows

    monkeypatch.setattr(backend, "read_shot_reader", zero_rows)
    with closing(
        iter_shot_data_file(path=path, format="b8", num_detectors=256, batch_size=65536)
    ) as batches:
        result = next(batches)
        assert isinstance(result, torch.Tensor) and result.shape == (1, 256)
        with pytest.raises(StopIteration):
            next(batches)
    assert backend.draws == [1] and backend.closed == 2


def test_decode_exception_releases_cursor_and_snapshot(
    captured: tuple[Path, FakeReader],
) -> None:
    path, backend = captured
    with (
        pytest.raises(RuntimeError, match="decoder refused"),
        closing(
            iter_shot_data_file(path=path, format="b8", num_detectors=2, batch_size=1)
        ) as batches,
    ):
        next(batches)
        raise RuntimeError("decoder refused")
    assert backend.closed == 2
    assert all(not snapshot.exists() for snapshot in backend.snapshots)
