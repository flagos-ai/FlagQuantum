"""Verify complete source lifecycle and the existing seeded spin consumer."""

import csv
import json
import lzma
from collections.abc import Iterator
from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest

from flagquantum.simulation._packed_chacha import PackedChaCha, _quarter, _rotate32
from flagquantum.simulation.physical_basis_rng import (
    _mt19937_64,
    packed_chacha8,
    packed_words,
)

pytestmark = pytest.mark.unit


def execute(
    width: int, rounds: int, seed: int, mode: int, count: int
) -> tuple[PackedChaCha, int, list[int]]:
    rng = _mt19937_64(seed)

    def words() -> Iterator[int]:
        while True:
            yield sum(next(rng) << offset for offset in range(0, width, 64))

    draws = words()
    stream = PackedChaCha(width, rounds)
    if mode in (1, 5, 6):
        stream = stream.randomize(draws)
    if mode in (2, 7, 8, 10, 11):
        stream = replace(
            stream,
            state=tuple(next(draws) for _ in range(16)),
            buffer=tuple(next(draws) for _ in range(16)),
        )
    if mode == 2:
        state = list(stream.state)
        state[12] = sum(((1 << 32) - 1) << offset for offset in range(0, width, 32))
        stream = replace(stream, state=tuple(state))
    if mode == 5:
        for _ in range(7):
            _, stream = stream.next_word()
        stream = stream.randomize(draws)
    if mode == 6:
        stream = stream.refill().refill()
    if mode == 7:
        stream = replace(stream, buffer_idx=0)
    if mode == 8:
        stream = replace(stream, buffer_idx=(1 << 64) - 1)
    if mode == 10:
        lanes = []
        for offset in range(0, width, 32):
            lane = [(word >> offset) & ((1 << 32) - 1) for word in stream.state]
            _quarter(lane, seed & 3, (seed >> 2) & 3, (seed >> 4) & 3, (seed >> 6) & 3)
            lanes.append(lane)
        packed_state = tuple(
            sum(lane[k] << (32 * index) for index, lane in enumerate(lanes))
            for k in range(16)
        )
        stream = replace(stream, state=packed_state)
    if mode == 11:
        rotated = sum(
            _rotate32((stream.state[0] >> offset) & ((1 << 32) - 1), count) << offset
            for offset in range(0, width, 32)
        )
        stream = replace(stream, state=(rotated, *stream.state[1:]))
    output = []
    if mode < 10:
        for _ in range(count):
            word, stream = stream.next_word()
            output.append(word)
    return stream, next(rng), output


def test_complete_unchanged_source_lifecycle() -> None:
    fixture = Path(__file__).with_name("data") / "packed_chacha.jsonl.xz"
    total = 0
    with lzma.open(fixture, "rt", encoding="utf-8") as stream:
        for line in stream:
            record: object = json.loads(line)
            assert isinstance(record, list) and len(record) == 2
            case, expected = record
            assert isinstance(case, list) and len(case) == 5
            assert all(type(field) is int for field in case)
            assert isinstance(expected, str)
            actual, next_rng, output = execute(*case)
            fields = expected.split()
            assert actual.buffer_idx == int(fields[0]), case
            assert next_rng == int(fields[1]), case
            assert actual.state == tuple(int(word, 16) for word in fields[2:18]), case
            assert actual.buffer == tuple(int(word, 16) for word in fields[18:34]), case
            assert output == [int(word, 16) for word in fields[34:]], case
            total += 1
    assert total == 3064


@pytest.mark.parametrize("width", [64, 128, 256, 512])
def test_existing_seeded_entry_matches_source_and_draw_ownership(width: int) -> None:
    words = packed_words(width)
    stream = packed_chacha8(width, words)
    after_seed = next(words)
    original = packed_words(width)
    for _ in range(11):
        next(original)
    assert after_seed == next(original)
    expected, _, output = execute(width, 8, 0, 1, 65)
    assert list(islice(stream, 65)) == output
    assert next(words) == next(original)
    assert expected.state[12] == sum(5 << offset for offset in range(0, width, 32))


def test_snapshot_ownership_and_seed_width_adaptation() -> None:
    source = PackedChaCha(64).randomize(iter(range(11)))
    first, successor = source.next_word()
    assert source.buffer_idx == 16 and source.buffer == (0,) * 16
    assert successor.buffer_idx == 1
    assert first == next(source.draws()) == next(source.draws())
    low = packed_chacha8(32, iter([-1] * 11))
    canonical = PackedChaCha(32).randomize(iter([(1 << 32) - 1] * 11)).draws()
    assert list(islice(low, 33)) == list(islice(canonical, 33))
    rounded = packed_chacha8(35, iter(range(11)))
    direct = PackedChaCha(64).randomize(iter(range(11))).draws()
    assert list(islice(rounded, 33)) == list(islice(direct, 33))


def test_insufficient_seed_keeps_original_exception_and_snapshot() -> None:
    snapshot = PackedChaCha(64)
    with pytest.raises(StopIteration):
        snapshot.randomize(iter(range(10)))
    assert snapshot == PackedChaCha(64)
    with pytest.raises(StopIteration):
        packed_chacha8(64, iter(range(10)))


@pytest.mark.parametrize("invalid", [0, 1, 3, -2, True, 2.0])
def test_invalid_rounds_refused(invalid: int) -> None:
    with pytest.raises(ValueError, match="rounds"):
        PackedChaCha(64, invalid)


@pytest.mark.parametrize("invalid", [0, -32, 33, True])
def test_invalid_width_refused(invalid: int) -> None:
    with pytest.raises(ValueError, match="width"):
        PackedChaCha(invalid)


@pytest.mark.parametrize("invalid", [-1, 1 << 64, True])
def test_invalid_cursor_refused(invalid: int) -> None:
    with pytest.raises(ValueError, match="index"):
        PackedChaCha(64, buffer_idx=invalid)


@pytest.mark.parametrize(
    "invalid", [[0] * 16, (0,) * 15, (-1,) + (0,) * 15, (1 << 64,) + (0,) * 15]
)
def test_invalid_state_and_buffer_refused(invalid: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        PackedChaCha(64, state=invalid)
    with pytest.raises(ValueError):
        PackedChaCha(64, buffer=invalid)


@pytest.mark.parametrize("invalid", [-1, 1 << 64, True])
def test_invalid_seed_refused(invalid: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        PackedChaCha(64).randomize(iter([invalid] * 11))


@pytest.mark.parametrize("invalid", [32 - ((1 << 31) - 1) - 1, 1 << 31, True, 1.0])
def test_invalid_rotation_refused(invalid: int) -> None:
    with pytest.raises(ValueError, match="rotation"):
        _rotate32(1, invalid)


def test_complete_defined_sse_rotation_vectors_and_source_provenance() -> None:
    data = json.loads(
        (
            Path(__file__).with_name("data") / "packed_chacha_sse_rotations.json"
        ).read_text()
    )
    inventory = (
        Path(__file__).resolve().parents[2]
        / "docs/development/ftqc_source_inventory.csv"
    )
    rows = {
        row["path"]: row for row in csv.DictReader(inventory.read_text().splitlines())
    }
    assert data["revision"] == "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453"
    assert data["ubsan"] == "no diagnostics"
    assert len(data["frozen_blobs"]) == 2
    for name, digest in data["frozen_blobs"].items():
        assert rows["kickmix/src/" + name]["git_blob"] == digest
    assert set(range(33)) <= set(data["amounts"])
    assert len(data["records"]) == len(data["amounts"]) * 36 == 1548
    for record in data["records"]:
        assert len(record) == 9 and all(type(value) is int for value in record)
        amount, *words = record
        assert [_rotate32(value, amount) for value in words[:4]] == words[4:]


# Original frozen chacha.test.cc vectors, Apache-2.0.
@pytest.mark.parametrize("width", [64, 128, 256, 512])
@pytest.mark.parametrize(
    "rounds, initial, expected",
    [
        (
            8,
            (
                1634760805,
                857760878,
                2036477234,
                1797285236,
                2982244036,
                2024335500,
                3879164530,
                901242752,
                787310623,
                4219030553,
                3787370182,
                1570203558,
                0,
                0,
                3576814106,
                562194639,
            ),
            (
                3025241987,
                2329775405,
                2297722685,
                1975165477,
                2080699436,
                2544175283,
                2541023886,
                3379880763,
                105437290,
                2067886424,
                3862229000,
                2840002177,
                1633451327,
                342241489,
                1475521236,
                2347814823,
            ),
        ),
        (
            20,
            (
                1634760805,
                857760878,
                2036477234,
                1797285236,
                50462976,
                117835012,
                185207048,
                252579084,
                319951120,
                387323156,
                454695192,
                522067228,
                1,
                150994944,
                1241513984,
                0,
            ),
            (
                3840405776,
                358169553,
                534581072,
                3295748259,
                3354710471,
                57196595,
                2594841092,
                1315755203,
                1180992210,
                162176775,
                98026004,
                2718075865,
                3516666549,
                3108902622,
                3900952779,
                1312575650,
            ),
        ),
    ],
)
def test_unchanged_original_vector_scenarios(
    width: int, rounds: int, initial: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    broadcast = sum(1 << offset for offset in range(0, width, 32))
    source = PackedChaCha(width, rounds, tuple(value * broadcast for value in initial))
    output = list(islice(source.draws(), 16))
    assert output == [value * broadcast for value in expected]
    successor = source.refill()
    assert successor.state[12] == (initial[12] + 1) * broadcast
