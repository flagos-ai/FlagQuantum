"""Verify complete striped/paired state against the original packed counter."""

import json
import lzma
from pathlib import Path

import pytest

from flagquantum.core._population_counter import PopulationCounter
from flagquantum.core._word_operations import WORD_MASK, add_carry
from flagquantum.simulation.physical_basis_rng import _mt19937_64

pytestmark = pytest.mark.unit


def run_counter(
    width: int, seed: int, repeats: int, mode: int
) -> tuple[PopulationCounter, int]:
    counter = PopulationCounter(width)
    mask = 0
    full = (1 << width) - 1
    if mode == 4:
        counter = PopulationCounter(
            width, (0,) * 5 + (full,), (0, full) + (0,) * 30, (1 << 32) - 1, 255
        )
    if mode == 5:
        initial = sum(0x12345678 << bit for bit in range(0, width, 32))
        counter = PopulationCounter(width, (initial,) + (0,) * 5)
    draws = _mt19937_64(seed)
    for _ in range(repeats):
        if mode == 1:
            mask = full
        elif mode == 2:
            mask = sum(next(draws) << bit for bit in range(0, width, 64))
        elif mode == 3:
            mask ^= sum(0x11111111 << bit for bit in range(0, width, 32))
            increment = 0xBCDEF012BCDEF012
            mask = sum(
                add_carry(False, (mask >> bit) & WORD_MASK, increment)[0] << bit
                for bit in range(0, width, 64)
            )
        elif mode == 5:
            mask = counter.bit_totals[0]
        counter = counter.masked_increments(mask)
    return counter, next(draws)


def test_complete_original_counter_states() -> None:
    fixture = Path(__file__).with_name("data") / "population_counter.jsonl.xz"
    cache: dict[tuple[int, ...], tuple[PopulationCounter, int]] = {}
    count = 0
    with lzma.open(fixture, "rt", encoding="utf-8") as stream:
        for line in stream:
            record: object = json.loads(line)
            assert isinstance(record, list) and len(record) == 2
            raw, expected = record
            assert isinstance(raw, list) and len(raw) == 5
            assert all(type(field) is int for field in raw)
            assert isinstance(expected, str)
            width, seed, repeats, mode, clear = raw
            key = (width, seed, repeats, mode)
            if key not in cache:
                cache[key] = run_counter(width, seed, repeats, mode)
            counter, next_word = cache[key]
            if clear:
                counter = counter.clear()
            fields = expected.split()
            assert counter.step == int(fields[0]), raw
            assert counter.shared_count == int(fields[1]), raw
            assert next_word == int(fields[2]), raw
            assert counter.bit_totals == tuple(
                int(value, 16) for value in fields[3:9]
            ), raw
            assert counter.paired_totals == tuple(
                int(value, 16) for value in fields[9:41]
            ), raw
            assert tuple(counter.compute_total(k) for k in range(width)) == tuple(
                map(int, fields[41:])
            ), raw
            count += 1
    assert count == 960


@pytest.mark.parametrize("width", [64, 128, 256, 512])
def test_independent_lane_counts_and_prior_snapshot(width: int) -> None:
    value = PopulationCounter(width)
    original = value
    draws = _mt19937_64(1084)
    expected = [0] * width
    for _ in range(500):
        mask = sum(next(draws) << bit for bit in range(0, width, 64))
        value = value.masked_increments(mask)
        for bit in range(width):
            expected[bit] += (mask >> bit) & 1
    assert [value.compute_total(k) for k in range(width)] == expected
    assert original == PopulationCounter(width)
    assert value.clear() == original


@pytest.mark.parametrize("invalid", [0, True])
def test_invalid_width_refused(invalid: int) -> None:
    with pytest.raises(ValueError):
        PopulationCounter(invalid)


@pytest.mark.parametrize("invalid", [[0] * 6, (0,) * 5, (-1,) + (0,) * 5])
def test_invalid_striped_storage_refused(invalid: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        PopulationCounter(64, bit_totals=invalid)


@pytest.mark.parametrize("invalid", [[0] * 32, (0,) * 31, (1 << 64,) + (0,) * 31])
def test_invalid_paired_storage_refused(invalid: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        PopulationCounter(64, paired_totals=invalid)


@pytest.mark.parametrize("invalid", [-1, 1 << 32, True])
def test_invalid_shared_count_refused(invalid: int) -> None:
    with pytest.raises(ValueError):
        PopulationCounter(64, shared_count=invalid)


@pytest.mark.parametrize("invalid", [-1, 256, True])
def test_invalid_step_refused(invalid: int) -> None:
    with pytest.raises(ValueError):
        PopulationCounter(64, step=invalid)


@pytest.mark.parametrize("invalid", [-1, 64, True])
def test_invalid_index_refused(invalid: int) -> None:
    with pytest.raises(ValueError, match="index"):
        PopulationCounter(64).compute_total(invalid)


@pytest.mark.parametrize("invalid", [-1, 1 << 64, True])
def test_invalid_mask_refused(invalid: int) -> None:
    with pytest.raises(ValueError, match="mask"):
        PopulationCounter(64).masked_increments(invalid)
