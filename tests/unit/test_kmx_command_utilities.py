"""Verify both consumers of the locked unsigned-word and batch-init helpers."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import pytest
from typing_extensions import NotRequired

from flagquantum.compiler.formats.unsigned_word import (
    format_unsigned_word,
    parse_unsigned_word,
)
from flagquantum.ecosystem.kmx_sample import main
from flagquantum.ecosystem.kmx_sampled_counts import main as count_main


class _InitCase(TypedDict):
    argv: list[str]
    accepted: bool
    header: NotRequired[str]
    row: NotRequired[str]
    source_domain: NotRequired[list[int]]


class _WordCase(TypedDict):
    width: int
    binary: str
    decimal: str
    hex: str


_SOURCE = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_command_utilities.json").read_text()
)
_WORDS: dict[str, _WordCase] = _SOURCE["words"]
_INITIALIZATION: list[_InitCase] = _SOURCE["initialization"]


@pytest.mark.parametrize("literal", _WORDS)
def test_source_unsigned_word_width_and_digits(literal: str) -> None:
    expected = _WORDS[literal]
    value, width = parse_unsigned_word(literal)
    assert width == expected["width"]
    assert format_unsigned_word(value, width, 2) == expected["binary"]
    assert format_unsigned_word(value, width, 10) == expected["decimal"]
    assert format_unsigned_word(value, width, 16) == expected["hex"]


def test_decimal_width_above_python_default_round_trip() -> None:
    value, width = parse_unsigned_word("9" * 4500)
    record = ",".join(
        (
            str(width),
            format_unsigned_word(value, width, 2),
            format_unsigned_word(value, width, 10),
            format_unsigned_word(value, width, 16),
        )
    )
    assert hashlib.sha256(record.encode()).hexdigest() == _SOURCE["large_sha"]


@pytest.mark.parametrize(
    "case", _INITIALIZATION, ids=lambda case: repr(case["argv"][3:])
)
def test_source_overlapping_initialization_and_radix(
    case: _InitCase, tmp_path: Path
) -> None:
    source, report = tmp_path / "input", tmp_path / "report"
    source.write_text(_SOURCE["circuit"])
    report.write_bytes(b"previous")
    common = ["--in", str(source), "--out", str(report)]
    if not case["accepted"]:
        assert main([*common, *case["argv"]]) == 1
        assert count_main([*common, *case["argv"][:4]]) == 1
        assert report.read_bytes() == b"previous"
        return
    assert main([*common, *case["argv"]]) == 0
    header, *rows = report.read_text().splitlines()
    assert len(rows) == 65
    if "row" in case:
        assert header == case["header"]
        assert rows == [case["row"]] * 65
    else:
        expected = case["source_domain"]
        radix = (
            16 if "--hex" in case["argv"] else 10 if "--decimal" in case["argv"] else 2
        )
        for line in rows:
            cells = line.split(",")
            before, after, alias_before, alias_after = (
                int(cell, radix) for cell in cells[:4]
            )
            assert before in expected and before == alias_before
            assert after == alias_after == before ^ (before >> 1)
            assert int(cells[-1]) == 1
