"""Nested bridge bounds are checked before SDK expansion or compilation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.simulation.stabilizer import engine

pytestmark = pytest.mark.unit


class _Repeat:
    def __init__(self, count: int, body: tuple[Any, ...]) -> None:
        self.repeat_count = count
        self.body = body
        self.copies = 0

    def body_copy(self) -> tuple[Any, ...]:
        self.copies += 1
        return self.body


@pytest.mark.parametrize(
    ("outer", "inner", "expected"), [(2, 3, 6), (1000, 1000, 1_000_000)]
)
def test_nested_instruction_count_is_symbolic(
    outer: int, inner: int, expected: int
) -> None:
    inner_block = _Repeat(inner, (object(),))
    outer_block = _Repeat(outer, (inner_block,))
    assert engine._bounded_stim_expansion((outer_block,)) == expected
    # A million logical instructions still require only two body inspections.
    assert (outer_block.copies, inner_block.copies) == (1, 1)


def test_nested_product_exceeding_limit_is_refused_without_expansion() -> None:
    inner = _Repeat(1001, (object(),))
    outer = _Repeat(1001, (inner,))
    with pytest.raises(CapabilityError, match="1,000,000 expanded instructions"):
        engine._bounded_stim_expansion((outer,))
    assert (outer.copies, inner.copies) == (1, 1)


def test_separate_repeat_blocks_share_the_total_instruction_budget() -> None:
    first = _Repeat(500_000, (object(),))
    second = _Repeat(500_001, (object(),))
    with pytest.raises(CapabilityError, match="1,000,000 expanded instructions"):
        engine._bounded_stim_expansion((first, second))
    assert (first.copies, second.copies) == (1, 1)


@pytest.mark.parametrize(
    "blocks",
    [
        (_Repeat(1001, (_Repeat(1001, (object(),)),)),),
        (_Repeat(500_000, (object(),)), _Repeat(500_001, (object(),))),
    ],
)
def test_over_limit_parser_result_is_refused_before_analysis_or_compilation(
    monkeypatch: pytest.MonkeyPatch, blocks: tuple[_Repeat, ...]
) -> None:
    constructions = 0
    text = "test-owned parser input"

    class GuardedCircuit:
        def __init__(self, source: str) -> None:
            nonlocal constructions
            constructions += 1
            assert source == text

        def __iter__(self) -> Any:
            return iter(blocks)

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"over-limit source reached SDK analysis: {name}")

    monkeypatch.setattr(
        engine, "_stim", lambda: SimpleNamespace(Circuit=GuardedCircuit)
    )
    with pytest.raises(CapabilityError, match="1,000,000 expanded instructions"):
        engine._prepare_stim_experiment(text, seed=53)
    assert constructions == 1
