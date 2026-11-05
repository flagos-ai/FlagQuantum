"""Shared physical budgets and validation for actual comparison-network consumers."""

from .bitwise import _validate_wires
from .less_equal import less_equal_requirements


def comparison_network_requirements(width: int, history_count: int) -> tuple[int, int]:
    """Return one clean comparison pool and stored slots, excluding logical history."""
    if type(history_count) is not int or history_count < 0:
        raise ValueError("Comparison history count must be a nonnegative integer")
    q, c = less_equal_requirements(width, width)
    return (q, c) if history_count else (0, 0)


def validate_comparison_layout(
    words: tuple[tuple[int, ...], ...],
    histories: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    history_count: int,
    adjoint: bool,
) -> None:
    """Validate distinct logical words/history/work and explicit stored classical slots."""
    if type(adjoint) is not bool:
        raise ValueError("Comparison direction must be boolean")
    if not words or not words[0] or any(len(word) != len(words[0]) for word in words):
        raise ValueError("Comparison layout requires equal positive word widths")
    logical_wires = tuple(wire for word in words for wire in word)
    _validate_wires((*logical_wires, *histories, *workspace))
    q, c = comparison_network_requirements(len(words[0]), history_count)
    if len(histories) != history_count:
        raise ValueError(
            "Comparison layout needs one logical history per source comparison"
        )
    if (
        len(workspace) != q
        or len(classical_bits) != c
        or len(set(classical_bits)) != c
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            "Comparison layout needs clean work and distinct classical slots"
        )
