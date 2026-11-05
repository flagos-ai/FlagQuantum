"""Independent contract fake and primitive truth-table conformance."""

import itertools

import pytest

from flagquantum.core.classical_dataflow import ClassicalXor, parse_classical_xor
from flagquantum.core.ir import CircuitIR
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit


def _reference_xor(
    values: list[int], sources: tuple[int, ...], target: int
) -> list[int]:
    """Reference fake: count old set bits and assign modulo two."""
    result = values.copy()
    result[target] = sum(values[bit] for bit in sources) % 2
    return result


@pytest.mark.parametrize(
    "sources,target",
    [((), 2), ((0,), 2), ((0, 1), 2), ((0, 0, 1), 2), ((0, 1), 0), ((0, 0, 1), 0)],
)
def test_xor_contract_against_independent_fake(
    sources: tuple[int, ...], target: int
) -> None:
    operation = ClassicalXor(sources, target)
    for bits in itertools.product((0, 1), repeat=3):
        values = list(bits)
        result = operation.evaluate(values)
        values[target] = result
        assert values == _reference_xor(list(bits), sources, target)
    instruction = operation.instruction()
    assert parse_classical_xor(instruction) == operation
    ir = CircuitIR(1, (instruction,))
    assert CircuitIR.from_dict(ir.to_dict()).instructions == ir.instructions


def test_undefined_sources_are_not_cancelled_before_validation() -> None:
    with pytest.raises(ValidationError, match="undefined"):
        ClassicalXor((0, 0), 1).evaluate([-1, 0])
    with pytest.raises(ValidationError, match="nonnegative"):
        ClassicalXor((True,), 0)
