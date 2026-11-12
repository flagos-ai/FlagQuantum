"""Closed nested scratch must stay bounded, matched and nonescaping."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.register_profile import register_type
from flagquantum.compiler._hybrid.model import Operation, Value, ValueId
from flagquantum.compiler._hybrid.unitary_effects import (
    inverse_effects,
    unitary_regions,
)

pytestmark = pytest.mark.unit


def nested_scratch(depth: int) -> tuple[Operation, ...]:
    locals_ = [Value(ValueId("local", i), register_type(1)) for i in range(depth)]
    effect = Value(ValueId("effect", 0), register_type(1))
    target = Value(ValueId("target", 0), register_type(1))
    operations: list[Operation] = []
    for value in locals_:
        operations.extend(
            (
                Operation(
                    "qreg.local",
                    results=(value,),
                    attributes={"name": str(value.id.index)},
                ),
                Operation(
                    "qreg.compute",
                    (effect, value, target),
                    attributes={"callee": "predicate"},
                ),
            )
        )
    operations.append(Operation("qreg.phase", (effect,), attributes={"angle": 0.37}))
    for value in reversed(locals_):
        operations.extend(
            (
                Operation("qreg.uncompute", (effect, value)),
                Operation("qreg.release", (effect, value)),
            )
        )
    return tuple(operations)


def test_nested_scratch_inverse_preserves_compute_and_negates_action() -> None:
    operations = nested_scratch(3)
    assert unitary_regions(operations) == (operations,)
    backward = inverse_effects(operations)
    assert backward[:6] == operations[:6]
    assert backward[7:] == operations[7:]
    assert backward[6].attributes["angle"] == -0.37
    assert inverse_effects(backward) == operations


@pytest.mark.parametrize(
    "damage", ["missing_cleanup", "wrong_local", "escape", "depth"]
)
def test_nested_scratch_rejects_missing_mismatched_escaping_or_excessive_lifetimes(
    damage: str,
) -> None:
    operations = list(nested_scratch(2))
    if damage == "missing_cleanup":
        del operations[-4]
    elif damage == "wrong_local":
        operations[-4] = replace(operations[-4], operands=operations[-2].operands)
    elif damage == "escape":
        operations.append(
            Operation(
                "qreg.gate",
                (operations[-1].operands[0], operations[2].results[0]),
                attributes={"gate": "x", "index": 0},
            )
        )
    else:
        operations = list(nested_scratch(34))
    with pytest.raises(ValueError):
        unitary_regions(operations)
