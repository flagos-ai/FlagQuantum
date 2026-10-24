"""Keep verification linear for flat code while preserving region capture scope."""

from __future__ import annotations

from typing import Any

import pytest

from flagquantum.compiler._hybrid import (
    BOOL,
    Block,
    HybridProgram,
    HybridVerificationError,
    IRType,
    Operation,
    Region,
    Value,
    ValueId,
    scalar_type,
    verify_program,
)

pytestmark = pytest.mark.unit


def _flat_program(size: int) -> HybridProgram:
    values = [Value(ValueId("entry", i), scalar_type("float64")) for i in range(size)]
    operations = tuple(
        Operation("arith.constant", results=(value,), attributes={"value": 1.0})
        for value in values
    ) + (
        Operation("program.return", operands=(values[-1],)),
    )
    return HybridProgram("flat", Region((Block(operations=operations),)))


def test_flat_program_type_visits_grow_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = IRType.__getattribute__

    def visits(size: int) -> int:
        program = _flat_program(size)
        count = 0

        def observe(self: IRType, name: str) -> Any:
            nonlocal count
            if name == "linear":
                count += 1
            return original(self, name)

        with monkeypatch.context() as patch:
            patch.setattr(IRType, "__getattribute__", observe)
            assert verify_program(program) is program
        return count

    small, large = visits(64), visits(128)
    # Count deterministic type inspections, not wall time or a speedup target.
    assert small > 0
    assert large <= 3 * small, (small, large)


@pytest.mark.parametrize("capture_own_result", [False, True])
def test_region_captures_prior_values_but_not_its_parent_result(
    capture_own_result: bool,
) -> None:
    scalar = scalar_type("float64")
    condition = Value(ValueId("entry", 0), BOOL)
    carried = Value(ValueId("entry", 1), scalar)
    prior = Value(ValueId("entry", 2), scalar)
    joined = Value(ValueId("entry", 3), scalar)
    regions = []
    for scope in ("then", "else"):
        argument = Value(ValueId(scope, 0), scalar)
        result = Value(ValueId(scope, 1), scalar)
        regions.append(
            Region(
                (
                    Block(
                        arguments=(argument,),
                        operations=(
                            Operation(
                                "arith.add",
                                operands=(
                                    argument,
                                    joined if capture_own_result else prior,
                                ),
                                results=(result,),
                            ),
                            Operation("scf.yield", operands=(result,)),
                        ),
                    ),
                )
            )
        )
    program = HybridProgram(
        "branch_capture",
        Region(
            (
                Block(
                    arguments=(condition, carried, prior),
                    operations=(
                        Operation(
                            "scf.if",
                            operands=(condition, carried),
                            results=(joined,),
                            regions=tuple(regions),
                        ),
                        Operation("program.return", operands=(joined,)),
                    ),
                ),
            )
        ),
    )
    if capture_own_result:
        with pytest.raises(HybridVerificationError) as caught:
            verify_program(program)
        assert [d.code for d in caught.value.diagnostics] == [
            "ssa.use_before_definition",
            "ssa.use_before_definition",
        ]
    else:
        assert verify_program(program) is program
