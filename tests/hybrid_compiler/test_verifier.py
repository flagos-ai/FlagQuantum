from __future__ import annotations

import pytest

from flagquantum.compiler._hybrid import (
    BOOL,
    INDEX,
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    HybridVerificationError,
    Operation,
    Region,
    Value,
    ValueId,
    scalar_type,
    verify_program,
)

pytestmark = pytest.mark.unit
F64 = scalar_type("float64")


def value(scope: str, index: int, value_type) -> Value:
    return Value(ValueId(scope, index), value_type)


def codes(error: HybridVerificationError) -> set[str]:
    return {diagnostic.code for diagnostic in error.diagnostics}


def test_unknown_operation_fails_closed() -> None:
    result = value("entry", 0, F64)
    program = HybridProgram(
        "unknown",
        Region(
            (
                Block(
                    operations=(
                        Operation("foreign.call", results=(result,)),
                        Operation("program.return", operands=(result,)),
                    )
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "operation.unknown" in codes(caught.value)


def test_use_before_definition_is_rejected() -> None:
    missing = value("entry", 0, F64)
    program = HybridProgram(
        "undefined",
        Region(
            (Block(operations=(Operation("program.return", operands=(missing,)),)),)
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "ssa.use_before_definition" in codes(caught.value)


def test_duplicate_definition_is_rejected() -> None:
    result = value("entry", 0, F64)
    program = HybridProgram(
        "duplicate",
        Region(
            (
                Block(
                    arguments=(result,),
                    operations=(
                        Operation(
                            "arith.constant",
                            results=(result,),
                            attributes={"value": 1.0},
                        ),
                        Operation("program.return", operands=(result,)),
                    ),
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "ssa.duplicate_definition" in codes(caught.value)


def test_use_type_must_match_definition_type() -> None:
    result = value("entry", 0, F64)
    mistyped_use = value("entry", 0, BOOL)
    program = HybridProgram(
        "mistyped",
        Region(
            (
                Block(
                    arguments=(result,),
                    operations=(Operation("program.return", operands=(mistyped_use,)),),
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "ssa.use_type" in codes(caught.value)


def test_linear_quantum_effect_cannot_be_reused() -> None:
    effect = value("entry", 0, QUANTUM_EFFECT)
    first = value("entry", 1, F64)
    second = value("entry", 2, F64)
    program = HybridProgram(
        "reuse",
        Region(
            (
                Block(
                    arguments=(effect,),
                    operations=(
                        Operation(
                            "quantum.expectation",
                            operands=(effect,),
                            results=(first,),
                            attributes={"terms": (("z", 0),)},
                        ),
                        Operation(
                            "quantum.expectation",
                            operands=(effect,),
                            results=(second,),
                            attributes={"terms": (("z", 0),)},
                        ),
                        Operation("program.return", operands=(second,)),
                    ),
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "effect.reuse" in codes(caught.value)


def test_unknown_expectation_axis_fails_before_lowering() -> None:
    effect = value("entry", 0, QUANTUM_EFFECT)
    expectation = value("entry", 1, F64)
    program = HybridProgram(
        "unknown_axis",
        Region(
            (
                Block(
                    arguments=(effect,),
                    operations=(
                        Operation(
                            "quantum.expectation",
                            operands=(effect,),
                            results=(expectation,),
                            attributes={"terms": (("unknown", 0),)},
                        ),
                        Operation("program.return", operands=(expectation,)),
                    ),
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "quantum.observable" in codes(caught.value)


def test_branch_argument_signature_must_match_carried_effect() -> None:
    condition = value("entry", 0, BOOL)
    effect = value("entry", 1, QUANTUM_EFFECT)
    branch_effect = value("then", 0, QUANTUM_EFFECT)
    wrong_argument = value("else", 0, F64)
    joined = value("entry", 2, QUANTUM_EFFECT)
    expectation = value("entry", 3, F64)
    then_region = Region(
        (
            Block(
                arguments=(branch_effect,),
                operations=(Operation("scf.yield", operands=(branch_effect,)),),
            ),
        )
    )
    else_region = Region(
        (
            Block(
                arguments=(wrong_argument,),
                operations=(Operation("scf.yield", operands=(wrong_argument,)),),
            ),
        )
    )
    program = HybridProgram(
        "bad_branch",
        Region(
            (
                Block(
                    arguments=(condition, effect),
                    operations=(
                        Operation(
                            "scf.if",
                            operands=(condition, effect),
                            results=(joined,),
                            regions=(then_region, else_region),
                        ),
                        Operation(
                            "quantum.expectation",
                            operands=(joined,),
                            results=(expectation,),
                            attributes={"terms": (("z", 0),)},
                        ),
                        Operation("program.return", operands=(expectation,)),
                    ),
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "scf.if_arguments" in codes(caught.value)
    assert "scf.yield_signature" in codes(caught.value)


def test_loop_carried_signature_must_match_result() -> None:
    lower = value("entry", 0, INDEX)
    upper = value("entry", 1, INDEX)
    step = value("entry", 2, INDEX)
    effect = value("entry", 3, QUANTUM_EFFECT)
    result = value("entry", 4, F64)
    iteration = value("loop", 0, INDEX)
    loop_effect = value("loop", 1, QUANTUM_EFFECT)
    body = Region(
        (
            Block(
                arguments=(iteration, loop_effect),
                operations=(Operation("scf.yield", operands=(loop_effect,)),),
            ),
        )
    )
    program = HybridProgram(
        "bad_loop",
        Region(
            (
                Block(
                    arguments=(lower, upper, step, effect),
                    operations=(
                        Operation(
                            "scf.for",
                            operands=(lower, upper, step, effect),
                            results=(result,),
                            regions=(body,),
                        ),
                        Operation("program.return", operands=(result,)),
                    ),
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "scf.for_result" in codes(caught.value)
    assert "scf.yield_signature" in codes(caught.value)


def test_linear_effect_cannot_be_dropped() -> None:
    effect = value("entry", 0, QUANTUM_EFFECT)
    result = value("entry", 1, F64)
    program = HybridProgram(
        "dropped",
        Region(
            (
                Block(
                    arguments=(effect,),
                    operations=(
                        Operation(
                            "arith.constant",
                            results=(result,),
                            attributes={"value": 1.0},
                        ),
                        Operation("program.return", operands=(result,)),
                    ),
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "effect.dropped" in codes(caught.value)


def test_unknown_attributes_fail_closed() -> None:
    result = value("entry", 0, F64)
    program = HybridProgram(
        "attribute",
        Region(
            (
                Block(
                    operations=(
                        Operation(
                            "arith.constant",
                            results=(result,),
                            attributes={"value": 1.0, "unchecked": True},
                        ),
                        Operation("program.return", operands=(result,)),
                    )
                ),
            )
        ),
    )

    with pytest.raises(HybridVerificationError) as caught:
        verify_program(program)

    assert "operation.attribute_unknown" in codes(caught.value)
