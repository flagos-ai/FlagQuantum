"""Ordinary native composition has a distinct bounded expanded-effect budget."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.codegen.register_expansion import (
    expand_register_calls,
)
from flagquantum.compiler._construction.frontend.program import capture_program
from flagquantum.compiler._construction.frontend.python import capture
from flagquantum.compiler._construction.verification.register_profile import (
    verify_register_program,
)
from flagquantum.compiler._hybrid.model import (
    QUANTUM_EFFECT,
    Operation,
    Region,
    Value,
    ValueId,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python_program import program_definition
from tests.team.compiler.test_construction_register_calls import chain, identity

pytestmark = pytest.mark.unit


def source(effects: int, *, qpe_first: bool = False, qpe_last: bool = False) -> str:
    gates = effects - 1
    bulk, tail = divmod(gates, 256)
    body = "    allocate(q)\n"
    if qpe_first:
        body += "    allocate(p)\n    qpe(lambda: X(q[0]),p)\n"
    body += f"    for j in range({bulk}):\n        hadamard_transform(q)\n"
    if tail:
        body += f"    for i in range({tail}):\n        H(q[i])\n"
    if qpe_last:
        body += "    allocate(p)\n    qpe(lambda: X(q[0]),p)\n"
    ports = "q:Output[QNum[256]]"
    if qpe_first or qpe_last:
        ports += ",p:Output[QNum[1,False,1]]"
    return (
        "from flagquantum.synth import H, X, qpe\n@qfunc\ndef main("
        + ports
        + "):\n"
        + body
    )


@pytest.mark.parametrize("effects", [4096, 4097, 16384])
def test_exact_ordinary_effect_boundaries_and_repeat_isolation(effects: int) -> None:
    definition = program_definition(source(effects))
    program, functions = capture_program(definition)
    assert (
        sum(
            bool(op.results) and op.results[0].type.linear
            for op in program.body.blocks[0].operations
        )
        == effects
    )
    assert verify_register_program(program, functions, max_effects=16384) is program
    assert capture_program(definition) == (program, functions)
    with pytest.raises(HybridVerificationError, match="node budget|4096 effects"):
        verify_register_program(program, functions)


def test_effect_overflow_is_refused_before_returning_graph() -> None:
    with pytest.raises(HybridVerificationError, match="16384 effect"):
        capture_program(program_definition(source(16385)))


@pytest.mark.parametrize("first", [False, True])
def test_qpe_retains_old_whole_program_cap_before_and_after_call(first: bool) -> None:
    with pytest.raises(HybridVerificationError, match="4096"):
        capture_program(
            program_definition(source(4097, qpe_first=first, qpe_last=not first))
        )


def test_static_loop_and_source_limits_are_not_effect_capacity() -> None:
    with pytest.raises(HybridVerificationError, match="256 iterations"):
        capture_program(
            program_definition(
                "@qfunc\ndef main(q:Output[QNum[1]]):\n    allocate(q)\n    for i in range(257):\n        hadamard_transform(q)\n"
            )
        )


@pytest.mark.parametrize("depth,expected", [(13, 4097), (14, None)])
def test_nested_const_helpers_use_the_expanded_effect_guard(
    depth: int, expected: int | None
) -> None:
    root, helpers = chain(depth, repeat=2)
    leaf = helpers[0]
    block = leaf.body.blocks[0]
    effect, word = block.arguments
    following = Value(ValueId("leaf.phase", 0), QUANTUM_EFFECT)
    leaf = replace(
        leaf,
        body=Region(
            (
                replace(
                    block,
                    operations=(
                        Operation(
                            "qreg.phase",
                            (effect, word),
                            (following,),
                            {"callee": "identity"},
                        ),
                        Operation("qreg.return", (following, word)),
                    ),
                ),
            )
        ),
    )
    functions = (capture(identity, {"x": 1}), leaf, *helpers[1:])
    if expected is None:
        with pytest.raises(HybridVerificationError, match="visit budget"):
            verify_register_program(root, functions, max_effects=16384)
        return
    assert verify_register_program(root, functions, max_effects=16384) is root
    expanded = expand_register_calls(root, functions, max_effects=16384)
    effects = sum(
        bool(op.results) and op.results[0].type == QUANTUM_EFFECT
        for op in expanded.body.blocks[0].operations
    )
    assert effects == expected
    with pytest.raises(HybridVerificationError, match="budget"):
        verify_register_program(root, functions)
