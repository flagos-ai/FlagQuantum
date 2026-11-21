"""Ordinary native composition has a distinct bounded expanded-effect budget."""

import pytest

from flagquantum.compiler._construction.frontend.program import capture_program
from flagquantum.compiler._construction.verification.register_profile import (
    verify_register_program,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python_program import program_definition

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
