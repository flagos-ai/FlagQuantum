"""Constant-place updates preserve the full modular domain without unit repeats."""

import pytest

from flagquantum.compiler._construction.frontend.program import capture_program
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", range(1, 7))
def test_every_admitted_constant_and_basis_matches_independent_modular_sum(
    width: int,
) -> None:
    for offset in range(-256, 257):
        program, _ = capture_program(
            program_definition(
                f"@qfunc\ndef main(q:Output[QNum[{width}]]):\n    allocate(q)\n    q += {offset}\n"
            )
        )
        operations = program.body.blocks[0].operations
        for word in range(1 << width):
            actual = word
            for op in operations:
                if op.name == "qreg.gate" and op.attributes["gate"] == "x":
                    actual ^= 1 << op.attributes["index"]
                elif op.name == "qreg.control_x":
                    indices = op.attributes["indices"]
                    if all(actual & (1 << index) for index in indices[:-1]):
                        actual ^= 1 << indices[-1]
            assert actual == (word + offset) % (1 << width)


def test_power_of_two_offset_leaves_low_bits_out_of_the_gate_support() -> None:
    program, _ = capture_program(
        program_definition(
            "@qfunc\ndef main(q:Output[QNum[16]]):\n    allocate(q)\n    q += 256\n"
        )
    )
    updates = [
        op
        for op in program.body.blocks[0].operations
        if op.name in {"qreg.gate", "qreg.control_x"}
    ]
    assert len(updates) == 8
    for op in updates:
        assert min(op.attributes.get("indices", (op.attributes.get("index", 16),))) >= 8
