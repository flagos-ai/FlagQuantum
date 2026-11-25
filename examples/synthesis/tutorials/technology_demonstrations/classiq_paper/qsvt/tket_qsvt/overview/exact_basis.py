"""Caller-owned operator-exact S/SDG rules for a CX/U3 target."""

import math
from collections.abc import Mapping
from dataclasses import replace

from flagquantum.compiler.basis_translation import (
    EquivalenceRule,
    with_equivalence_rule,
)
from flagquantum.core.ir import CircuitIR, Instruction

from ...classiq_qsvt.overview.exact_basis import exact_rules as qsvt_rules
from ...classiq_qsvt.overview.exact_basis import rz_to_two_u3


def s_to_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    """Preserve diag(1,i) exactly, including absolute phase."""
    return (
        Instruction(
            "u3",
            instruction.wires,
            params={"theta": 0.0, "phi": 0.0, "lbd": math.pi / 2},
            metadata=instruction.metadata,
        ),
    )


def sdg_to_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    """Preserve diag(1,-i) exactly, including absolute phase."""
    return (
        Instruction(
            "u3",
            instruction.wires,
            params={"theta": 0.0, "phi": 0.0, "lbd": -math.pi / 2},
            metadata=instruction.metadata,
        ),
    )


def rx_to_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    """RX(theta) equals U3(theta,-pi/2,pi/2), without a quarter-turn scalar."""
    return (
        Instruction(
            "u3",
            instruction.wires,
            params={
                "theta": instruction.params["theta"],
                "phi": -math.pi / 2,
                "lbd": math.pi / 2,
            },
            metadata=instruction.metadata,
        ),
    )


def exact_rules() -> Mapping[str, tuple[EquivalenceRule, ...]]:
    """Add local exact identities; leave shared decomposition tables untouched."""
    rules = with_equivalence_rule(qsvt_rules(), "s", s_to_u3)
    rules = with_equivalence_rule(rules, "sdg", sdg_to_u3)
    return with_equivalence_rule(rules, "rx", rx_to_u3)


def preserve_rotation_phases(program: CircuitIR) -> CircuitIR:
    """Expand source RX/RZ before generic quarter-turn simplification.

    Generic quarter-turn words may retain an operator only up to scalar phase.
    This caller compares absolute amplitudes, so exact U3 identities are required
    even for pi/2, pi and zero angles. No shared compiler table is changed.
    """
    return replace(
        program,
        instructions=tuple(
            leaf
            for instruction in program.instructions
            for leaf in (
                rx_to_u3(instruction)
                if instruction.name == "rx"
                else (
                    rz_to_two_u3(instruction)
                    if instruction.name == "rz"
                    else (instruction,)
                )
            )
        ),
    )
