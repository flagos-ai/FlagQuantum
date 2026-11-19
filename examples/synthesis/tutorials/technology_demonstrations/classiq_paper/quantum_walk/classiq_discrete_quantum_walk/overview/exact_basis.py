"""Operator-exact rules for a caller-requested CX/U3 named basis."""

import math
from collections.abc import Mapping

from flagquantum.compiler.basis_translation import (
    EQUIVALENCE_RULES,
    EquivalenceRule,
    with_equivalence_rule,
)
from flagquantum.core.ir import Instruction


def h_to_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    """H equals U3(pi/2,0,pi), including its absolute phase."""
    return (
        Instruction(
            "u3",
            instruction.wires,
            params={"theta": math.pi / 2, "phi": 0.0, "lbd": math.pi},
            metadata=instruction.metadata,
        ),
    )


def phase_to_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    """A phase gate equals U3(0,0,theta)."""
    return (
        Instruction(
            "u3",
            instruction.wires,
            params={"theta": 0.0, "phi": 0.0, "lbd": instruction.params["theta"]},
            metadata=instruction.metadata,
        ),
    )


def cphase_to_u3_cx(instruction: Instruction) -> tuple[Instruction, ...]:
    """Use three phase gates and two CX gates without a lost global phase."""
    control, target = instruction.wires
    half = instruction.params["theta"] * 0.5
    metadata = instruction.metadata
    return (
        Instruction(
            "u3",
            (control,),
            params={"theta": 0.0, "phi": 0.0, "lbd": half},
            metadata=metadata,
        ),
        Instruction(
            "u3",
            (target,),
            params={"theta": 0.0, "phi": 0.0, "lbd": half},
            metadata=metadata,
        ),
        Instruction("cx", (control, target), metadata=metadata),
        Instruction(
            "u3",
            (target,),
            params={"theta": 0.0, "phi": 0.0, "lbd": -half},
            metadata=metadata,
        ),
        Instruction("cx", (control, target), metadata=metadata),
    )


def x_to_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    return (
        Instruction(
            "u3",
            instruction.wires,
            params={"theta": math.pi, "phi": 0.0, "lbd": math.pi},
            metadata=instruction.metadata,
        ),
    )


def t_to_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    return (
        Instruction(
            "u3",
            instruction.wires,
            params={"theta": 0.0, "phi": 0.0, "lbd": math.pi / 4},
            metadata=instruction.metadata,
        ),
    )


def tdg_to_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    return (
        Instruction(
            "u3",
            instruction.wires,
            params={"theta": 0.0, "phi": 0.0, "lbd": -math.pi / 4},
            metadata=instruction.metadata,
        ),
    )


def exact_rules() -> Mapping[str, tuple[EquivalenceRule, ...]]:
    """Extend the immutable public rule table; leave all shared defaults untouched."""
    rules = EQUIVALENCE_RULES
    for opcode, builder in (
        ("h", h_to_u3),
        ("phase", phase_to_u3),
        ("cphase", cphase_to_u3_cx),
        ("x", x_to_u3),
        ("t", t_to_u3),
        ("tdg", tdg_to_u3),
    ):
        rules = with_equivalence_rule(rules, opcode, builder)
    return rules
