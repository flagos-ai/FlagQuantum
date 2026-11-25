"""Operator-exact caller-owned RZ identity for the source CX/U3 target."""

import math
from collections.abc import Mapping

from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.exact_basis import (
    exact_rules as walk_rules,
)
from flagquantum.compiler.basis_translation import (
    EquivalenceRule,
    with_equivalence_rule,
)
from flagquantum.core.ir import Instruction


def rz_to_two_u3(instruction: Instruction) -> tuple[Instruction, ...]:
    # In execution order U3(pi,0,pi-t/2) @ U3(pi,0,pi+t/2) = RZ(t).
    half = instruction.params["theta"] * 0.5
    return tuple(
        Instruction(
            "u3",
            instruction.wires,
            params={"theta": math.pi, "phi": 0.0, "lbd": math.pi + angle},
            metadata=instruction.metadata,
        )
        for angle in (half, -half)
    )


def exact_rules() -> Mapping[str, tuple[EquivalenceRule, ...]]:
    return with_equivalence_rule(walk_rules(), "rz", rz_to_two_u3)
