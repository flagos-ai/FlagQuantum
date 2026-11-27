"""Keep native expanded effects separate from structural construction limits."""

from collections.abc import Sequence

from ..._hybrid.model import QUANTUM_EFFECT, HybridProgram
from .profile import _fail, _positive_integer


def check_register_budget(
    programs: Sequence[HybridProgram],
    *,
    max_nodes: int,
    max_effects: int,
    expanded: bool = False,
) -> None:
    _positive_integer(max_nodes, "max_nodes")
    _positive_integer(max_effects, "max_effects")
    if max_nodes > 4096 or max_effects > 16384:
        _fail(
            "register.budget", "safety caps are 4096 structural nodes and 16384 effects"
        )
    nodes = effects = 0
    for program in programs:
        for block in program.body.blocks:
            nodes += len(block.arguments) + len(block.operations)
            effects += sum(
                bool(op.results) and op.results[0].type == QUANTUM_EFFECT
                for op in block.operations
            )
    # Existing Qmod, QPE and direct register callers retain their combined cap.
    structural = nodes if max_effects <= 4096 else nodes - effects
    if structural > max_nodes:
        message = (
            "expanded register model exceeds node budget"
            if expanded
            else "register and expression source exceeds node budget"
        )
        _fail("register.budget", message)
    if effects > max_effects:
        _fail(
            "register.budget", f"expanded register model exceeds {max_effects} effects"
        )
