"""Construct owned register templates for checked invariant reset cycles."""

from dataclasses import replace

from .._hybrid.model import Block, HybridProgram, Operation, Region, Value, ValueId


def reset_cycle_template(
    name: str, effects: tuple[Operation, ...]
) -> tuple[HybridProgram, tuple[Value, ...]]:
    """Preserve the existing reset replay layout without parsing external syntax."""
    if (
        not effects
        or len(effects) > 512
        or not any(op.name == "qreg.reset" for op in effects)
    ):
        raise ValueError("a reset cycle requires 1..512 effects and an actual reset")
    owners = tuple(dict.fromkeys(value for op in effects for value in op.operands[1:]))
    # Declaration and allocation only establish the callee's checked qubit layout.
    # A replay call operates on existing initialized storage without allocating it.
    arguments = {
        actual: Value(ValueId(name, index + 1), actual.type)
        for index, actual in enumerate(owners)
    }
    entry = Value(ValueId(name, 0), effects[0].operands[0].type)
    effect = entry
    next_index = len(owners) + 1
    body = [
        Operation("qreg.declare", (), (formal,), {"name": f"port{index}"})
        for index, formal in enumerate(arguments.values())
    ]
    for formal in arguments.values():
        following = Value(ValueId(name, next_index), effect.type)
        next_index += 1
        body.append(Operation("qreg.allocate", (effect, formal), (following,)))
        effect = following
    for op in effects:
        following = Value(ValueId(name, next_index), effect.type)
        next_index += 1
        body.append(
            replace(
                op,
                operands=(effect, *(arguments[v] for v in op.operands[1:])),
                results=(following,),
            )
        )
        effect = following
    body.append(Operation("qreg.return", (effect, *arguments.values())))
    template = HybridProgram(name, Region((Block((entry,), tuple(body)),)))
    return template, owners
