"""Build a logical width model from verified register effects and ownership."""

from collections.abc import Callable, Sequence
from typing import cast

from .._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)
from .._hybrid.model import HybridProgram, Operation, Value, ValueId
from .register_expansion import expand_register_calls
from .register_profile import register_width, verify_register_program


def estimate_register_program_width(
    program: HybridProgram,
    functions: Sequence[HybridProgram] = (),
    *,
    local_peak: Callable[[Operation, int], int] | None = None,
) -> int:
    """Estimate ordered width from register storage and optional local costs.

    Outputs are reserved at entry as in register lowering. A local exists
    between its declaration and release. A cost callback receives the effect
    and the width of its distinct operand registers, including implicit
    uncompute inputs. Its result includes those operands and local workspace.
    The default counts only declared register storage.
    """
    verify_register_program(program, functions)
    expanded = expand_register_calls(program, functions)
    operations = expanded.body.blocks[0].operations
    producers = {op.results[0].id: 0 for op in operations if op.name == "qreg.declare"}
    computed_inputs: dict[ValueId, tuple[Value, ...]] = {}
    steps = [WidthStep("input")]
    connections: list[WidthConnection] = []
    for op in operations:
        if op.name == "qreg.declare":
            continue

        index = len(steps)
        # The effect chain orders all steps without adding storage.
        connections.append(WidthConnection(index - 1, index, 0))
        if op.name == "qreg.local":
            register = op.results[0]
            producers[register.id] = index
            steps.append(WidthStep("component", register_width(register)))
            continue

        operands = op.operands[1:]
        if op.name == "qreg.compute":
            computed_inputs[operands[0].id] = operands[1:]
        elif op.name == "qreg.uncompute":
            operands += computed_inputs.pop(operands[0].id)

        operand_width = 0
        for register in dict.fromkeys(operands):
            width = register_width(register)
            connections.append(WidthConnection(producers[register.id], index, width))
            operand_width += width
            if op.name == "qreg.release":
                del producers[register.id]
            else:
                producers[register.id] = index

        if op.name == "qreg.return":
            steps.append(WidthStep("output"))
        else:
            peak = (
                operand_width if local_peak is None else local_peak(op, operand_width)
            )
            if type(peak) is not int or peak < operand_width:
                raise ValueError("Local peak must cover the distinct operand registers")
            steps.append(WidthStep("component", peak))

    return cast(int, estimate_connection_width(steps, connections))
