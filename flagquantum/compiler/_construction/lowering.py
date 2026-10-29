"""Conservative exact addition: compute, XOR-copy, then uncompute all workspace.

Inputs and output are borrowed physical maps. Every supplied workspace wire must
enter zero. All intermediates remain allocated until the reverse sweep; no
lifetime optimizer, measurement, approximation or arbitrary gate inversion is
used. Only X/CX/CCX are emitted, and each is exactly self-adjoint.
"""

from collections.abc import Mapping, Sequence
from typing import Literal

from ...core.ir import CircuitIR, Instruction
from .._arithmetic_kernels import ripple_add
from .._hybrid.model import HybridProgram, Operation, ValueId
from .bindings import verify_bindings
from .profile import _fail, _positive_integer
from .ranges import unsigned_width

_WireValue = tuple[int, ...] | int
_BoundValue = tuple[_WireValue, int]


def _copy(value: _WireValue, target: tuple[int, ...]) -> tuple[Instruction, ...]:
    """XOR a basis encoding or classical constant into a zero-extended target."""
    if isinstance(value, int):
        return tuple(
            Instruction("x", (wire,))
            for bit, wire in enumerate(reversed(target))
            if value & (1 << bit)
        )
    if len(value) > len(target):
        raise ValueError("internal addition copy would truncate a quantum value")
    return tuple(
        Instruction("cx", (source, dest))
        for source, dest in zip(value, target[-len(value) :], strict=True)
    )


def lower_addition(
    functions: Sequence[HybridProgram],
    inputs: Mapping[ValueId, Sequence[int]],
    output: Sequence[int],
    *,
    entry: str,
    n_wires: int,
    workspace: Sequence[int] = (),
    workspace_state: Literal[
        "fresh_zero", "caller_asserted_zero", "unknown"
    ] = "unknown",
    max_operations: int = 100_000,
) -> CircuitIR:
    """Lower verified constant/add/call definitions to existing Core IR.

    Port widths follow declared types; internal wires may omit proven high zero
    bits. Workspace exhaustion or operation-budget failure returns no partial IR.
    External zero-state assertions remain caller preconditions, not state proofs.
    """
    _positive_integer(n_wires, "n_wires")
    _positive_integer(max_operations, "max_operations")
    if n_wires > 256 or max_operations > 100_000:
        _fail("lowering.budget", "safety caps are 256 wires and 100000 operations")
    verify_bindings(
        functions,
        inputs,
        output,
        entry=entry,
        n_wires=n_wires,
        workspace=workspace,
        workspace_state=workspace_state,
    )
    for function in functions:
        for operation in function.body.blocks[0].operations:
            if operation.name == "qint.multiply":
                _fail(
                    "lowering.unsupported",
                    "qint.multiply has no addition-only lowering",
                    operation,
                )
    symbols = {function.name: function for function in functions}
    available = tuple(workspace)
    used = 0
    forward: list[Instruction] = []

    def add(left: _BoundValue, right: _BoundValue, op: Operation) -> _BoundValue:
        nonlocal used
        width = max(unsigned_width(left[1]), unsigned_width(right[1]))
        needed = 2 * width + 2
        if used + needed > len(available):
            _fail(
                "lowering.workspace",
                f"addition needs {needed} more zero wires; "
                f"{len(available) - used} remain (retain-all schedule)",
                op,
            )
        wires = available[used : used + needed]
        a, b = wires[:width], wires[width : 2 * width]
        carry, carry_out = wires[-2:]
        instructions = (
            *_copy(left[0], a),
            *_copy(right[0], b),
            *ripple_add(a, b, carry, carry_out),
        )
        if 2 * (len(forward) + len(instructions)) > max_operations:
            _fail(
                "lowering.operations", "compute/uncompute exceeds operation budget", op
            )
        forward.extend(instructions)
        used += needed
        bound = left[1] + right[1]
        result = (carry_out, *b)[-unsigned_width(bound) :]
        return result, bound

    def evaluate(name: str, arguments: tuple[_BoundValue, ...]) -> _BoundValue:
        block = symbols[name].body.blocks[0]
        values = {
            argument.id: actual
            for argument, actual in zip(block.arguments, arguments, strict=True)
        }
        for operation in block.operations:
            operands = tuple(values[v.id] for v in operation.operands)
            if operation.name == "qint.return":
                return operands[0]
            if operation.name == "arith.constant":
                constant = operation.attributes["value"]
                result: _BoundValue = constant, constant
            elif operation.name == "qint.call":
                result = evaluate(operation.attributes["callee"], operands)
            else:  # The verified addition-only profile leaves only qint.add.
                result = add(operands[0], operands[1], operation)
            values[operation.results[0].id] = result
        raise AssertionError("verified construction did not return")

    block = symbols[entry].body.blocks[0]
    arguments: tuple[_BoundValue, ...] = tuple(
        (tuple(inputs[arg.id]), (1 << len(inputs[arg.id])) - 1)
        for arg in block.arguments
    )
    result, _ = evaluate(entry, arguments)
    copy = _copy(result, tuple(output))
    if 2 * len(forward) + len(copy) > max_operations:
        _fail(
            "lowering.operations",
            "final XOR copy exceeds operation budget",
            block.operations[-1],
        )
    return CircuitIR(
        n_wires=n_wires,
        instructions=(*forward, *copy, *reversed(forward)),
    )
