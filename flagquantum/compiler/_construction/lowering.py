"""Conservative exact unsigned arithmetic: compute, XOR-copy, then uncompute all workspace.

Inputs and output are borrowed physical maps. Every supplied workspace qubit must
enter zero. All results remain allocated until the reverse sweep; locally cleaned
multiplier scratch may be reused; no
lifetime optimizer, measurement, approximation or arbitrary gate inversion is
used. Only X/CX/CCX are emitted, and each is exactly self-adjoint.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from ...core.ir import CircuitIR, Instruction
from .._arithmetic_kernels import multiply_accumulate, ripple_add
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
        raise ValueError("internal arithmetic copy would truncate a quantum value")
    return tuple(
        Instruction("cx", (source, dest))
        for source, dest in zip(value, target[-len(value) :], strict=True)
    )


def lower_arithmetic(
    functions: Sequence[HybridProgram],
    inputs: Mapping[ValueId, Sequence[int]],
    output: Sequence[int],
    *,
    entry: str,
    n_qubits: int,
    workspace: Sequence[int] = (),
    workspace_state: Literal[
        "fresh_zero", "caller_asserted_zero", "unknown"
    ] = "unknown",
    max_operations: int = 100_000,
) -> CircuitIR:
    """Lower verified constant/add/multiply/call definitions to existing Core IR.

    Port widths follow declared types; internal qubits may omit proven high zero
    bits. Workspace exhaustion or operation-budget failure returns no partial IR.
    External zero-state assertions remain caller preconditions, not state proofs.
    """
    _positive_integer(n_qubits, "n_qubits")
    _positive_integer(max_operations, "max_operations")
    if n_qubits > 256 or max_operations > 100_000:
        _fail("lowering.budget", "safety caps are 256 qubits and 100000 operations")
    verify_bindings(
        functions,
        inputs,
        output,
        entry=entry,
        n_wires=n_qubits,
        workspace=workspace,
        workspace_state=workspace_state,
    )
    symbols = {function.name: function for function in functions}
    available = tuple(workspace)
    used = 0
    forward: list[Instruction] = []

    def allocate(
        count: int, op: Operation, *, temporary: bool = False
    ) -> tuple[int, ...]:
        nonlocal used
        if used + count > len(available):
            _fail(
                "lowering.workspace",
                f"{op.name} needs {count} more zero qubits; "
                f"{len(available) - used} remain (results retained)",
                op,
            )
        result = available[used : used + count]
        if not temporary:
            used += count
        return result

    def emit(instructions: Iterable[Instruction], op: Operation) -> None:
        for instruction in instructions:
            if 2 * (len(forward) + 1) > max_operations:
                _fail(
                    "lowering.operations",
                    "compute/uncompute exceeds operation budget",
                    op,
                )
            forward.append(instruction)

    def add(left: _BoundValue, right: _BoundValue, op: Operation) -> _BoundValue:
        if isinstance(left[0], int):
            left, right = right, left
        # The verified profile requires a quantum operand. Cuccaro restores a,
        # so borrow it instead of copying it. Copy b before a is touched, even
        # when the two expressions share a physical encoding.
        source = left[0]
        assert not isinstance(source, int)
        width = max(unsigned_width(left[1]), unsigned_width(right[1]))
        padding = allocate(width - len(source), op)
        b = allocate(width, op)
        carry, carry_out = allocate(2, op)
        emit(_copy(right[0], b), op)
        emit(ripple_add((*padding, *source), b, carry, carry_out), op)
        bound = left[1] + right[1]
        return (carry_out, *b)[-unsigned_width(bound) :], bound

    def multiply(left: _BoundValue, right: _BoundValue, op: Operation) -> _BoundValue:
        bound = left[1] * right[1]
        product = allocate(unsigned_width(bound), op)
        scratch = allocate(len(product), op, temporary=True)
        emit(multiply_accumulate(left[0], right[0], product, scratch), op)
        # Every increment cleans its scratch before returning. Later retained
        # results may occupy it; the reverse sweep clears those results before
        # the multiplier inverse needs the same zero scratch again.
        return product, bound

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
            elif operation.name == "qint.add":
                result = add(operands[0], operands[1], operation)
            else:  # The closed verified profile leaves only qint.multiply.
                result = multiply(operands[0], operands[1], operation)
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
        n_wires=n_qubits,
        instructions=(*forward, *copy, *reversed(forward)),
    )
