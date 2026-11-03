"""Bounded substitution of verified register callees into an owned root model."""

from collections.abc import Sequence
from dataclasses import replace

from .._hybrid.model import HybridProgram, IRType, Operation, Value, ValueId
from .profile import _fail


def is_register_function(program: HybridProgram) -> bool:
    return (
        len(program.body.blocks) == 1
        and bool(program.body.blocks[0].operations)
        and program.body.blocks[0].operations[-1].name == "qreg.return"
    )


def expand_register_calls(
    program: HybridProgram,
    functions: Sequence[HybridProgram],
    *,
    max_nodes: int = 4096,
) -> HybridProgram:
    """Substitute only after the caller has verified every definition and role."""
    block = program.body.blocks[0]
    if not any(op.name == "qreg.call" for op in block.operations):
        return program
    symbols = {f.name: f for f in functions}
    occupied = {
        v.id
        for v in (*block.arguments, *(v for op in block.operations for v in op.results))
    }
    effect = block.arguments[0]
    emitted: list[Operation] = []
    visits = counter = 0
    labels = {
        op.attributes["name"]
        for op in block.operations
        if op.name in {"qreg.declare", "qreg.local"}
    }

    def fresh(kind: IRType) -> Value:
        nonlocal counter
        identifier = ValueId(program.name + ".calls", counter)
        while identifier in occupied:
            counter += 1
            identifier = ValueId(program.name + ".calls", counter)
        occupied.add(identifier)
        counter += 1
        return Value(identifier, kind)

    def inverse_operations(operations: Sequence[Operation]) -> None:
        nonlocal effect, visits
        declarations = {
            op.results[0]: op for op in operations if op.name == "qreg.local"
        }
        active_computations: dict[Value, Operation] = {}
        computations: dict[int, Operation] = {}
        for index, op in enumerate(operations):
            if op.name == "qreg.compute":
                active_computations[op.operands[1]] = op
            elif op.name == "qreg.uncompute" and op.operands[1] in active_computations:
                computations[index] = active_computations.pop(op.operands[1])
        for index, op in reversed(tuple(enumerate(operations))):
            visits += 1
            if visits > max_nodes:
                _fail(
                    "register.budget",
                    "register call expansion exceeds visit budget",
                    op,
                )
            if op.name == "qreg.release" and op.operands[1] in declarations:
                emitted.append(declarations[op.operands[1]])
                continue
            following = fresh(effect.type)
            if op.name == "qreg.local":
                inverted = Operation(
                    "qreg.release",
                    (effect, op.results[0]),
                    (following,),
                    location=op.location,
                )
            elif op.name == "qreg.uncompute" and index in computations:
                forward = computations[index]
                inverted = replace(
                    forward,
                    operands=(effect, *forward.operands[1:]),
                    results=(following,),
                )
            elif op.name == "qreg.compute" and op.operands[1] in declarations:
                inverted = Operation(
                    "qreg.uncompute",
                    (effect, op.operands[1]),
                    (following,),
                    location=op.location,
                )
            elif op.name == "qreg.phase":
                attributes = dict(op.attributes)
                if "angle" in attributes:
                    attributes["angle"] = -attributes["angle"]
                inverted = replace(
                    op,
                    operands=(effect, *op.operands[1:]),
                    results=(following,),
                    attributes=attributes,
                )
            else:
                _fail(
                    "register.adjoint",
                    "callee effects are outside the checked inverse subset",
                    op,
                )
            emitted.append(inverted)
            effect = following

    def emit(op: Operation, bindings: dict[Value, Value]) -> None:
        nonlocal effect, visits
        visits += 1
        if visits > max_nodes:
            _fail("register.budget", "register call expansion exceeds visit budget", op)
        actual = tuple(bindings.get(v, v) for v in op.operands[1:])
        if op.name == "qreg.local":
            value = fresh(op.results[0].type)
            bindings[op.results[0]] = value
            label = f"{program.name}.calls.local{value.id.index}"
            while label in labels:
                label += ".local"
            labels.add(label)
            emitted.append(replace(op, results=(value,), attributes={"name": label}))
        elif op.name == "qreg.call":
            callee = symbols[op.attributes["callee"]].body.blocks[0]
            formal = (
                *callee.arguments[1:],
                *(q.results[0] for q in callee.operations if q.name == "qreg.declare"),
            )
            nested = dict(zip(formal, actual, strict=True))
            start, incoming = len(emitted), effect
            for child in callee.operations:
                if child.name not in {"qreg.declare", "qreg.return"}:
                    emit(replace(child, location=op.location), nested)
            if op.attributes.get("inverse"):
                forward = emitted[start:]
                del emitted[start:]
                effect = incoming
                inverse_operations(forward)
        else:
            following = fresh(effect.type)
            emitted.append(
                replace(op, operands=(effect, *actual), results=(following,))
            )
            effect = following

    for op in block.operations:
        if op.name in {"qreg.declare", "qreg.local"}:
            emitted.append(op)
        elif op.name == "qreg.return":
            emitted.append(replace(op, operands=(effect, *op.operands[1:])))
        else:
            emit(op, {})
    expanded = replace(
        program,
        body=replace(program.body, blocks=(replace(block, operations=tuple(emitted)),)),
    )
    arithmetic = tuple(f for f in functions if not is_register_function(f))
    nodes = sum(
        len(b.arguments) + len(b.operations)
        for f in (expanded, *arithmetic)
        for b in f.body.blocks
    )
    if nodes > max_nodes:
        _fail("register.budget", "expanded register model exceeds node budget")
    return expanded
