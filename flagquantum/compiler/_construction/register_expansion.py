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
            for child in callee.operations:
                if child.name not in {"qreg.declare", "qreg.return"}:
                    emit(replace(child, location=op.location), nested)
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
