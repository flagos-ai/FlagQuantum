"""Direct uses and inverse dependencies over verified, inlined construction SSA."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from .profile import verify_construction


@dataclass(frozen=True)
class ValueLifetime:
    value: Value
    upper_bound: int
    dependencies: tuple[ValueId, ...]
    uses: tuple[int, ...]

    @property
    def last_use(self) -> int | None:
        return self.uses[-1] if self.uses else None


@dataclass(frozen=True)
class LifetimeAnalysis:
    program: HybridProgram
    values: Mapping[ValueId, ValueLifetime]


def analyze_lifetimes(
    functions: Sequence[HybridProgram], *, entry: str
) -> LifetimeAnalysis:
    """Inline verified calls and record dependencies without assigning physical qubits.

    Each call instance gets distinct SSA results; returned aliases keep their
    identity. Cleanup needs the original dependency *values*, even after last use.
    Immutable basis borrows let a scheduler recompute those values later.
    """
    main = verify_construction(functions, entry=entry)
    symbols = {function.name: function for function in functions}
    arguments = main.body.blocks[0].arguments
    bounds = {arg.id: (1 << arg.type.parameters[0]) - 1 for arg in arguments}
    operations: list[Operation] = []
    counter = len(arguments)
    scope = entry
    while any(arg.id.scope == scope for arg in arguments):
        scope += ".inline"

    def inline(name: str, actual: tuple[Value, ...]) -> Value:
        nonlocal counter
        block = symbols[name].body.blocks[0]
        values = {
            arg.id: value for arg, value in zip(block.arguments, actual, strict=True)
        }
        for operation in block.operations:
            operands = tuple(values[value.id] for value in operation.operands)
            if operation.name == "qint.return":
                return operands[0]
            if operation.name == "qint.call":
                result = inline(operation.attributes["callee"], operands)
            else:
                result = Value(ValueId(scope, counter), operation.results[0].type)
                counter += 1
                if operation.name == "arith.constant":
                    bound = operation.attributes["value"]
                elif operation.name in {"qint.equal", "qint.less"}:
                    bound = 1
                elif operation.name == "qint.select":
                    bound = max(bounds[value.id] for value in operands[1:])
                else:
                    left, right = (bounds[value.id] for value in operands)
                    bound = (
                        left + right if operation.name == "qint.add" else left * right
                    )
                bounds[result.id] = bound
                operations.append(
                    Operation(
                        operation.name,
                        operands,
                        (result,),
                        operation.attributes,
                        location=operation.location,
                    )
                )
            values[operation.results[0].id] = result
        raise AssertionError("verified construction did not return")

    result = inline(entry, arguments)
    operations.append(
        Operation(
            "qint.return",
            (result,),
            location=main.body.blocks[0].operations[-1].location,
        )
    )
    program = HybridProgram(entry, Region((Block(arguments, tuple(operations)),)))
    uses: dict[ValueId, list[int]] = {arg.id: [] for arg in arguments}
    dependencies: dict[ValueId, tuple[ValueId, ...]] = {arg.id: () for arg in arguments}
    values = {arg.id: arg for arg in arguments}
    for position, operation in enumerate(operations):
        for operand in dict.fromkeys(operation.operands):
            uses[operand.id].append(position)
        for value in operation.results:
            uses[value.id] = []
            values[value.id] = value
            dependencies[value.id] = tuple(
                dict.fromkeys(v.id for v in operation.operands if v.type != INDEX)
            )
    return LifetimeAnalysis(
        program,
        MappingProxyType(
            {
                key: ValueLifetime(
                    value, bounds[key], dependencies[key], tuple(uses[key])
                )
                for key, value in values.items()
            }
        ),
    )
