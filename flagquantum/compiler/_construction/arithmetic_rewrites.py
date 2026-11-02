"""Bounded pure-SSA normalization, CSE and source-preserving bit evidence.

The original inlined graph remains available. No fact crosses a register effect,
call epoch or region, and declared input/result types are never narrowed.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)
from .bit_facts import BitFacts, add_facts, constant_facts, facts, multiply_facts
from .lifetimes import LifetimeAnalysis
from .profile import _fail, _positive_integer, verify_construction


@dataclass(frozen=True)
class ArithmeticNormalization:
    original: HybridProgram
    program: HybridProgram
    facts: Mapping[ValueId, BitFacts]
    replacements: Mapping[ValueId, Value]
    origins: Mapping[ValueId, tuple[SourceLocation, ...]]
    squares: frozenset[ValueId]


def normalize_arithmetic(
    analysis: LifetimeAnalysis, *, max_nodes: int = 4096
) -> ArithmeticNormalization:
    """Normalize only a verified closed flat graph, within a fixed node ceiling.

    The incoming lifetime graph has already inlined pure calls with actual SSA
    aliases. Bit bounds reuse that analysis. Non-arithmetic predicate/select
    nodes remain operations; effectful or nested IR is rejected before rewriting.
    """
    _positive_integer(max_nodes, "max_nodes")
    if max_nodes > 4096:
        _fail("normalization.budget", "normalization safety ceiling is 4096 nodes")
    source = analysis.program
    verify_construction((source,), entry=source.name, max_nodes=max_nodes)
    block = source.body.blocks[0]
    scope = source.name
    while any(arg.id.scope == scope for arg in block.arguments):
        scope += ".inline"
    values = {arg.id: arg for arg in block.arguments}
    ranks = {arg.id: index for index, arg in enumerate(block.arguments)}
    known = {
        arg.id: facts(arg.type.parameters[0], analysis.values[arg.id].upper_bound)
        for arg in block.arguments
    }
    origins: dict[ValueId, list[SourceLocation]] = {
        arg.id: list(analysis.source_origins[arg.id]) for arg in block.arguments
    }
    emitted: list[Operation] = []
    common: dict[tuple[object, ...], Value] = {}
    squares: set[ValueId] = set()

    for op in block.operations[:-1]:
        operands = tuple(values[value.id] for value in op.operands)
        old = op.results[0]
        upper = analysis.values[old.id].upper_bound
        width = (
            old.type.parameters[0] if old.type != INDEX else max(1, upper.bit_length())
        )
        alias = None
        key: tuple[object, ...] | None = None
        if op.name == "arith.constant":
            evidence = constant_facts(upper)
            key = (op.name, upper)
        elif op.name in {"qint.add", "qint.multiply"}:
            # Wide quantum operands first avoid needless ripple padding. Constants
            # stay last; equal domains do not imply equal quantum values.
            operands = tuple(
                sorted(
                    operands,
                    key=lambda v: (
                        v.type == INDEX,
                        -known[v.id].effective_width,
                        -known[v.id].width,
                        ranks[v.id],
                    ),
                )
            )
            left, right = operands
            a, b = known[left.id], known[right.id]
            same = left.id == right.id
            if op.name == "qint.add" and same:
                mask = (1 << width) - 1
                zero = ((a.known_zero << 1) | 1) & mask
                zero |= mask ^ ((1 << min(width, a.width + 1)) - 1)
                evidence = facts(width, upper, zero, a.known_one << 1)
            elif op.name == "qint.add":
                evidence = add_facts(a, b, width, upper)
            else:
                evidence = multiply_facts(a, b, width, upper, square=same)
            for variable, constant in ((left, b.constant), (right, a.constant)):
                identity = 0 if op.name == "qint.add" else 1
                if constant == identity and variable.type == old.type:
                    alias = variable
            key = (op.name, old.type, tuple(value.id for value in operands))
            if evidence.constant is not None:
                # Keep a typed quantum witness; INDEX alone cannot be returned
                # as quantum storage. All original reads were verified first.
                key = ("quantum.constant", old.type, evidence.constant)
        else:
            # Predicates retain a conservative full one-bit domain. Select only
            # intersects facts common to both zero-extended branches.
            evidence = facts(width, upper)
            if op.name == "qint.select":
                yes, no = (known[value.id] for value in operands[1:])
                mask = (1 << width) - 1
                yes_zero = (
                    yes.known_zero | (mask ^ ((1 << min(width, yes.width)) - 1))
                ) & mask
                no_zero = (
                    no.known_zero | (mask ^ ((1 << min(width, no.width)) - 1))
                ) & mask
                evidence = facts(
                    width, upper, yes_zero & no_zero, yes.known_one & no.known_one
                )
        result = (
            alias if alias is not None else common.get(key) if key is not None else None
        )
        if result is None:
            result = Value(
                ValueId(scope, len(block.arguments) + len(emitted)), old.type
            )
            emitted.append(
                Operation(
                    op.name, operands, (result,), op.attributes, location=op.location
                )
            )
            known[result.id] = evidence
            ranks[result.id] = len(block.arguments) + len(emitted) - 1
            origins[result.id] = []
            if key is not None:
                common[key] = result
            if op.name == "qint.multiply" and operands[0].id == operands[1].id:
                squares.add(result.id)
        values[old.id] = result
        locations = analysis.source_origins[old.id]
        origins[result.id] = list(dict.fromkeys((*origins[result.id], *locations)))

    returned = values[block.operations[-1].operands[0].id]
    needed = {returned.id}
    live: list[Operation] = []
    for op in reversed(emitted):
        if op.results[0].id in needed:
            live.append(op)
            needed.update(value.id for value in op.operands)
    live.reverse()
    # Compact IDs after DCE so a second normalization is exactly idempotent.
    compact = {arg.id: arg for arg in block.arguments}
    operations = []
    for index, op in enumerate(live, start=len(block.arguments)):
        result = Value(ValueId(scope, index), op.results[0].type)
        compact[op.results[0].id] = result
        operations.append(
            Operation(
                op.name,
                tuple(compact[v.id] for v in op.operands),
                (result,),
                op.attributes,
                location=op.location,
            )
        )
    operations.append(
        Operation(
            "qint.return",
            (compact[returned.id],),
            location=block.operations[-1].location,
        )
    )
    program = HybridProgram(
        source.name, Region((Block(block.arguments, tuple(operations)),))
    )
    verify_construction((program,), entry=program.name, max_nodes=max_nodes)
    return ArithmeticNormalization(
        source,
        program,
        MappingProxyType({new.id: known[old] for old, new in compact.items()}),
        MappingProxyType(
            {
                old: compact[value.id]
                for old, value in values.items()
                if value.id in compact
            }
        ),
        MappingProxyType({new.id: tuple(origins[old]) for old, new in compact.items()}),
        frozenset(compact[key].id for key in squares if key in compact),
    )
