"""Deterministic definition/use analysis for immutable QuantumIR."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.modules import QuantumModule
from ..ir.values import ValueId
from .base import OperationSite, block_arguments, walk_operations


@dataclass(frozen=True)
class DefinitionRecord:
    value: ValueId
    site: OperationSite
    result_index: int
    block_argument: bool = False


@dataclass(frozen=True)
class UseRecord:
    value: ValueId
    site: OperationSite
    operand_index: int


@dataclass(frozen=True)
class DefUseResult:
    definitions: tuple[DefinitionRecord, ...]
    uses: tuple[UseRecord, ...]

    def definitions_of(self, value: ValueId) -> tuple[DefinitionRecord, ...]:
        return tuple(item for item in self.definitions if item.value == value)

    def uses_of(self, value: ValueId) -> tuple[UseRecord, ...]:
        return tuple(item for item in self.uses if item.value == value)


class DefUseAnalysis:
    name = "def_use"
    version = "1.0"

    def run(self, module: QuantumModule) -> DefUseResult:
        definitions = [
            DefinitionRecord(argument.id, site, -site.operation_index - 1, True)
            for site, argument in block_arguments(module)
        ]
        uses: list[UseRecord] = []
        for site, operation in walk_operations(module):
            definitions.extend(
                DefinitionRecord(result.id, site, result_index)
                for result_index, result in enumerate(operation.results)
            )
            uses.extend(
                UseRecord(operand.id, site, operand_index)
                for operand_index, operand in enumerate(operation.operands)
            )
        return DefUseResult(tuple(definitions), tuple(uses))


__all__ = [
    "DefinitionRecord",
    "DefUseAnalysis",
    "DefUseResult",
    "UseRecord",
]
