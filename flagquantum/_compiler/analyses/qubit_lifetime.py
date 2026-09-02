"""Linear qubit lifetime analysis built on deterministic program points."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.modules import QuantumModule
from ..ir.values import ValueId
from .base import OperationSite, block_arguments, walk_operations


@dataclass(frozen=True)
class QubitLifetime:
    value: ValueId
    defined_at: OperationSite
    consumed_at: OperationSite | None
    successor_values: tuple[ValueId, ...]
    terminal: bool


@dataclass(frozen=True)
class QubitLifetimeResult:
    lifetimes: tuple[QubitLifetime, ...]

    def lifetime_of(self, value: ValueId) -> QubitLifetime | None:
        return next((item for item in self.lifetimes if item.value == value), None)


class QubitLifetimeAnalysis:
    name = "qubit_lifetime"
    version = "1.0"

    def run(self, module: QuantumModule) -> QubitLifetimeResult:
        definitions: dict[ValueId, OperationSite] = {
            argument.id: site
            for site, argument in block_arguments(module)
            if argument.linear
        }
        consumption: dict[ValueId, OperationSite] = {}
        successors: dict[ValueId, tuple[ValueId, ...]] = {}
        terminal: set[ValueId] = set()
        for site, operation in walk_operations(module):
            linear_operands = tuple(item for item in operation.operands if item.linear)
            linear_results = tuple(item for item in operation.results if item.linear)
            for result in linear_results:
                definitions[result.id] = site
            for operand_index, operand in enumerate(linear_operands):
                consumption[operand.id] = site
                successors[operand.id] = (
                    (linear_results[operand_index].id,)
                    if operand_index < len(linear_results)
                    else ()
                )
                if operation.name in {"quantum.measure", "quantum.release"}:
                    terminal.add(operand.id)
        lifetimes = tuple(
            QubitLifetime(
                value,
                defined_at,
                consumption.get(value),
                successors.get(value, ()),
                value in terminal,
            )
            for value, defined_at in sorted(definitions.items())
        )
        return QubitLifetimeResult(lifetimes)


__all__ = ["QubitLifetime", "QubitLifetimeAnalysis", "QubitLifetimeResult"]
