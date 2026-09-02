"""Revision-aware analysis contracts and cache management."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, TypeVar

from ..ir.modules import Block, QuantumModule, Region
from ..ir.operations import Operation
from ..ir.values import ValueRef

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, order=True)
class OperationSite:
    """A deterministic operation position including nested region ancestry."""

    path: tuple[int, ...]
    operation_index: int


@dataclass(frozen=True)
class BlockView:
    path: tuple[int, ...]
    block: Block


class Analysis(Protocol[ResultT]):
    name: str
    version: str

    def run(self, module: QuantumModule) -> ResultT: ...


def walk_blocks(region: Region, prefix: tuple[int, ...] = ()) -> Iterator[BlockView]:
    """Walk blocks in canonical pre-order without relying on object identity."""

    for block_index, block in enumerate(region.blocks):
        block_path = prefix + (block_index,)
        yield BlockView(block_path, block)
        for operation_index, operation in enumerate(block.operations):
            for region_index, nested in enumerate(operation.regions):
                nested_prefix = block_path + (operation_index, region_index)
                yield from walk_blocks(nested, nested_prefix)


def walk_operations(
    module: QuantumModule,
) -> Iterator[tuple[OperationSite, Operation]]:
    for view in walk_blocks(module.body):
        for operation_index, operation in enumerate(view.block.operations):
            yield OperationSite(view.path, operation_index), operation


def block_arguments(module: QuantumModule) -> Iterator[tuple[OperationSite, ValueRef]]:
    for view in walk_blocks(module.body):
        for argument_index, argument in enumerate(view.block.arguments):
            yield OperationSite(view.path, -(argument_index + 1)), argument


class AnalysisManager:
    """Instance-local cache keyed by analysis, program identity, and revision."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str, int], object] = {}

    @staticmethod
    def _key(
        analysis_name: str,
        analysis_version: str,
        module: QuantumModule,
    ) -> tuple[str, str, str, int]:
        return (
            analysis_name,
            analysis_version,
            module.program_identity,
            module.revision,
        )

    def get(self, analysis: Analysis[ResultT], module: QuantumModule) -> ResultT:
        key = self._key(analysis.name, analysis.version, module)
        if key not in self._cache:
            self._cache[key] = analysis.run(module)
        return self._cache[key]  # type: ignore[return-value]

    def carry_preserved(
        self,
        old_module: QuantumModule,
        new_module: QuantumModule,
        preserved_analyses: frozenset[str],
    ) -> None:
        """Carry only explicitly preserved results across a semantic-equivalent revision."""

        if old_module.program_identity != new_module.program_identity:
            return
        old_prefix = (old_module.program_identity, old_module.revision)
        for key, value in tuple(self._cache.items()):
            name, version, identity, revision = key
            if (identity, revision) != old_prefix or name not in preserved_analyses:
                continue
            new_key = (name, version, new_module.program_identity, new_module.revision)
            self._cache[new_key] = value

    def invalidate(self, module: QuantumModule) -> None:
        doomed = [
            key
            for key in self._cache
            if key[2:] == (module.program_identity, module.revision)
        ]
        for key in doomed:
            del self._cache[key]

    @property
    def entry_count(self) -> int:
        return len(self._cache)


__all__ = [
    "Analysis",
    "AnalysisManager",
    "BlockView",
    "OperationSite",
    "block_arguments",
    "walk_blocks",
    "walk_operations",
]
