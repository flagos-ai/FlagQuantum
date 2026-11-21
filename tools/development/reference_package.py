# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_reference_docs/_make.py.

"""Walk static package objects into reference pages and link mappings."""

import ast
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import griffe
from griffe import GriffeLoader, Kind

from tools.development.reference_context import RenderContext
from tools.development.reference_pages import (
    MajorClassPage,
    MemberType,
    ModulePage,
    ModulePageMember,
    Page,
)

MAJOR_CLASSES = [
    "flagquantum.circuit.Circuit",
    "flagquantum.core.ir.program.model.ProgramRecord",
    "flagquantum.compiler.registers.component_builder.ComponentProgramBuilder",
    "flagquantum.core.ir.program.signatures.Signature",
    "flagquantum.core.ir.program.registers.RegisterType",
    "flagquantum.core.ir.program.registers.Register",
]
SKIP_MODULES: list[str] = []
MINIMAL_SKIP_CANONPATHS = [
    "flagquantum.ecosystem",
    "flagquantum.runtime.experiment",
]
MINIMAL = True
DEFINED_IN_CONTAINER_EXCEPTIONS = ["flagquantum.core.ir.program"]


def _get_all_aliases(obj: griffe.Object | griffe.Alias) -> set[str]:
    aliases = set(obj.aliases)
    aliases.update((obj.path, obj.canonical_path))
    defined_container = ".".join(obj.canonical_path.split(".")[:-1])

    def valid(alias: str) -> bool:
        if "._" in alias or alias.startswith("_"):
            return False
        container = ".".join(alias.split(".")[:-1])
        return (
            defined_container.startswith(container)
            or container in DEFINED_IN_CONTAINER_EXCEPTIONS
        )

    result = {alias for alias in aliases if valid(alias)}
    if not result:
        warnings.warn(f"No valid aliases for {obj} found in {aliases}")
    return result


def _get_preferred_dotpath(aliases: set[str]) -> str:
    # Preserve the actual source preference for the longest path. Equal-depth
    # choices retain the original set iteration order.
    return sorted(aliases, key=lambda alias: -len(alias.split(".")))[0]


@dataclass(kw_only=True)
class _PackageWalker:
    seen: set[str] = field(default_factory=set)
    _active: set[str] = field(default_factory=set, init=False, repr=False)
    pages_d: dict[str, Page] = field(default_factory=lambda: defaultdict(ModulePage))
    aliases_d: dict[str, str] = field(default_factory=dict)
    link_d: dict[str, tuple[Page, str | None]] = field(default_factory=dict)

    def _walk_table_of_contents(self, obj: griffe.Object | griffe.Alias) -> None:
        if obj.canonical_path in SKIP_MODULES:
            return
        if MINIMAL and obj.canonical_path in MINIMAL_SKIP_CANONPATHS:
            return
        canonical = obj.canonical_path
        if canonical in self._active:
            return
        self._active.add(canonical)
        try:
            if obj.is_module:
                for child in obj.members.values():
                    if child.kind is Kind.ALIAS:
                        continue
                    if child.canonical_path in self.seen:
                        continue
                    if child.is_private or child.is_special:
                        continue
                    if child.is_module and child.name.endswith(("_test", "_pb2")):
                        continue
                    self._walk_table_of_contents(child)
            aliases = _get_all_aliases(obj)
            if not aliases:
                return
            preferred = _get_preferred_dotpath(aliases)
            parent = ".".join(preferred.split(".")[:-1])
            for alias in aliases:
                assert alias not in self.aliases_d
                self.aliases_d[alias] = preferred
            if obj.is_module:
                member_type = MemberType.MODULE
            elif preferred in MAJOR_CLASSES:
                member_type = MemberType.MAJOR
            else:
                member_type = MemberType.MINOR
            parent_page = self.pages_d[parent]
            assert isinstance(
                parent_page, ModulePage
            ), f"Parent isn't a module: {parent_page}"
            parent_page.members.append(ModulePageMember(obj, preferred, member_type))
            if obj.is_module:
                assert isinstance(obj, (griffe.Module, griffe.Alias))
                page = self.pages_d[preferred]
                assert isinstance(page, ModulePage), f"Module page required: {page}"
                page.obj = obj
                page.section = ".".join(preferred.split(".")[:2])
                page.pref_path = preferred
                self.link_d[preferred] = (page, None)
                self.seen.add(obj.canonical_path)
            elif preferred in MAJOR_CLASSES:
                assert isinstance(obj, (griffe.Class, griffe.Alias))
                class_page = MajorClassPage(
                    obj=obj,
                    section=".".join(parent.split(".")[:2]),
                    pref_path=preferred,
                )
                assert preferred not in self.pages_d
                self.pages_d[preferred] = class_page
                self.link_d[preferred] = (class_page, None)
                self.seen.add(obj.canonical_path)
            else:
                self.seen.add(obj.canonical_path)
                self.link_d[preferred] = (parent_page, obj.name)
        finally:
            self._active.remove(canonical)


def get_pages(
    root_mod: griffe.Module,
) -> tuple[list[Page], dict[str, str], dict[str, tuple[Page, str | None]]]:
    assert root_mod.is_module
    assert root_mod.is_init_module
    assert root_mod.parent is None
    walker = _PackageWalker()
    walker._walk_table_of_contents(root_mod)
    for preferred in list(walker.pages_d):
        if walker.pages_d[preferred].obj is None:
            warnings.warn(f"Orphan {preferred!r}")
            del walker.pages_d[preferred]
    return list(walker.pages_d.values()), walker.aliases_d, walker.link_d


class ReferenceExtension(griffe.Extension):
    """Keep source property and attrs parsing local to each Griffe visitor."""

    def on_module_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        mod: griffe.Module,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        if isinstance(agent, griffe.Visitor):
            original_labels = agent.decorators_to_labels

            def labels(decorators: list[griffe.Decorator]) -> set[str]:
                return original_labels(
                    [
                        decorator
                        for decorator in decorators
                        if decorator.callable_path
                        not in {"property", "functools.cached_property"}
                    ]
                )

            # Griffe's extension callback is the optional parser boundary.
            # Bind only this agent, preserving every other visitor's defaults.
            setattr(agent, "decorators_to_labels", labels)

    def _is_attrs_class(self, cls: griffe.Class) -> bool:
        return "attrs.frozen" in [
            decorator.callable_path for decorator in cls.decorators
        ]

    def on_class_members(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        cls: griffe.Class,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        if self._is_attrs_class(cls):
            cls.members["__init__"] = griffe.Function(
                name="__init__",
                parameters=griffe.Parameters(
                    *[
                        griffe.Parameter(
                            name=name.lstrip("_"), annotation=attribute.annotation
                        )
                        for name, attribute in cls.attributes.items()
                    ]
                ),
            )


def walk_and_configure(
    rootmod_dotpath: str,
    refdoc_relpath: Path,
    top_sections: Sequence[str],
    fake_sections: Sequence[str],
    addtl_linkable: dict[str, str],
) -> tuple[list[Page], RenderContext]:
    extensions = griffe.load_extensions(ReferenceExtension())
    loader = GriffeLoader(extensions=extensions)
    root_mod = loader.load(rootmod_dotpath)
    assert isinstance(root_mod, griffe.Module), root_mod
    unresolved, _ = loader.resolve_aliases()
    assert len(unresolved) == 0
    pages, aliases, links = get_pages(root_mod)
    sections = list(top_sections) + sorted(
        {page.section_not_none for page in pages}
        - set(top_sections)
        - set(fake_sections)
    )
    context = RenderContext(
        refdoc_relpath=refdoc_relpath,
        sections=sections,
        aliases_d=aliases,
        link_d=links,
        linkable_to_prefpath=aliases | addtl_linkable,
    )
    return pages, context
