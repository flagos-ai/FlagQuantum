# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/bloq_finder.py.

"""Discover owned component classes and existing example metadata."""

import importlib
import inspect
from collections.abc import Callable, Iterable
from pathlib import Path

from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tools.development import process
from tools.development.repository import get_git_root


def _get_git_paths(root: Path, accept: Callable[[Path], bool]) -> list[Path]:
    output = process.output_of(["git", "ls-files", "*.py"], cwd=root, timeout=15)
    return [path for line in output.splitlines() if accept(path := Path(line))]


def _get_paths(
    root: Path, accept: Callable[[Path], bool], committed_only: bool = True
) -> list[Path]:
    if committed_only:
        return _get_git_paths(root, accept)
    return [path.relative_to(root) for path in root.glob("**/*.py") if accept(path)]


def get_component_module_paths(root: Path, committed_only: bool = True) -> list[Path]:
    """List source modules, excluding both native and original test conventions."""

    def is_module(path: Path) -> bool:
        return (
            path.name != "__init__.py"
            and not path.name.endswith("_test.py")
            and not path.name.startswith("test_")
        )

    return _get_paths(root, is_module, committed_only)


def get_component_test_module_paths(
    root: Path, committed_only: bool = True
) -> list[Path]:
    """List test modules under either supported filename convention."""
    return _get_paths(
        root,
        lambda path: path.name.endswith("_test.py") or path.name.startswith("test_"),
        committed_only,
    )


def _module_name(path: Path, package: str) -> str:
    return package + "." + str(path)[: -len(".py")].replace("/", ".")


def module_components(
    path: Path, *, package: str, is_component: Callable[[type], bool]
) -> Iterable[type]:
    """Yield public classes defined here and accepted by the caller's domain."""
    name = _module_name(path, package)
    module = importlib.import_module(name)
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != name or not is_component(cls):
            continue
        if not cls.__name__.startswith("_"):
            yield cls


def module_examples(
    path: Path, *, package: str
) -> Iterable[tuple[str, str, ComponentExample[object]]]:
    """Yield existing example metadata without running its factory."""
    name = _module_name(path, package)
    module = importlib.import_module(name)
    for attribute, value in inspect.getmembers(
        module, lambda item: isinstance(item, ComponentExample)
    ):
        yield name, attribute, value


def module_documentation(
    path: Path, *, package: str
) -> Iterable[tuple[str, str, ComponentDocumentation[object]]]:
    """Yield existing documentation metadata, including imported records."""
    name = _module_name(path, package)
    module = importlib.import_module(name)
    for attribute, value in inspect.getmembers(
        module, lambda item: isinstance(item, ComponentDocumentation)
    ):
        yield name, attribute, value


def get_component_classes(
    root: Path | None = None,
    *,
    package: str = "flagquantum.algorithms",
    is_component: Callable[[type], bool],
) -> list[type]:
    """Collect owned component classes using an explicit domain predicate.

    With no root, inspect tracked algorithms. An explicit root includes local
    source files, matching the source discovery tool's checkout convention.
    Native components need no common superclass.
    """
    committed_only = root is None
    if root is None:
        root = get_git_root() / "flagquantum" / "algorithms"
    return [
        cls
        for path in get_component_module_paths(root, committed_only)
        for cls in module_components(path, package=package, is_component=is_component)
    ]


def get_component_examples(
    root: Path | None = None, *, package: str = "flagquantum.algorithms"
) -> list[ComponentExample[object]]:
    """Collect lazy example records from a native package tree."""
    committed_only = root is None
    if root is None:
        root = get_git_root() / "flagquantum" / "algorithms"
    return [
        example
        for path in get_component_module_paths(root, committed_only)
        for _, _, example in module_examples(path, package=package)
    ]


def get_component_documentation(
    root: Path | None = None, *, package: str = "flagquantum.algorithms"
) -> list[ComponentDocumentation[object]]:
    """Collect documentation records from a native package tree."""
    committed_only = root is None
    if root is None:
        root = get_git_root() / "flagquantum" / "algorithms"
    return [
        record
        for path in get_component_module_paths(root, committed_only)
        for _, _, record in module_documentation(path, package=package)
    ]
