# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 reference_docs.py.
# Command entry adapted from dev_tools/build-reference-docs.py.

"""Generate template-driven reference pages through optional TensorFlow-docs."""

from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from types import ModuleType

from tensorflow_docs.api_generator.generate_lib import DocGenerator
from tensorflow_docs.api_generator.public_api import local_definitions_filter

from tools.development.reference_output import apply_fixups, generate_ref_toc
from tools.development.reference_template_pages import page_info_classes
from tools.development.repository import get_git_root

_TYPE_ALIAS_LOCATIONS = {
    "SymbolicInt": ("flagquantum", "core", "value_math"),
    "SymbolicReal": ("flagquantum", "core", "value_math"),
    "SymbolicComplex": ("flagquantum", "core", "value_math"),
    "TypeParameter": ("flagquantum", "core", "ir", "program", "registers"),
    "ControlScalar": ("flagquantum", "core", "ir", "program", "physical"),
}


def filter_type_checking(
    path: tuple[str, ...], parent: object, children: Iterable[tuple[str, object]]
) -> list[tuple[str, object]]:
    return [(name, value) for name, value in children if name != "TYPE_CHECKING"]


def filter_type_aliases_in_the_wrong_place(
    path: tuple[str, ...], parent: object, children: Iterable[tuple[str, object]]
) -> list[tuple[str, object]]:
    return [
        (name, value)
        for name, value in children
        if name not in _TYPE_ALIAS_LOCATIONS or path == _TYPE_ALIAS_LOCATIONS[name]
    ]


def generate_ref_docs(reporoot: Path, root_module: ModuleType | None = None) -> None:
    """Generate pages for the native package or an explicit developer fixture."""
    if root_module is None:
        root_module = import_module("flagquantum")
        for name in (
            "core.ir.program",
            "compiler",
            "drawer",
            "simulation",
            "ecosystem",
            "testing",
            "runtime",
            "core.ir.program.physical_codec",
        ):
            import_module("flagquantum." + name)
    package = root_module.__name__
    generator = DocGenerator(
        root_title="FlagQuantum",
        py_modules=[(package, root_module)],
        base_dir=[reporoot / package],
        code_url_prefix=f"https://github.com/flagos-ai/FlagQuantum/blob/main/{package}",
        callbacks=[
            local_definitions_filter,
            filter_type_checking,
            filter_type_aliases_in_the_wrong_place,
        ],
        page_builder_classes=page_info_classes(),
    )
    generator.build(output_dir=reporoot / "docs/reference")


def build_reference_docs(
    reporoot: Path | None = None, root_module: ModuleType | None = None
) -> None:
    """Generate pages, repair local link suffixes and write Sphinx contents."""
    root = get_git_root() if reporoot is None else reporoot
    generate_ref_docs(root, root_module)
    package = "flagquantum" if root_module is None else root_module.__name__
    apply_fixups(root, package)
    generate_ref_toc(root, package)


def main() -> None:
    """Build template reference pages for the current native checkout."""
    build_reference_docs()


if __name__ == "__main__":
    main()
