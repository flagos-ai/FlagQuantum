# Copyright 2024 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_reference_docs/_make.py.
# Command entry adapted from dev_tools/build-reference-docs-2.py.

"""Generate reference pages using the existing static package and page owners."""

from pathlib import Path

from tools.development.reference_class_pages import render_major_class
from tools.development.reference_module_pages import render_module
from tools.development.reference_package import walk_and_configure
from tools.development.reference_pages import MajorClassPage, ModulePage, Page
from tools.development.reference_toc import render_toc
from tools.development.repository import get_git_root

TOP_SECTIONS = [
    "flagquantum",
    "flagquantum.core",
    "flagquantum.compiler",
    "flagquantum.simulation",
    "flagquantum.drawer",
    "flagquantum.ecosystem",
]
GROUPED_SECTIONS: dict[str, tuple[str, ...]] = {}
ADDITIONAL_LINKS = {
    "Circuit": "flagquantum.circuit.Circuit",
    "flagquantum.Circuit": "flagquantum.circuit.Circuit",
    "ProgramRecord": "flagquantum.core.ir.program.model.ProgramRecord",
    "ComponentProgramBuilder": "flagquantum.compiler.registers.component_builder.ComponentProgramBuilder",
}


def make_reference_docs(
    repo_root_path: Path,
    refdoc_relpath: Path = Path("reference"),
    rootmod_dotpath: str = "flagquantum",
) -> None:
    """Generate the configured package's TOC, module pages and class pages.

    Optional Griffe and Markdown tooling follow the frozen documentation
    environment. Existing signature/decorator refusals propagate to the caller.
    """

    def sort_key(page: Page) -> list[str]:
        assert page.pref_path is not None, f"Uninitialized {page}"
        return page.pref_path.split(".")

    def in_section(page: Page, section: str) -> bool:
        return page.section == section or page.section in GROUPED_SECTIONS.get(
            section, ()
        )

    pages, context = walk_and_configure(
        rootmod_dotpath,
        refdoc_relpath,
        TOP_SECTIONS,
        [value for group in GROUPED_SECTIONS.values() for value in group],
        ADDITIONAL_LINKS,
    )
    output = repo_root_path / "docs" / refdoc_relpath
    # The original opens the TOC before creating this root. A fresh native
    # checkout owns this destination creation before invoking any renderer.
    output.mkdir(parents=True, exist_ok=True)
    render_toc(output, pages, in_section, sort_key, context)
    for page in pages:
        if isinstance(page, MajorClassPage):
            render_major_class(base_dir=output, page=page, render_context=context)
        elif isinstance(page, ModulePage):
            render_module(out_dir=output, page=page, render_context=context)
        else:
            raise ValueError(page)


def main() -> None:
    """Build the configured static reference for the current checkout."""
    make_reference_docs(get_git_root())


if __name__ == "__main__":
    main()
