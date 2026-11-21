# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_reference_docs/_render_context.py.

"""Reference documentation context shared by the page renderers."""

from dataclasses import dataclass
from pathlib import Path

from tools.development.reference_links import LinkingWriter, Writable
from tools.development.reference_pages import Page


@dataclass
class RenderContext:
    refdoc_relpath: Path
    sections: list[str]
    aliases_d: dict[str, str]
    link_d: dict[str, tuple[Page, str | None]]
    linkable_to_prefpath: dict[str, str]

    def get_linking_writer(self, f: Writable) -> LinkingWriter:
        return LinkingWriter(
            f,
            link_aliases=self.linkable_to_prefpath,
            link_d=self.link_d,
            refdoc_relpath=self.refdoc_relpath,
        )
