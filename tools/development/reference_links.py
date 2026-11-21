# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_reference_docs/_linking_writer.py and
# make_reference_docs/_components/aliases.py.

"""Reference-style Markdown links and ordered public alias descriptions."""

import re
from pathlib import Path
from typing import Protocol

from mdit_py_plugins.anchors.index import slugify

from tools.development.reference_pages import Page


class Writable(Protocol):
    def write(self, text: str) -> object: ...


def get_aliases_str(pref_dotpath: str, *, aliases_d: dict[str, str]) -> str:
    """Describe the preferred path and aliases in dictionary insertion order."""
    aliases = [
        key
        for key, value in aliases_d.items()
        if value == pref_dotpath and key != pref_dotpath
    ]
    result = f"`{pref_dotpath}`"
    alias_text = ", ".join(f"`{alias}`" for alias in aliases)
    if len(aliases) > 1:
        result += f". **Aliases:** {alias_text}"
    elif len(aliases) == 1:
        result += f". **Alias:** {alias_text}"
    return result


class LinkingWriter:
    """Insert reference links and emit their targets after the document body.

    write_nl writes text verbatim. Each writer resolves against its own page
    mapping, so another document using the same output root cannot poison it.
    Target order retains the original set iteration behavior.
    """

    def __init__(
        self,
        f: Writable,
        *,
        link_aliases: dict[str, str],
        link_d: dict[str, tuple[Page, str | None]],
        refdoc_relpath: Path,
    ) -> None:
        self._f = f
        self._linked: set[str] = set()
        self._link_aliases = link_aliases
        self._link_d = link_d
        self._reference_relpath = refdoc_relpath

    def resolve_dotpath(self, dotpath: str) -> str | None:
        try:
            page, subname = self._link_d[self._link_aliases[dotpath]]
        except KeyError:
            return None
        path = page.out_path(out_dir=self._reference_relpath)
        if subname is None:
            return f"/{path}"
        return f"/{path}#{slugify(subname)}"

    def repl(self, match: re.Match[str]) -> str:
        name = match.group(1)
        if self.resolve_dotpath(name) is None:
            return f"`{name}`"
        self._linked.add(name)
        return f"[`{name}`]"

    def linkify(self, text: str) -> str:
        return re.sub(r"`([\w\.]+)`", self.repl, text)

    def write(self, text: str) -> None:
        self._f.write(self.linkify(text))

    def write_nl(self, text: str) -> None:
        self._f.write(text)

    def write_link_targets(self) -> None:
        self._f.write("\n")
        for name in self._linked:
            self._f.write(f"[`{name}`]: {self.resolve_dotpath(name)}\n")
