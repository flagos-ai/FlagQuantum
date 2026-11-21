# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_reference_docs/_components/render_docstring.py.

"""Render Griffe Google-style docstring sections into reference Markdown."""

import re
import warnings

from griffe import (
    DocstringSection,
    DocstringSectionAdmonition,
    DocstringSectionAttributes,
    DocstringSectionExamples,
    DocstringSectionKind,
    DocstringSectionParameters,
    DocstringSectionRaises,
    DocstringSectionReturns,
    DocstringSectionText,
    DocstringSectionYields,
    Object,
)

from tools.development.docstrings import parse_references
from tools.development.reference_links import Writable


def split_docstring(obj: Object) -> tuple[str, list[DocstringSection]]:
    if obj.docstring is None:
        return "", []
    first, *parts = obj.docstring.parse("google")
    if first.kind is not DocstringSectionKind.text:
        raise ValueError(obj)
    if not isinstance(first, DocstringSectionText):
        raise TypeError("text docstring sections must use DocstringSectionText")
    summary, *paragraphs = re.split(r"\n{2,}", first.value, flags=re.MULTILINE)
    remainder = "\n\n".join(paragraphs).strip()
    if remainder:
        return summary, [DocstringSectionText(remainder), *parts]
    return summary, parts


def _write_text(f: Writable, part: DocstringSectionText, level: int) -> None:
    f.write(part.value.strip())
    f.write("\n\n")


def _write_parameters(
    f: Writable, part: DocstringSectionParameters, level: int
) -> None:
    f.write(f"###{'#' * level} Args\n")
    for parameter in part.value:
        if parameter.annotation:
            f.write(f"{parameter.name}: `{parameter.annotation}`\n")
        else:
            f.write(f"{parameter.name}\n")
        f.write(f": {parameter.description}\n\n")


def _write_attributes(
    f: Writable, part: DocstringSectionAttributes, level: int
) -> None:
    f.write(f"###{'#' * level} Attributes\n")
    for attribute in part.value:
        if attribute.annotation:
            f.write(f"{attribute.name}: `{attribute.annotation}`\n")
        else:
            f.write(f"{attribute.name}\n")
        f.write(f": {attribute.description}\n\n")


def _write_returns(f: Writable, part: DocstringSectionReturns, level: int) -> None:
    f.write(f"###{'#' * level} Returns\n")
    if len(part.value) == 1 and not part.value[0].name:
        f.write(part.value[0].description)
        f.write("\n")
    else:
        for parameter in part.value:
            name = parameter.name if parameter.name else "ret"
            f.write(f"{name}\n: {parameter.description}\n\n")


def _write_admonition(
    f: Writable, part: DocstringSectionAdmonition, level: int
) -> None:
    value = part.value
    if value.kind == "see-also":
        f.write(f"###{'#' * level} See Also\n")
        f.write(value.description)
        f.write("\n\n")
    elif value.kind == "references":
        f.write(f"###{'#' * level} References\n")
        for reference in parse_references(value.contents):
            f.write(f" - {reference.text}\n")
        f.write("\n")
    elif value.kind == "note":
        f.write(f"**Note:** {value.contents}\n\n")
    else:
        warnings.warn(f"Unknown admonition type {value.kind}")


def _write_examples(f: Writable, part: DocstringSectionExamples, level: int) -> None:
    for kind, text in part.value:
        if kind is DocstringSectionKind.examples:
            f.write(f"```pycon\n{text}\n````")
        elif kind is DocstringSectionKind.text:
            f.write(text)
        else:
            raise ValueError()
        f.write("\n")
    f.write("\n\n")


def _write_raises(f: Writable, part: DocstringSectionRaises, level: int) -> None:
    f.write(f"###{'#' * level} Raises\n")
    for error in part.value:
        if not error.annotation:
            raise ValueError()
        f.write(f"{error.annotation}\n: {error.description}\n\n")


def _write_yields(f: Writable, part: DocstringSectionYields, level: int) -> None:
    f.write(f"###{'#' * level} Yields\n")
    if len(part.value) == 1 and not part.value[0].name:
        f.write(part.value[0].description)
        f.write("\n")
    else:
        for parameter in part.value:
            name = parameter.name if parameter.name else "yld"
            f.write(f"{name}\n: {parameter.description}\n\n")


def _write_unknown(f: Writable, part: DocstringSection, level: int) -> None:
    f.write(str(part))
    f.write("\n")
    warnings.warn(f"Unknown docstring {part}")


def write_docstring_parts(
    f: Writable, parts: list[DocstringSection], level: int
) -> None:
    """Render the original eight section types and warn for other sections."""
    for part in parts:
        if isinstance(part, DocstringSectionText):
            _write_text(f, part, level)
        elif isinstance(part, DocstringSectionParameters):
            _write_parameters(f, part, level)
        elif isinstance(part, DocstringSectionReturns):
            _write_returns(f, part, level)
        elif isinstance(part, DocstringSectionAdmonition):
            _write_admonition(f, part, level)
        elif isinstance(part, DocstringSectionExamples):
            _write_examples(f, part, level)
        elif isinstance(part, DocstringSectionRaises):
            _write_raises(f, part, level)
        elif isinstance(part, DocstringSectionYields):
            _write_yields(f, part, level)
        elif isinstance(part, DocstringSectionAttributes):
            _write_attributes(f, part, level)
        else:
            _write_unknown(f, part, level)
    f.write("\n\n")
