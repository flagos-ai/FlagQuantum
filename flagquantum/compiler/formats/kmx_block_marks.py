# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked CircuitBuilder deferred marks and KMX quoted text.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix

"""Round-trip named component ranges alongside the reversible-phase stream."""

import re
from collections.abc import Sequence

from .._component_trace.trace import ComponentCall
from ..artifacts.reversible_phase import ReversiblePhaseProgram
from ._kmx_text import QUOTED_UNIT, escape_kmx_text, unescape_kmx_text
from .kmx import decode_kmx, encode_kmx

_MARK = re.compile(rf'INFO_TAG "@(enter|exit):({QUOTED_UNIT}+)"')


def encode_kmx_block_marks(
    program: ReversiblePhaseProgram, calls: Sequence[ComponentCall]
) -> str:
    """Encode named ranges indexed by lowered KMX operations, including scopes."""
    header, *operations = encode_kmx(program).splitlines(keepends=True)
    lines = [header]

    def emit(start: int, stop: int, children: Sequence[ComponentCall]) -> None:
        for child in children:
            if not start <= child.start <= child.stop <= stop:
                raise ValueError("Invalid or overlapping KMX component range")
            if not child.name or child.parameters:
                raise ValueError("KMX block marks require a name and no parameters")
            lines.extend(operations[start : child.start])
            name = escape_kmx_text(child.name)
            lines.append(f'INFO_TAG "@enter:{name}"\n')
            emit(child.start, child.stop, child.children)
            lines.append(f'INFO_TAG "@exit:{name}"\n')
            start = child.stop
        lines.extend(operations[start:stop])

    emit(0, len(operations), calls)
    return "".join(lines)


def decode_kmx_block_marks(
    text: str,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Extract balanced named marks, leaving operation validation to the codec."""
    header, *body = text.removesuffix("\n").split("\n")
    operations: list[str] = []
    roots: list[ComponentCall] = []
    stack: list[tuple[str, int, list[ComponentCall]]] = []
    for line in body:
        mark = _MARK.fullmatch(line)
        if mark is None:
            if line.startswith(('INFO_TAG "@enter:', 'INFO_TAG "@exit:')):
                raise ValueError("Invalid KMX block mark")
            operations.append(line)
            continue
        action, label = mark.groups()
        name = unescape_kmx_text(label)
        if action == "enter":
            stack.append((name, len(operations), []))
        else:
            if not stack or stack[-1][0] != name:
                raise ValueError("Unmatched KMX block exit")
            _, start, children = stack.pop()
            siblings = stack[-1][2] if stack else roots
            siblings.append(
                ComponentCall(name, start, len(operations), children=tuple(children))
            )
    if stack:
        raise ValueError("Unclosed KMX block mark")
    return decode_kmx("\n".join((header, *operations))), tuple(roots)
