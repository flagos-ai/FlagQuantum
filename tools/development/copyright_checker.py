# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 pylint_copyright_checker.py.

"""Optional Pylint check for retained Google Apache-2.0 source notices."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, BinaryIO

from astroid import nodes
from pylint.checkers import BaseRawFileChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter
    from pylint.typing import MessageDefinitionTuple


_MESSAGES: dict[str, MessageDefinitionTuple] = {
    "R0001": (
        "Missing or wrong copyright notice",
        "wrong-or-nonexistent-copyright-notice",
        "Consider putting a correct copyright notice at the beginning of a file.",
    )
}


def _notice_lines(stream: BinaryIO) -> Iterator[tuple[int, bytes]]:
    lines = iter(enumerate(stream))
    first = next(lines, None)
    if first is None:
        return
    if first[1].startswith(b"#!"):
        for item in lines:
            if item[1].strip():
                yield item
                break
    else:
        yield first
    yield from lines


class CopyrightChecker(BaseRawFileChecker):
    """Validate the original Apache notice, retaining file-level opt-out.

    This opt-in checker audits retained imported source headers. License
    ownership follows the inspected source.
    """

    name = "copyright-notice"
    msgs = _MESSAGES
    options = ()

    def process_module(self, node: nodes.Module) -> None:
        if not self.linter.is_message_enabled("wrong-or-nonexistent-copyright-notice"):
            return
        golden = [
            b"#  Copyright 20XX Google LLC",
            b"#",
            b'#  Licensed under the Apache License, Version 2.0 (the "License");',
            b"#  you may not use this file except in compliance with the License.",
            b"#  You may obtain a copy of the License at",
            b"#",
            b"#      https://www.apache.org/licenses/LICENSE-2.0",
            b"#",
            b"#  Unless required by applicable law or agreed to in writing, software",
            b'#  distributed under the License is distributed on an "AS IS" BASIS,',
            b"#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
            b"#  See the License for the specific language governing permissions and",
            b"#  limitations under the License.",
        ]
        with node.stream() as stream:
            lines = _notice_lines(stream)
            last_line = -1
            for index, expected in enumerate(golden):
                item = next(lines, None)
                if item is None:
                    self.add_message(
                        "wrong-or-nonexistent-copyright-notice",
                        line=last_line + 2,
                        col_offset=0,
                    )
                    return
                lineno, line = item
                last_line = lineno
                for column, (expected_char, char) in enumerate(
                    zip(expected, line, strict=False)
                ):
                    if expected_char != char and not (
                        index == 0 and 15 <= column <= 16
                    ):
                        self.add_message(
                            "wrong-or-nonexistent-copyright-notice",
                            line=lineno + 1,
                            col_offset=column,
                        )
                        return
                if len(line) < len(expected) or line[len(expected) :].strip():
                    self.add_message(
                        "wrong-or-nonexistent-copyright-notice",
                        line=lineno + 1,
                        col_offset=min(len(line), len(expected)),
                    )
                    return


def register(linter: PyLinter) -> None:
    """Register the opt-in checker with Pylint's plugin loader."""
    linter.register_checker(CopyrightChecker(linter))
