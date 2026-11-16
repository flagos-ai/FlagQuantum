# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked KMX writer and quoted string reader.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit

"""Quoted text shared by KMX metadata codecs."""

import re

_ESCAPES = {"\n": r"\n", "\r": r"\r", "#": r"\P", '"': r"\Q", "\\": r"\B"}
_ESCAPE = str.maketrans(_ESCAPES)
_UNESCAPE = {escaped: char for char, escaped in _ESCAPES.items()}
QUOTED_UNIT = r'(?:[^"\\#\r\n]|\\[nrPQB])'


def escape_kmx_text(text: str) -> str:
    return text.translate(_ESCAPE)


def unescape_kmx_text(text: str) -> str:
    """Decode text already checked against the quoted character grammar."""
    return re.sub(r"\\.", lambda match: _UNESCAPE[match[0]], text)
