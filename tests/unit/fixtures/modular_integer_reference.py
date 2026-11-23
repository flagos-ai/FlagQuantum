"""Typed input records for the original modular integer oracle."""

import json
import lzma
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


class ModularCase(NamedTuple):
    operation: str
    width: int
    value: str
    other_width: int
    other: str
    modulus_width: int
    modulus: str
    other_modulus: str
    scalar: int
    seed: int
    text: str

    def source_input(self) -> str:
        return " ".join(map(str, self[:-1])) + " " + json.dumps(self.text) + "\n"


def read_cases(path: Path) -> Iterator[tuple[ModularCase, str]]:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record: object = json.loads(line)
            if not isinstance(record, list) or len(record) != 2:
                raise ValueError("Reference record requires an input and output")
            fields, output = record
            if (
                not isinstance(fields, list)
                or len(fields) != 11
                or not isinstance(output, str)
                or any(type(fields[i]) is not int for i in (1, 3, 5, 8, 9))
                or any(not isinstance(fields[i], str) for i in (0, 2, 4, 6, 7, 10))
            ):
                raise ValueError("Modular reference operand types are invalid")
            yield ModularCase(*fields), output
