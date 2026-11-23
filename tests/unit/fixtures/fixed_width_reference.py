"""Stream validated input/output records for the frozen scalar integer oracle."""

import json
import lzma
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


class IntegerCase(NamedTuple):
    operation: str
    width: int
    value: str
    right_width: int
    right: str
    third_width: int
    third: str
    scalar: int
    shift: int
    alias: int

    def source_input(self) -> str:
        fields = list(map(str, self))
        fields[2] = '"' + self.value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        return " ".join(fields) + "\n"


def read_cases(path: Path) -> Iterator[tuple[IntegerCase, str]]:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record: object = json.loads(line)
            if not isinstance(record, list) or len(record) != 2:
                raise ValueError("Reference record must contain input and output")
            raw, expected = record
            if (
                not isinstance(raw, list)
                or len(raw) != 10
                or not isinstance(expected, str)
            ):
                raise ValueError(
                    "Reference input must contain ten fields and a text result"
                )
            if any(type(raw[i]) is not int for i in (1, 3, 5, 7, 8, 9)) or any(
                not isinstance(raw[i], str) for i in (0, 2, 4, 6)
            ):
                raise ValueError("Reference operand field types are invalid")
            yield IntegerCase(*raw), expected
