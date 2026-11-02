"""Compare AO-01 resources with the immutable AO-00 mathematical corpus.

Generate the current artifact with the existing arithmetic_baseline runner;
this module adds comparison only, not another estimator or execution engine.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from .results import _object, read_result

METRICS = ("allocated_qubits", "workspace_qubits", "operations", "depth")
FROZEN_SHA256 = "c5f084ae48066f8055a94a3be02d0d30d71c233783c7fd1234ccb2b3d464c864"


@dataclass(frozen=True)
class Difference:
    case: str
    before_status: str
    after_status: str
    classification: str
    deltas: dict[str, int]


class Comparison(TypedDict):
    schema: str
    artifact_sha256: dict[str, str]
    summary: dict[str, int]
    rows: list[dict[str, object]]


def differences(before: Path, after: Path) -> tuple[Difference, ...]:
    if hashlib.sha256(before.read_bytes()).hexdigest() != FROZEN_SHA256:
        raise ValueError("AO-00 historical artifact changed")
    old, new = read_result(before), read_result(after)
    if old["configuration"] != new["configuration"]:
        raise ValueError("incompatible corpus or budget configuration")
    old_rows, new_rows = old["rows"], new["rows"]
    assert isinstance(old_rows, list) and isinstance(new_rows, list)
    result = []
    for old_row, new_row in zip(old_rows, new_rows, strict=True):
        a, b = _object(old_row), _object(new_row)
        deltas: dict[str, int] = {}
        kind = "incomplete"
        if a["status"] == b["status"] == "compiled":
            left, right = _object(a["resources"]), _object(b["resources"])
            for name in METRICS:
                x, y = left[name], right[name]
                if type(x) is not int or type(y) is not int:
                    raise ValueError("resource metrics require plain integers")
                deltas[name] = y - x
            better, worse = any(d < 0 for d in deltas.values()), any(
                d > 0 for d in deltas.values()
            )
            kind = (
                "tradeoff"
                if better and worse
                else "regressed" if worse else "improved" if better else "unchanged"
            )
        result.append(
            Difference(str(a["case"]), str(a["status"]), str(b["status"]), kind, deltas)
        )
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = differences(args.baseline, args.current)
    payload: Comparison = {
        "schema": "flagquantum.synthesis.normalization_comparison.v1",
        "artifact_sha256": {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in (("baseline", args.baseline), ("current", args.current))
        },
        "summary": dict(Counter(row.classification for row in rows)),
        "rows": [asdict(row) for row in rows],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["summary"])


if __name__ == "__main__":
    main()
