"""Record complete comparator streams from the frozen source implementation."""

import argparse
import hashlib
import json
import subprocess
from itertools import product
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "fd73b653bd9ef09000e56021aab804a0b2733cf3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = set()
    for n, kind, relation, phase, control in product(
        (0, 1, 2, 3, 5, 9),
        ("b", "q", "k", "m"),
        ("lt", "le", "gt", "ge", "eq"),
        (0, 1),
        ("t", "b", "q"),
    ):
        if kind == "m" and n < 3:
            continue
        if relation not in ("lt", "gt") and control == "b":
            continue
        for clean, tolerance in (
            (n, "inf"),
            (max(3, n - 2), "inf"),
            (max(3, n - 2), "2.2"),
        ):
            if relation == "eq" and (clean < n or tolerance != "inf"):
                continue
            if kind == "m" and clean < n - 1 and tolerance == "inf":
                continue
            cases.add((n, clean, kind, relation, phase, control, tolerance))
    cases.update((256, work, "b", "lt", 0, "t", "inf") for work in (3, 256))
    fixture = []
    with source_oracle(
        args.source_root,
        "comparators/gen_cmp.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_comparison_file_oracle.cc"),
        dependencies=("comparators/gen_cmp_low_space.cc", "comparators/gen_cmp_qq.cc"),
    ) as oracle:
        for case in sorted(cases):
            text = subprocess.check_output(
                [str(oracle), *map(str, case)],
                text=True,
                timeout=5,
            )
            lines = [
                line
                for line in text.splitlines()
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
            fixture.append(json.dumps([*case, len(lines), digest]))
    body = ",\n".join(f"    {row}" for row in fixture)
    args.output.write_text(
        f'{{\n  "source_blob": "{SOURCE_BLOB}",\n  "cases": [\n{body}\n  ]\n}}\n'
    )


if __name__ == "__main__":
    main()
