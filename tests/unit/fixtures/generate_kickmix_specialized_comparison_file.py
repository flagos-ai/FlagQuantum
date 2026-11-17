"""Record frozen specialized comparator streams and original prefix diagrams."""

import argparse
import hashlib
import json
import subprocess
from itertools import product
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "7de1b7a8bd5ba36226385d7c319a5260635ebd8e"
QUANTUM_BLOB = "7c618915a2b61ea26472652cf88bc79fe17c4096"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quantum = (
        args.source_root / "src/kickmix/gen/comparators/gen_cmp_qq.cc"
    ).read_bytes()
    assert (
        hashlib.sha1(f"blob {len(quantum)}\0".encode() + quantum).hexdigest()
        == QUANTUM_BLOB
    )
    cases = []
    for n, phase, ctrl, equal in product(
        (0, 1, 2, 3, 6, 9, 17), (0, 1), (0, 1), ("0", "1", "b", "q")
    ):
        for work, minimize in ((0, 0), (1, 0), (n + 1, 0), (n + 1, 1)):
            if work == 0 and n > 1 and equal != "q":
                continue
            cases.append(("qq", n, n, work, phase, ctrl, minimize, 0, equal, "0"))
    for n, phase, ctrl, classical in product(
        (0, 1, 2, 3, 7, 16, 41, 256), (0, 1), (0, 1), (0, 1)
    ):
        for m, equal, inverted in (
            (n, "0", "0"),
            (n, "q", "b"),
            (max(0, n - 2), "b", "q"),
            (n, "1", "1"),
        ):
            cases.append(("low", n, m, 3, phase, ctrl, 0, classical, equal, inverted))
    rows = []
    diagrams = []
    with source_oracle(
        args.source_root,
        "comparators/gen_cmp_low_space.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_specialized_comparison_oracle.cc"),
        dependencies=("comparators/gen_cmp_qq.cc",),
    ) as oracle:
        for case in sorted(set(cases)):
            raw = subprocess.check_output(
                [str(oracle), *map(str, case)],
                text=True,
                timeout=5,
            )
            lines = [
                line
                for line in raw.splitlines()
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
            rows.append(json.dumps([*case, len(lines), digest]))
        for clean in (0, 2):
            raw = subprocess.check_output(
                [str(oracle), "diagram", str(clean)],
                text=True,
                timeout=5,
            )
            kmx, diagram = raw.split("\n===DIAGRAM===\n", 1)
            diagrams.append(dict(clean=clean, kmx=kmx, diagram=diagram))
    body = ",\n".join(f"    {row}" for row in rows)
    args.output.write_text(
        f'{{\n  "source_blob": "{SOURCE_BLOB}",\n  "quantum_blob": "{QUANTUM_BLOB}",\n'
        f'  "diagrams": {json.dumps(diagrams)},\n  "cases": [\n{body}\n  ]\n}}\n'
    )


if __name__ == "__main__":
    main()
