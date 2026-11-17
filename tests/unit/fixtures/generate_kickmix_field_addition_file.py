"""Record whole physical field-addition invocations from the frozen source."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "4fec5c24910e0cb9f605d2572dc8e334f1e0e59b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases: list[tuple[int, str, str, int]] = []
    for n in (0, 1, 4, 9):
        for kind in ("q", "b", "k", "m", "z"):
            if kind == "m" and n < 4:
                continue
            controls: tuple[str, ...] = ("true", "q", "b", "false")
            if kind in ("m", "b") and n:
                controls += ("alias",)
            layouts = (0, 1, 2) if kind == "q" else (0,)
            cases.extend(
                (n, kind, control, layout) for control in controls for layout in layouts
            )
    cases.extend((512, "q", control, 0) for control in ("true", "q"))
    rows = []
    with source_oracle(
        args.source_root,
        "gf_arithmetic/gen_gf_iadd.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_field_addition_file_oracle.cc"),
    ) as oracle:
        for n, kind, control, layout in cases:
            raw = subprocess.check_output(
                [str(oracle), str(n), kind, control, str(layout)],
                text=True,
                timeout=5,
            )
            records = [
                line
                for line in raw.splitlines()
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            rows.append(
                {
                    "width": n,
                    "kind": kind,
                    "control": control,
                    "layout": layout,
                    "counts": dict(Counter(line.split()[0] for line in records)),
                    "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
                }
            )
    args.output.write_text(
        json.dumps({"source_blob": SOURCE_BLOB, "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
