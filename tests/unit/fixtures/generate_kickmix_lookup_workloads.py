"""Record full physical lookup workloads from the unchanged source generator."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from kickmix_source_oracle import source_oracle

_SOURCE_BLOB = "d527eeeddf06c10ffe69ab6408594c60fd248182"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    with source_oracle(
        args.source_root,
        "lookup/gen_lookup.cc",
        _SOURCE_BLOB,
        Path(__file__).with_name("kickmix_lookup_workloads_oracle.cc"),
    ) as oracle:
        for kind in ("lookup", "erase"):
            raw = subprocess.check_output([str(oracle), kind], text=True, timeout=8)
            records = [
                line
                for line in raw.splitlines()
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            cases[kind] = {
                "counts": dict(Counter(line.split()[0] for line in records)),
                "records": len(records),
                "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
            }
    args.output.write_text(
        json.dumps({"source_blob": _SOURCE_BLOB, "cases": cases}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
