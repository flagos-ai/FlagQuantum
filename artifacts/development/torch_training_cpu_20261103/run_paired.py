"""Counterbalance clean before/after sources for public no-callback training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {"before": args.before.resolve(), "after": args.after.resolve()}
    records: list[dict[str, Any]] = []
    failed = False
    for trial in range(args.trials):
        phases = ("before", "after") if trial % 2 == 0 else ("after", "before")
        for phase in phases:
            output = args.output / f"{phase}-{trial}"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("reproduce.py")),
                    "--output",
                    str(output),
                    "--without-callback",
                    "--only-default",
                    "--trials",
                    "1",
                ],
                env={**os.environ, "FQ_TRAINING_SOURCE_ROOT": str(sources[phase])},
                timeout=1800,
            )
            failed = failed or completed.returncode != 0
            for record in json.loads((output / "campaign.json").read_text())["records"]:
                records.append({**record, "paired_round": trial, "phase": phase})
            (args.output / "paired.json").write_text(
                json.dumps(
                    {
                        "records": records,
                        "phase_order": "before-after, after-before, before-after",
                        "native_extension_sha256": {
                            key: hashlib.sha256(
                                (
                                    source
                                    / "flagquantum/simulation/native_cpu/_C.abi3.so"
                                ).read_bytes()
                            ).hexdigest()
                            for key, source in sources.items()
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
