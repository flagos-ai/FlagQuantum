"""Summarize and validate the cold time-to-quality receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipts", type=Path, nargs="?", default=HERE / "receipts.json.gz"
    )
    args = parser.parse_args()
    if args.receipts.is_dir():
        evidence = {
            "protocol": json.loads((args.receipts / "protocol.json").read_text()),
            "rows": [
                json.loads(path.read_text())
                for path in sorted(args.receipts.glob("trial*.json"))
            ],
        }
    else:
        with gzip.open(args.receipts, "rt") as stream:
            evidence = json.load(stream)
    protocol = evidence["protocol"]
    rows = evidence["rows"]
    for name, expected in protocol["producer_sha256"].items():
        path = (
            HERE / name
            if name == "run.py"
            else HERE.parent / "torch_training_cpu_20261103" / name
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    grouped = {(r["trial"], r["case"], r["optimizer"], r["route"]): r for r in rows}
    assert len(grouped) == len(rows) == protocol["trials"] * 6 * 2 * 3
    max_loss_error = 0.0
    for trial in range(protocol["trials"]):
        for case in range(6):
            for optimizer in ("SGD", "Adam"):
                eager = grouped[trial, case, optimizer, "default_eager"]
                for route in (
                    "default_eager",
                    "forced_candidate",
                    "tensorcircuit_whole_jax",
                ):
                    row = grouped[trial, case, optimizer, route]
                    assert row["target"] == protocol["targets"][f"{case}-{optimizer}"]
                    if row["status"] != "passed":
                        continue
                    assert len(row["updates"]) == 60 and row["shape_probe"]
                    step = next(
                        (
                            i
                            for i, update in enumerate(row["updates"], 1)
                            if update["loss"] <= row["target"]
                        ),
                        None,
                    )
                    assert step == row["target_step"]
                    if eager["status"] == "passed":
                        max_loss_error = max(
                            max_loss_error,
                            *(
                                abs(a["loss"] - b["loss"])
                                for a, b in zip(
                                    eager["updates"], row["updates"], strict=True
                                )
                            ),
                        )
    print(f"Maximum paired pre-update loss error {max_loss_error:.3g}")
    assert max_loss_error <= 1e-8
    for route in ("default_eager", "forced_candidate", "tensorcircuit_whole_jax"):
        results = [row for row in rows if row["route"] == route]
        passed = [row for row in results if row["status"] == "passed"]
        hit = [row for row in passed if row["target_step"] is not None]
        print(
            f"{route}: passed {len(passed)}/{len(results)}, target reached {len(hit)}/{len(results)}"
        )
        for case in range(6):
            for optimizer in ("SGD", "Adam"):
                samples = [
                    row
                    for row in hit
                    if (row["case"], row["optimizer"]) == (case, optimizer)
                ]
                if samples:
                    seconds = statistics.median(
                        row["cold_target_seconds"] for row in samples
                    )
                    change = statistics.median(
                        row["shape_probe"]["first_changed_shape"]["seconds"]
                        for row in samples
                    )
                    print(
                        f"  {case} {optimizer}: target {seconds:.4f}s, shape change {change:.4f}s ({len(samples)} hits)"
                    )


if __name__ == "__main__":
    main()
