"""Submit one Bell circuit only when --submit is explicitly supplied."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import flagquantum as fq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", required=True, help="Current workshop target, e.g. quafu:Dongling"
    )
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.target.startswith("quafu:") or not args.target.split(":", 1)[1]:
        parser.error("target must name a Quafu backend")
    if args.shots <= 0 or args.shots % 1024:
        parser.error("shots must be a positive multiple of 1024")
    if not args.submit:
        print(
            "Preview only: Bell circuit, compiler=qsteed, target="
            + args.target
            + ", shots="
            + str(args.shots)
        )
        return
    if not os.getenv("QUAFU_API_TOKEN"):
        parser.error("Set QUAFU_API_TOKEN outside the notebook/source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path before submission. An uncertain outcome must not be retried blindly.
    record = {
        "source": "live",
        "target": args.target,
        "shots": args.shots,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submission_started",
    }
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2)
    result = fq.run(
        fq.Circuit(2).h(0).cx(0, 1),
        compiler="qsteed",
        target=args.target,
        shots=args.shots,
        name="flagos2026_bell",
    )
    record.update(
        status="completed",
        task_id=str(result.provenance["task_id"]),
        counts=result.counts[0],
    )
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
