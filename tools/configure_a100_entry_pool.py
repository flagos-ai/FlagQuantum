"""Inspect or partition two existing A100 CPU registrations without host changes."""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import TypedDict, cast

REPOSITORY = "flagos-ai/FlagQuantum"
ORDINARY = "flagquantum-a100-cpu"
GENERAL = "flagquantum-cpu-general"
ENTRY = "flagquantum-a100-cpu-entry"
RESERVED = frozenset({"fq-a100-cpu15", "fq-a100-cpu16"})


class Label(TypedDict):
    name: str
    type: str


class Runner(TypedDict):
    id: int
    name: str
    status: str
    busy: bool
    labels: list[Label]


class Inventory(TypedDict):
    total_count: int
    runners: list[Runner]


class Change(TypedDict):
    id: int
    name: str
    labels: list[str]


class Labels(TypedDict):
    labels: list[Label]


def api(path: str, labels: list[str] | None = None) -> object:
    """Bound each repository-scoped GitHub request; use the existing gh login."""
    command = ["gh", "api", f"repos/{REPOSITORY}/{path}"]
    payload = None
    if labels is not None:
        command += ["--method", "PUT", "--input", "-"]
        payload = json.dumps({"labels": labels})
    return json.loads(
        subprocess.check_output(command, input=payload, text=True, timeout=45)
    )


def partition(runners: list[Runner]) -> list[Change]:
    """Refuse an unexpected inventory or busy registration before changing labels."""
    names = {f"fq-a100-cpu{index:02d}" for index in range(1, 17)}
    pool = [runner for runner in runners if runner["name"] in names]
    if len(pool) != 16 or {runner["name"] for runner in pool} != names:
        raise ValueError("Expected exactly sixteen distinct A100 CPU registrations")
    if any(
        ENTRY in {label["name"] for label in runner["labels"]}
        and runner["name"] not in RESERVED
        for runner in runners
    ):
        raise ValueError("The entry label belongs only to the two reserved slots")
    changes: list[Change] = []
    for runner in pool:
        labels = {label["name"] for label in runner["labels"]}
        slot = runner["name"].removeprefix("fq-a100-")
        if not {"self-hosted", "Linux", "X64", "a100-cpu", slot} <= labels:
            raise ValueError("A100 host identity or validation slot labels changed")
        if runner["name"] not in RESERVED:
            if not {ORDINARY, GENERAL} <= labels or ENTRY in labels:
                raise ValueError("The fourteen ordinary registrations must stay intact")
            continue
        if runner["status"] != "online" or runner["busy"]:
            raise ValueError("Both reserved registrations must be online and idle")
        custom = {
            label["name"] for label in runner["labels"] if label["type"] == "custom"
        }
        desired = sorted((custom - {ORDINARY, GENERAL}) | {ENTRY})
        changes.append({"id": runner["id"], "name": runner["name"], "labels": desired})
    return sorted(changes, key=lambda change: change["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Replace only the two custom-label sets"
    )
    args = parser.parse_args()
    inventory = cast(Inventory, api("actions/runners?per_page=100"))
    if inventory["total_count"] > len(inventory["runners"]):
        raise ValueError("Inventory is paginated; do not apply an incomplete snapshot")
    changes = partition(inventory["runners"])
    print(json.dumps({"apply": args.apply, "changes": changes}, indent=2), flush=True)
    if args.apply:
        for change in changes:
            # Refresh both slots before every mutation; refuse an observed active job.
            current = cast(Inventory, api("actions/runners?per_page=100"))
            if current["total_count"] > len(current["runners"]):
                raise ValueError("Refuse an incomplete refreshed inventory")
            fresh = partition(current["runners"])
            if fresh != changes:
                raise ValueError("Registration identity changed after inspection")
            result = cast(
                Labels, api(f"actions/runners/{change['id']}/labels", change["labels"])
            )
            actual = {
                label["name"] for label in result["labels"] if label["type"] == "custom"
            }
            if actual != set(change["labels"]):
                raise ValueError("GitHub did not retain the complete custom-label set")
        print("Entry pool configured: 2 entry, 14 ordinary A100 CPU registrations")


if __name__ == "__main__":
    main()
