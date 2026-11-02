"""Fail-closed PF-00 receipts; no missing trial becomes a speedup or zero RSS."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

from .oracle import output_width
from .performance_workloads import cases, workloads
from .results import _cost, _integer, _object

STATUSES = {
    "verified",
    "resource_refused",
    "timeout",
    "unsupported",
    "validation_failed",
    "measurement_failed",
    "cancelled",
    "cold",
}


def validate_sample(sample: dict[str, object]) -> None:
    status = sample["status"]
    if status not in STATUSES or sample.get("process_group_reaped") is not True:
        raise ValueError("invalid status or unreaped process group")
    mode = sample["mode"]
    if mode not in {"timing", "memory", "diagnostic", "cold"}:
        raise ValueError("invalid measurement cohort")
    if status in {"verified", "resource_refused"}:
        elapsed = _integer(sample["e0_ns"])
        _integer(sample["first_call_ns"])
        if sample.get("gc_enabled") is not True:
            raise ValueError("normal GC must remain enabled")
        verification = _object(sample["verification"])
        if status == "verified":
            if (
                _integer(verification["basis_checks"]) < 1
                or verification["nonzero_targets"] is not True
            ):
                raise ValueError("successful sample lacks independent XOR checks")
        elif not sample.get("diagnostic_codes"):
            raise ValueError("refusal lacks diagnostics")
        if mode == "diagnostic":
            diagnostic = _object(sample["diagnostic"])
            spans = diagnostic["spans"]
            if not isinstance(spans, list):
                raise ValueError("missing diagnostic spans")
            total = 0
            for index, item in enumerate(spans):
                span = _object(item)
                if span["id"] != index or (
                    span["parent"] is not None
                    and not 0 <= _integer(span["parent"]) < index
                ):
                    raise ValueError("invalid parent span")
                inclusive, exclusive = _integer(span["inclusive_ns"]), _integer(
                    span["exclusive_ns"]
                )
                if exclusive > inclusive or _integer(span["calls"]) < 1:
                    raise ValueError("invalid span time/call count")
                if span["stage"] not in {"C1", "C2", "C3", "C4", "C5", "C6"}:
                    raise ValueError("unknown diagnostic stage")
                total += exclusive
            if (
                total != diagnostic["exclusive_sum_ns"]
                or total + _integer(diagnostic["unattributed_ns"]) != elapsed
            ):
                raise ValueError("nested time was double counted or lost")
        if mode in {"memory", "diagnostic"}:
            high = _object(sample["compile_lifetime_high_water"])
            if high["raw_unit"] not in {"bytes", "KiB"} or high["bytes"] != _integer(
                high["raw"]
            ) * (1024 if high["raw_unit"] == "KiB" else 1):
                raise ValueError("invalid lifetime high-water units")
            memory = _object(sample["memory"])
            points = memory["points"]
            if not isinstance(points, list) or not points:
                raise ValueError("missing raw RSS observations")
            window = []
            for value in points:
                point = _object(value)
                rss = _integer(point["parent_rss_bytes"])
                if (
                    point["raw_unit"] != "KiB"
                    or _integer(point["raw_parent_rss"]) * 1024 != rss
                ):
                    raise ValueError("invalid RSS units")
                if _integer(point["process_tree_rss_sum_bytes"]) < rss:
                    raise ValueError("tree sum excludes parent")
                if point["phase"] == "compile":
                    window.append(rss)
            if not window or memory["window_sampled_peak_rss_bytes"] != max(window):
                raise ValueError("compile RSS peak includes wrong scope")


def validate(payload: dict[str, object], *, formal: bool = True) -> None:
    if payload.get("schema") != "flagquantum.synthesis.python_performance.v1":
        raise ValueError("unknown PF-00 schema")
    config = _object(payload["configuration"])
    if (
        config["profile_enabled_in_primary_timing"] is not False
        or config["profile_enabled_in_primary_memory"] is not False
    ):
        raise ValueError("primary cohorts must not be profiled")
    count = _integer(config["formal_samples"])
    if formal and (
        count < 5
        or _integer(config["warmups_per_fresh_worker"]) < 2
        or payload["source_dirty"] is not False
    ):
        raise ValueError(
            "formal baseline requires clean pinned source, two warmups and five fresh samples"
        )
    inventory = [
        dict(
            case=w.case,
            frontend=w.frontend,
            variant=w.variant,
            target=w.target,
            refused=w.refused,
        )
        for w in workloads()
    ]
    if payload["fixed_workload_inventory"] != inventory:
        raise ValueError("fixed ninety-case corpus or supplements changed")
    cold = payload["cold"]
    if not isinstance(cold, list) or len(cold) != count:
        raise ValueError("missing cold trials")
    for sample in cold:
        validate_sample(_object(sample))
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("missing measured workloads")
    ids = []
    pids = []
    for item in rows:
        row = _object(item)
        ids.append(str(row["id"]))
        w = _object(row["workload"])
        if w not in inventory:
            raise ValueError("unrecognized workload")
        named = {w.identifier: w for w in workloads()}.get(str(row["id"]))
        if named is None or inventory[list(workloads()).index(named)] != w:
            raise ValueError("workload ID and parameters disagree")
        case = cases()[named.case]
        raw = row["raw"]
        if not isinstance(raw, list):
            raise ValueError("missing raw formal trials")
        modes = Counter(_object(s)["mode"] for s in raw)
        if modes != {
            "timing": count,
            "memory": config["memory_trials"],
            "diagnostic": 1,
        }:
            raise ValueError("missing trial; failures must remain explicit")
        identities = []
        for s in raw:
            sample = _object(s)
            validate_sample(sample)
            pids.append((sample["worker_pid"], sample.get("cold_start_to_import_ns")))
            if sample["status"] == "verified":
                verification = _object(sample["verification"])
                cost = _object(verification["resources"])
                # The authoritative cost reader validates counts, widths and state bytes.
                _cost(
                    cost,
                    sum(case.widths) + output_width(case),
                )
                identities.append(cost)
        if identities and any(identity != identities[0] for identity in identities):
            raise ValueError(
                "instrumentation or repeated trials changed selected circuit/resources"
            )
        from .performance import summary

        if row["timing"] != summary([_object(s) for s in raw], "timing"):
            raise ValueError("incorrect or selectively reported timing statistics")
    if len(set(ids)) != len(ids) or len(set(pids)) != len(pids):
        raise ValueError("duplicate workload or reused formal worker")
    expected = {w.identifier for w in workloads()}
    not_selected = payload["not_selected"]
    if (
        not isinstance(not_selected, list)
        or set(ids) & set(not_selected)
        or set(ids) | set(not_selected) != expected
    ):
        raise ValueError("missing/unreported workload")
    verify_sources(payload, formal=formal)


def pinned_revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("record a full lowercase commit/tree SHA")
    return value


def git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, check=False, timeout=15
    )
    if result.returncode:
        raise ValueError(
            "recorded measurement revision unavailable; fetch its exact commit before validation: "
            + result.stderr.decode(errors="replace").strip()
        )
    return result.stdout


def verify_sources(payload: dict[str, object], *, formal: bool) -> None:
    root = Path(__file__).resolve().parents[2]
    hashes = _object(payload["source_hashes"])
    dirty = payload["source_dirty"]
    if type(dirty) is not bool:
        raise ValueError("source_dirty must be boolean")
    if dirty:
        if formal:
            raise ValueError("formal evidence cannot use dirty sources")
        expected_paths = {
            str(p.relative_to(root))
            for p in (root / "benchmarks/synthesis").glob("*.py")
        }

        def contents(name: str) -> bytes:
            return (root / name).read_bytes()

    else:
        revision = pinned_revision(payload["source_revision"])
        production = pinned_revision(payload["production_baseline_revision"])
        tree = pinned_revision(payload["production_tree"])
        if (
            git_bytes(root, "rev-parse", revision + ":flagquantum").decode().strip()
            != tree
            or git_bytes(root, "rev-parse", production + ":flagquantum")
            .decode()
            .strip()
            != tree
        ):
            raise ValueError("measurement and pinned production trees differ")
        names = (
            git_bytes(
                root,
                "ls-tree",
                "-r",
                "--name-only",
                revision,
                "--",
                "benchmarks/synthesis",
            )
            .decode()
            .splitlines()
        )
        expected_paths = {
            name
            for name in names
            if Path(name).parent == Path("benchmarks/synthesis")
            and name.endswith(".py")
        }

        def contents(name: str) -> bytes:
            return git_bytes(root, "show", revision + ":" + name)

    if set(hashes) != expected_paths:
        raise ValueError("measurement source manifest is incomplete")
    for name, expected_hash in hashes.items():
        if (
            not name.startswith("benchmarks/synthesis/")
            or ".." in Path(name).parts
            or hashlib.sha256(contents(name)).hexdigest() != expected_hash
        ):
            raise ValueError("recorded measurement script/oracle bytes changed")
    artifacts = _object(payload["original_resource_hashes"])
    expected_artifacts = {
        p.name for p in (root / "benchmarks/synthesis/baselines").glob("ao*.json")
    }
    if set(artifacts) != expected_artifacts:
        raise ValueError("historical resource manifest is incomplete")
    for name, expected_hash in artifacts.items():
        path = "benchmarks/synthesis/baselines/" + name
        if (
            Path(name).name != name
            or hashlib.sha256((root / path).read_bytes()).hexdigest() != expected_hash
            or hashlib.sha256(contents(path)).hexdigest() != expected_hash
        ):
            raise ValueError("historical resource artifact changed")


def archive_bytes(index_path: Path) -> bytes:
    index = _object(json.loads(index_path.read_text()))
    if index.get("schema") != "flagquantum.synthesis.gzip_parts.v1":
        raise ValueError("unknown segmented receipt format")
    parts = index["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 64:
        raise ValueError("invalid archive part count")
    blocks = []
    for value in parts:
        part = _object(value)
        name = part["file"]
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("archive parts must be sibling files")
        path = index_path.parent / name
        size = path.stat().st_size
        if not 0 < size <= 1_800_000 or size != part["bytes"]:
            raise ValueError("invalid archive part size")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != part["sha256"]:
            raise ValueError("archive part checksum differs")
        blocks.append(raw)
    raw = b"".join(blocks)
    if (
        len(raw) != index["gzip_bytes"]
        or hashlib.sha256(raw).hexdigest() != index["gzip_sha256"]
    ):
        raise ValueError("reassembled gzip differs")
    return raw


def read(path: Path, *, formal: bool = True) -> dict[str, object]:
    if path.name.endswith(".parts.json"):
        with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes(path))) as source:
            raw = source.read((512 << 20) + 1)
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as source:
            raw = source.read((512 << 20) + 1)
    if len(raw) > 512 << 20:
        raise ValueError("PF-00 receipt exceeds 512 MiB")
    data = _object(json.loads(raw))
    validate(data, formal=formal)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    data = read(args.path, formal=not args.development)
    if args.summary is not None:
        args.summary.write_text(
            json.dumps(digest_summary(data), indent=2, sort_keys=True) + "\n"
        )
    print(
        "Validated PF-00 workloads:",
        len(data["rows"]) if isinstance(data["rows"], list) else 0,
    )


def digest_summary(payload: dict[str, object]) -> dict[str, object]:
    """Small review table; raw samples remain authoritative in the full receipt."""
    import statistics

    def distribution(values: list[int]) -> dict[str, object]:
        return {
            "n": len(values),
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("missing rows")
    result = []
    for value in rows:
        row = _object(value)
        raw = row["raw"]
        if not isinstance(raw, list):
            raise ValueError("missing samples")
        trials = [_object(s) for s in raw]
        diagnostics = [
            _object(s["diagnostic"])
            for s in trials
            if s["mode"] == "diagnostic"
            and s["status"] in {"verified", "resource_refused"}
        ]
        stage_totals: dict[str, int] = {}
        watched: Counter[str] = Counter()
        for diagnostic in diagnostics:
            spans = diagnostic["spans"]
            if not isinstance(spans, list):
                raise ValueError("missing spans")
            for span_value in spans:
                span = _object(span_value)
                stage = str(span["stage"])
                stage_totals[stage] = stage_totals.get(stage, 0) + _integer(
                    span["exclusive_ns"]
                )
                name = str(span["function"])
                if any(
                    token in name
                    for token in (
                        "plan_to_dict",
                        "to_json",
                        "from_json",
                        "to_dict",
                        "from_dict",
                        "content_hash",
                        "Instruction.__init__",
                        "validate_plan_payload",
                    )
                ):
                    watched[name] += _integer(span["calls"])
        memories = [
            s
            for s in trials
            if s["mode"] == "memory" and s["status"] in {"verified", "resource_refused"}
        ]
        e1 = []
        for sample in trials:
            if sample["mode"] == "timing" and sample["status"] == "verified":
                verified = _object(sample["verification"])
                end = _object(verified["e1"])
                if end["status"] == "verified":
                    e1.append(end)
        result.append(
            {
                "id": row["id"],
                "timing": row["timing"],
                "statuses": dict(Counter(str(s["status"]) for s in trials)),
                "first_call_ns": distribution(
                    [
                        _integer(s["first_call_ns"])
                        for s in trials
                        if s["mode"] == "timing" and "first_call_ns" in s
                    ]
                ),
                "run_ns": distribution([_integer(s["run_ns"]) for s in e1]),
                "named_read_ns": distribution(
                    [_integer(s["named_read_ns"]) for s in e1]
                ),
                "e0_plus_run_plus_read_ns": distribution(
                    [_integer(s["e0_plus_run_plus_read_ns"]) for s in e1]
                ),
                "compile_window_sampled_peak_rss_bytes": distribution(
                    [
                        _integer(_object(s["memory"])["window_sampled_peak_rss_bytes"])
                        for s in memories
                    ]
                ),
                "lifetime_high_water_bytes": distribution(
                    [
                        _integer(_object(s["compile_lifetime_high_water"])["bytes"])
                        for s in memories
                    ]
                ),
                "diagnostic_exclusive_ns_by_stage": stage_totals,
                "diagnostic_watched_calls": dict(watched),
                "diagnostic_scope": "intrusive separate trial; not unprofiled stage latency; inclusive spans remain nested in raw receipt",
            }
        )
    return {
        k: payload[k]
        for k in [
            "schema",
            "source_revision",
            "production_baseline_revision",
            "configuration",
            "environment",
            "source_hashes",
            "original_resource_hashes",
        ]
    } | {"rows": result, "not_selected": payload["not_selected"]}


if __name__ == "__main__":
    main()
