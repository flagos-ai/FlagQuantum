"""Publish sanitized capsules, then regenerate the retained comparisons offline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

from benchmarks.module_execution_matrix import aggregate_process_trials
from benchmarks.training_comparison_report import render_report
from tools.sanitize_public_evidence import sanitize_text

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parents[1]
PINS = {
    "before": "abde02fd05100b3ae2a65d68128d459cfd437dbe",
    "initial": "053adcfbae359c8204c3617d9e1dbf8ac524e248",
    "final": "97b315ea02863583e88d13a4127be65133485cf6",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_hash(revision, path):
    return digest(
        subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=ROOT, timeout=20
        )
    )


def publish(campaign_root, gpu):
    inventory = {}

    def capsule(name, directory):
        members = {}
        omitted = {}
        files = {
            p.relative_to(directory).as_posix(): p
            for p in directory.rglob("*")
            if p.is_file()
        }
        supervisor = directory.with_suffix(".log")
        if supervisor.exists():
            files["supervisor.log"] = supervisor
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for relative, path in sorted(files.items()):
                raw = path.read_bytes()
                if path.suffix == ".pt":
                    omitted[relative] = digest(raw)
                    continue
                public = sanitize_text(raw.decode()).encode()
                info = tarfile.TarInfo(relative)
                info.size = len(public)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(public))
                members[relative] = {
                    "raw_sha256": digest(raw),
                    "published_sha256": digest(public),
                }
        target = EVIDENCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(stream.getvalue(), mtime=0))
        inventory[name] = {
            "sha256": digest(target.read_bytes()),
            "members": members,
            "omitted_checkpoints": omitted,
        }

    for round_name in ("initial", "final"):
        prefix = "fq918-ir-" + ("final-" if round_name == "final" else "")
        for phase in ("before", "after"):
            for trial in range(3):
                capsule(
                    f"{round_name}/matrix-{phase}-{trial}.tar.gz",
                    campaign_root / f"{prefix}matrix-{trial}-{phase}",
                )
            capsule(
                f"{round_name}/training-{phase}.tar.gz",
                campaign_root / f"{prefix}training-{phase}",
            )
    capsule("gpu/one-gpu.tar.gz", gpu)
    for phase in ("before", "after"):
        raw = (campaign_root / f"fq918-ir-repro-{phase}.json").read_text()
        target = EVIDENCE / f"reproduction-{phase}.json"
        target.write_text(sanitize_text(raw))
    write_json(
        EVIDENCE / "publication.json",
        {
            "scope": "development CPU comparison and GPU conformance, not release or scalability",
            "source_commits": PINS,
            "capsules": inventory,
            "sanitizer": "tools.sanitize_public_evidence.sanitize_text",
            "checkpoints": "Raw .pt files are excluded; hashes and numeric replay receipts are retained.",
        },
    )


def load_capsule(name, record):
    data = (EVIDENCE / name).read_bytes()
    assert digest(data) == record["sha256"], name
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in contents and path.suffix != ".pt"
            value = archive.extractfile(member).read()
            assert digest(value) == record["members"][member.name]["published_sha256"]
            assert sanitize_text(value.decode()) == value.decode()
            contents[member.name] = value
    assert contents.keys() == record["members"].keys()
    return contents


def regenerate():
    publication = json.loads((EVIDENCE / "publication.json").read_text())
    assert publication["source_commits"] == PINS
    capsules = {
        name: load_capsule(name, record)
        for name, record in publication["capsules"].items()
    }
    generated = {}
    for round_name in ("initial", "final"):
        phases = {}
        training = {}
        for phase in ("before", "after"):
            pin = PINS["before"] if phase == "before" else PINS[round_name]
            phases[phase] = []
            for trial in range(3):
                payload = capsules[f"{round_name}/matrix-{phase}-{trial}.tar.gz"]
                summary = json.loads(payload["summary.json"])
                assert summary["revision"] == pin and summary["source_dirty"] is False
                assert summary["repeats"] == 5 and summary["iterations"] == 30
                assert len(summary["cases"]) == 10
                assert summary["producer_sha256"] == source_hash(
                    pin, "benchmarks/module_execution_matrix.py"
                )
                phases[phase].append(summary)
            name = f"{round_name}/training-{phase}.tar.gz"
            payload = capsules[name]
            freeze = json.loads(payload["freeze.json"])
            assert freeze["source"]["code_version"] == pin
            assert freeze["source"]["dirty"] is False
            for path, expected in freeze["source"]["producer_sha256"].items():
                assert source_hash(pin, path) == expected
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for path, raw in payload.items():
                    target = directory / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                generated[f"{round_name}/training-{phase}.md"] = render_report(
                    directory
                )
            summary = json.loads(payload["summary.json"])
            assert len(summary["cases"]) == 30
            training[phase] = {}
            for case in summary["cases"]:
                assert case["status"] == "completed"
                result = json.loads(payload[case["result"]])
                for field in ("parameter", "gradient", "optimizer"):
                    assert result[f"checkpoint_replay_max_{field}_error"] == 0
                checkpoint = str(Path(case["result"]).parent / "checkpoint.pt")
                assert (
                    result["checkpoint_sha256"]
                    == publication["capsules"][name]["omitted_checkpoints"][checkpoint]
                )
                training[phase][case["case"]] = result
        generated[f"{round_name}/aggregate.json"] = aggregate_process_trials(phases)
        comparison = []
        for case, before in training["before"].items():
            after = training["after"][case]
            error = max(
                abs(a - b)
                for a, b in zip(
                    before["final_parameters"], after["final_parameters"], strict=True
                )
            )
            assert error < 1e-10
            comparison.append(
                {
                    "case": case,
                    "final_parameter_max_error": error,
                    "before_training_seconds": before["training_seconds"],
                    "after_training_seconds": after["training_seconds"],
                    "before_over_after": before["training_seconds"]
                    / after["training_seconds"],
                }
            )
        generated[f"{round_name}/training-comparison.json"] = comparison
    gpu = json.loads(capsules["gpu/one-gpu.tar.gz"]["module-ir-cuda.json"])
    assert gpu["source_commit"] == PINS["final"] and len(gpu["cases"]) == 4
    assert gpu["producer_sha256"] == source_hash(
        PINS["final"], "tests/benchmark_contract/test_module_ir_cuda.py"
    )
    for case in gpu["cases"]:
        assert case["status"] == "passed" and case["builder_captures"] == 1
        assert max(case["maximum_absolute_errors"].values()) < 1e-10
        if case["backend"] == "jax":
            assert case["jax_array_platforms"] == ["gpu"]
            assert case["builder_lowering"] == "canonical_ir_fx_bindings"
    for phase in ("before", "after"):
        repro = json.loads((EVIDENCE / f"reproduction-{phase}.json").read_text())
        assert (
            repro["source_commit"] == PINS["before" if phase == "before" else "final"]
        )
        assert repro["source_dirty"] is False
        assert repro["producer_sha256"] == digest(
            (EVIDENCE / "validation/reproduce_builder.py").read_bytes()
        )
        assert sum(c["status"] == "completed" for c in repro["results"]) == (
            4 if phase == "before" else 8
        )
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish", type=Path, help="Root containing raw campaign directories"
    )
    parser.add_argument("--gpu", type=Path)
    args = parser.parse_args()
    if args.publish:
        if not args.gpu:
            parser.error("--publish requires --gpu")
        publish(args.publish, args.gpu)
    for relative, value in regenerate().items():
        target = EVIDENCE / relative
        if args.publish:
            if isinstance(value, str):
                target.write_text(value)
            else:
                write_json(target, value)
        else:
            actual = (
                target.read_text()
                if isinstance(value, str)
                else json.loads(target.read_text())
            )
            assert actual == value, relative
    print(
        "Verified both CPU rounds, all matched training/replay cases, reproductions and GPU conformance"
    )
