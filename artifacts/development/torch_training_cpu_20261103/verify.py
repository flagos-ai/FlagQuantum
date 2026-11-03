"""Verify source pins/publication bytes and regenerate every training comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq

HERE = Path(__file__).resolve().parent
TOLERANCE = 1e-10


def maximum_error(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        assert set(left) == set(right)
        return max((maximum_error(left[key], right[key]) for key in left), default=0.0)
    return (
        float(
            (
                torch.as_tensor(left, dtype=torch.float64)
                - torch.as_tensor(right, dtype=torch.float64)
            )
            .abs()
            .max()
        )
        if isinstance(left, list) and left
        else abs(float(left) - float(right))
    )


def load_capsules() -> dict[str, dict[str, Any]]:
    publication = json.loads((HERE / "publication.json").read_text())
    capsules: dict[str, dict[str, Any]] = {}
    for filename, evidence in publication["capsules"].items():
        assert (
            hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
            == evidence["sha256"]
        )
        contents = {}
        with tarfile.open(HERE / filename, "r:gz") as archive:
            assert set(archive.getnames()) == set(evidence["members"])
            for member in archive.getmembers():
                assert (
                    member.isfile()
                    and member.mtime == 0
                    and member.uid == member.gid == 0
                )
                stream = archive.extractfile(member)
                assert stream is not None
                payload = stream.read()
                assert (
                    hashlib.sha256(payload).hexdigest()
                    == evidence["members"][member.name]["published_sha256"]
                )
                if member.name.endswith(".json"):
                    contents[member.name] = json.loads(payload)
        capsules[filename] = contents
    for pin, files in publication["sources"].items():
        assert len(pin) == 40
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", pin, "HEAD"], check=True, timeout=15
        )
        for path, digest in files.items():
            original = subprocess.check_output(
                ["git", "show", f"{pin}:{path}"], timeout=15
            )
            assert hashlib.sha256(original).hexdigest() == digest
    for filename, evidence in publication["validation"]["files"].items():
        compressed = (HERE / "validation" / filename).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == evidence["gzip_sha256"]
        assert (
            hashlib.sha256(gzip.decompress(compressed)).hexdigest()
            == evidence["published_sha256"]
        )
    return capsules


def consumed_hash_checks(records: list[dict[str, Any]]) -> int:
    checked = 0
    for record in records:
        width, depth, batch, layout = record["shape"]
        generator = torch.Generator().manual_seed(20261102)
        parameters = (
            torch.randn(2 * width, generator=generator, dtype=torch.float64) * 0.2
        )
        features = (
            torch.randn(batch, width, generator=generator, dtype=torch.float64) * 0.3
        )
        for index, update in enumerate(record["updates"]):
            circuit = fq.Circuit(width, bsz=batch, dtype=torch.complex128)
            data = features + index * 0.0005
            for layer in range(depth):
                for wire in range(width):
                    circuit.ry(
                        wire,
                        parameters[wire].square()
                        + parameters[width + wire] * data[:, wire].sin()
                        + layer * 0.01,
                    )
                    circuit.rz(
                        wire, parameters[width + wire] * 0.3 + parameters[wire].cos()
                    )
                for wire in range(width - 1):
                    circuit.cx(wire, wire + 1)
                if layout == "ring":
                    circuit.cx(width - 1, 0)
            assert circuit.to_ir().content_hash == update["ir_hash"]
            parameters = torch.tensor(update["parameters"], dtype=torch.float64)
            checked += 1
    return checked


def summary() -> dict[str, Any]:
    capsules = load_capsules()
    callback = capsules["callback-summary.tar.gz"]["campaign.json"]["records"]
    paired = capsules["unobserved.tar.gz"]["paired.json"]
    unobserved = paired["records"]
    external = capsules["whole_jax.tar.gz"]["campaign.json"]["records"]
    assert len(callback) == len(unobserved) == 72 and len(external) == 36
    assert len(set(paired["native_extension_sha256"].values())) == 1
    for records, dimensions in ((callback, "route"), (unobserved, "phase")):
        assert (
            len(
                {
                    (
                        r.get("paired_round", r["trial"]),
                        r["case"],
                        r["optimizer"],
                        r[dimensions],
                    )
                    for r in records
                }
            )
            == 72
        )
    assert len({(r["trial"], r["case"], r["optimizer"]) for r in external}) == 36
    for record in (*callback, *unobserved, *external):
        assert record["status"] == "passed" and not record["source_dirty"]
        assert record["peak_worker_rss_bytes"] <= 2 * 1024**3
        assert record["recovery_parameter_max_error"] <= TOLERANCE
        assert len(record["losses"]) == 60
    max_errors = {
        "compiled_loss": 0.0,
        "compiled_parameter": 0.0,
        "compiled_gradient": 0.0,
        "whole_jax_loss": 0.0,
        "whole_jax_parameter": 0.0,
        "whole_jax_gradient": 0.0,
        "unobserved_loss": 0.0,
        "unobserved_parameter": 0.0,
    }
    different_hashes = 0
    for record in (*callback, *external):
        if record["route"] == "default_eager":
            continue
        reference = next(
            r
            for r in callback
            if r["trial"] == record["trial"]
            and r["case"] == record["case"]
            and r["optimizer"] == record["optimizer"]
            and r["route"] == "default_eager"
        )
        prefix = "compiled" if record["route"] == "forced_candidate" else "whole_jax"
        for key, name in (
            ("loss", "loss"),
            ("parameters", "parameter"),
            ("gradients", "gradient"),
        ):
            error = maximum_error(
                [r[key] for r in record["updates"]],
                [r[key] for r in reference["updates"]],
            )
            assert error <= TOLERANCE
            max_errors[f"{prefix}_{name}"] = max(max_errors[f"{prefix}_{name}"], error)
        if prefix == "compiled":
            assert (
                maximum_error(record["final_optimizer"], reference["final_optimizer"])
                <= TOLERANCE
            )
            different_hashes += sum(
                a["ir_hash"] != b["ir_hash"]
                for a, b in zip(record["updates"], reference["updates"], strict=True)
            )
    for record in unobserved:
        if record["phase"] != "after":
            continue
        reference = next(
            r
            for r in unobserved
            if r["phase"] == "before"
            and r["paired_round"] == record["paired_round"]
            and r["case"] == record["case"]
            and r["optimizer"] == record["optimizer"]
        )
        for key, name in (("losses", "loss"), ("final_parameters", "parameter")):
            error = maximum_error(record[key], reference[key])
            assert error <= TOLERANCE
            max_errors[f"unobserved_{name}"] = max(
                max_errors[f"unobserved_{name}"], error
            )
        assert (
            maximum_error(record["final_optimizer"], reference["final_optimizer"])
            <= TOLERANCE
        )
    rows = []
    for case in range(6):
        for kind in ("SGD", "Adam"):

            def median(
                records: list[dict[str, Any]],
                field: str,
                *,
                _case: int = case,
                _kind: str = kind,
                **selection: Any,
            ) -> float:
                return statistics.median(
                    float(r[field])
                    for r in records
                    if r["case"] == _case
                    and r["optimizer"] == _kind
                    and all(r[key] == value for key, value in selection.items())
                )

            before = median(unobserved, "training_seconds", phase="before")
            after = median(unobserved, "training_seconds", phase="after")
            eager = median(callback, "training_seconds", route="default_eager")
            compiled = median(callback, "training_seconds", route="forced_candidate")
            jax = median(external, "training_seconds")
            rows.append(
                {
                    "case": case,
                    "optimizer": kind,
                    "default_before_seconds": before,
                    "default_after_seconds": after,
                    "default_before_after_ratio": before / after,
                    "callback_eager_seconds": eager,
                    "callback_candidate_seconds": compiled,
                    "callback_candidate_eager_total_ratio": compiled / eager,
                    "whole_jax_seconds": jax,
                    "whole_jax_first_update_seconds": median(
                        external, "first_update_seconds"
                    ),
                    "callback_candidate_first_update_seconds": median(
                        callback, "first_update_seconds", route="forced_candidate"
                    ),
                    "candidate_eager_steady_ratio": median(
                        callback, "steady_median_seconds", route="forced_candidate"
                    )
                    / median(callback, "steady_median_seconds", route="default_eager"),
                    "whole_jax_eager_steady_ratio": median(
                        external, "steady_median_seconds"
                    )
                    / median(callback, "steady_median_seconds", route="default_eager"),
                }
            )
    return {
        "numerical_tolerance": TOLERANCE,
        "process_counts": {"callback": 72, "unobserved": 72, "whole_jax": 36},
        "maximum_errors": max_errors,
        "independent_consumed_hash_checks": consumed_hash_checks(callback),
        "cross_route_hash_differences_from_rounding": different_hashes,
        "peak_worker_rss_bytes": max(
            r["peak_worker_rss_bytes"] for r in (*callback, *unobserved, *external)
        ),
        "compiled_default_adoption": False,
        "whole_step_fusion_claim": False,
        "issue_936_closure": False,
        "issue_918_closure": False,
        "time_to_quality_target": None,
        "invalidation_timing": None,
        "native_whole_circuit_adjoint_control": "not measured",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    actual = summary()
    path = HERE / "summary.json"
    if args.write_summary:
        path.write_text(json.dumps(actual, indent=2) + "\n")
    else:
        assert actual == json.loads(path.read_text())
    print(json.dumps(actual, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
