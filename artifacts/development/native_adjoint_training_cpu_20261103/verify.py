"""Check source identity, measured quality, and native execution evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="?", default=HERE / "results.json")
    packet = json.loads(parser.parse_args().receipt.read_text())
    assert (
        packet["producer_sha256"]
        == hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()
    )
    rows = {(row["task"], row["route"], row["seed"]): row for row in packet["results"]}
    seeds = (11, 29, 47)
    routes = ("fq_train", "native_adjoint")
    targets = {"classifier": (0.8, 1), "heisenberg": (0.01, -1)}
    assert len(packet["results"]) == len(rows) == 12
    max_loss_error = max_score_error = max_parameter_error = 0.0
    for task, (threshold, direction) in targets.items():
        for seed in seeds:
            public, native = (rows[task, route, seed] for route in routes)
            for row in (public, native):
                assert row["status"] == "passed"
                assert row["dense_value_error"] <= 1e-9
                assert row["dense_gradient_error"] <= 2e-6
                assert [step["step"] for step in row["trace"]] == list(range(1, 61))
                assert (row["initial_score"] - threshold) * direction < 0
                first = next(
                    step
                    for step in row["trace"]
                    if (step["score"] - threshold) * direction >= 0
                )
                assert row["target_step"] == first["step"]
                assert row["target_seconds"] == first["elapsed_seconds"]
                assert row["target_seconds"] <= row["total_seconds"]
            assert public["target_step"] == native["target_step"]
            assert all(step["ir_hash"] for step in public["trace"])
            assert native["native_counts"]["sweeps"] == (
                3840 if task == "classifier" else 180
            )
            assert native["native_counts"]["fused_parameter_adjoint_count"] > 0
            for left, right in zip(public["trace"], native["trace"], strict=True):
                max_loss_error = max(max_loss_error, abs(left["loss"] - right["loss"]))
                max_score_error = max(
                    max_score_error, abs(left["score"] - right["score"])
                )
            for left, right in zip(
                public["final_parameters"], native["final_parameters"], strict=True
            ):
                max_parameter_error = max(max_parameter_error, abs(left - right))
        for route in routes:
            samples = [rows[task, route, seed] for seed in seeds]
            target = statistics.median(row["target_seconds"] for row in samples)
            total = statistics.median(row["total_seconds"] for row in samples)
            print(
                f"{task} {route} cold time to target {target:.4f}s, 60 updates {total:.4f}s"
            )
    assert max(max_loss_error, max_score_error) <= 1e-10
    print(
        f"Maximum paired errors loss={max_loss_error:.3g}, score={max_score_error:.3g}, parameters={max_parameter_error:.3g}"
    )


if __name__ == "__main__":
    main()
