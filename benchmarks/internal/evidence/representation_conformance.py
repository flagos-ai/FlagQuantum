"""Scoped SVD observations and source identity for conformance measurements.

The observer reads the singular values returned by the actual factorization;
it neither changes factors nor adds instrumentation to the production runtime.
It is intended for these single-threaded, bounded benchmark processes only.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

import torch

from flagquantum.simulation.mps import factorization

NEAR_DEGENERATE_GAP = 1e-9


class SplitObservation(TypedDict):
    call: int
    spectrum: int
    kept_rank: int
    original_rank: int
    discarded_weight: float
    normalized_weight_gap: float
    near_degenerate: bool


class SplitBoundary(TypedDict):
    source: str
    gap_definition: str
    near_degenerate_threshold: float
    near_degenerate: bool
    min_gap: float | None
    records: list[SplitObservation]
    interpretation: str


class SourceRecord(TypedDict):
    source_commit: str
    base_commit: str
    measuring_source_committed: bool
    command: str


@contextmanager
def observe_svd_splits(max_bond: int) -> Iterator[list[SplitObservation]]:
    """Observe each actual cap decision, including batched factorization calls.

    The sweep uses cutoff zero and batch one, so the retained rank is the cap
    clipped to the spectrum length, also for each independent bucket member.
    Gaps and discarded weights are normalized by that matrix's squared norm.
    Null-space ties are retained in the record; they are not by themselves
    evidence of a physically significant loss or of non-differentiability.
    """
    original = factorization._cuda_svd
    records: list[SplitObservation] = []
    calls = 0

    def observe(
        matrix: torch.Tensor, *, driver: str | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal calls
        if matrix.shape[-3] != 1:
            raise ValueError("the SVD observer supports circuit batch one only")
        result = original(matrix, driver=driver)
        singular = result[1].detach().reshape(-1, result[1].shape[-1])
        for index, values in enumerate(singular):
            rank = min(max_bond, values.numel())
            if rank == values.numel():
                continue
            weights = values.double().square()
            total = float(weights.sum())
            if total == 0:
                raise ValueError("cannot normalize the spectrum of a zero matrix")
            gap = float(weights[rank - 1] - weights[rank]) / total
            records.append(
                {
                    "call": calls,
                    "spectrum": index,
                    "kept_rank": rank,
                    "original_rank": values.numel(),
                    "discarded_weight": float(weights[rank:].sum()) / total,
                    "normalized_weight_gap": gap,
                    "near_degenerate": gap < NEAR_DEGENERATE_GAP,
                }
            )
        calls += 1
        return result

    with patch.object(factorization, "_cuda_svd", observe):
        yield records


def split_boundary(records: Sequence[SplitObservation]) -> SplitBoundary:
    """Summarize observed boundaries without treating a final-state gap as one."""
    return {
        "source": "actual_training_forward_svd",
        "gap_definition": "(s[rank-1]^2 - s[rank]^2) / sum(s^2)",
        "near_degenerate_threshold": NEAR_DEGENERATE_GAP,
        "near_degenerate": any(record["near_degenerate"] for record in records),
        "min_gap": min(
            (record["normalized_weight_gap"] for record in records), default=None
        ),
        "records": list(records),
        "interpretation": (
            "a sampled boundary warning, not proof that a derivative exists or "
            "does not exist; null-space ties can discard negligible weight"
        ),
    }


def source_record(root: Path, argv: Sequence[str]) -> SourceRecord:
    """Name the committed measuring code and a portable worker command."""

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, timeout=15
        ).strip()

    # Runtime and producer changes both affect the measurement. Evidence outputs
    # and documentation do not make the numerical source dirty.
    changed = git(
        "status",
        "--porcelain",
        "--",
        "flagquantum",
        "benchmarks/*.py",
        "benchmarks/internal",
    )
    command = [argument.replace(str(root.resolve()) + "/", "") for argument in argv]
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "base_commit": git("rev-parse", "origin/main"),
        "measuring_source_committed": not changed,
        "command": shlex.join(["python", *command]),
    }
