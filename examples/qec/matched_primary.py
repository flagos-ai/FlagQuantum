"""Registered matched-model count reduction; no collection, file writer or CLI.

Only the future guarded primary replay may supply formal prediction shards.
Calling the private reducer on synthetic arrays provides software checks only;
it authenticates no inputs, devices or physical study.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import torch

from examples.qec import continued_primary as primary
from examples.qec import matched_parity as parity

SEED = 22000011
SHOTS = 4194304
CHUNK = 4096
ALPHA = 0.05 / 32
COMPARISONS = (
    tuple((name, ref) for name in parity.PATHS[3:] for ref in ("global", "fast"))
    + tuple(
        (f"cnn-{seed}-step-2000-{mode}", f"mlp-{seed}-step-2000-{mode}")
        for seed in (11, 22, 33)
        for mode in ("raw", "calibrated")
    )
    + (("fast", "global"), ("mask-only", "global"))
)


class CountReduction(TypedDict):
    shots: int
    comparison_alpha: float
    comparisons: list[primary.ComparisonReport]
    native_quality_gate: str
    matched_recipe_incremental_benefit: str
    competitive_with_fast: bool


def _reduce_predictions(
    chunks: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> CountReduction:
    """Reduce all 32 fixed comparisons; propagate any exact interval failure."""
    totals = {pair: [0] * 5 for pair in COMPARISONS}
    primary._accumulate_counts(chunks, totals, paths=parity.PATHS, shots=SHOTS)
    records, bounds = primary._comparison_reports(totals, alpha=ALPHA)

    def passes(reference: str, limit: float) -> bool:
        for seed in (11, 22, 33):
            candidate = f"cnn-{seed}-step-2000-calibrated"
            ref = (
                f"mlp-{seed}-step-2000-calibrated" if reference == "mlp" else reference
            )
            upper = bounds[candidate, ref]
            if upper is None or upper > limit:
                return False
        return True

    quality = passes("global", 0.9)
    return {
        "shots": SHOTS,
        "comparison_alpha": ALPHA,
        "comparisons": records,
        "native_quality_gate": (
            "supported_10_percent_reduction" if quality else "inconclusive"
        ),
        "matched_recipe_incremental_benefit": (
            "supported_10_percent_reduction"
            if quality and passes("mlp", 0.9)
            else "inconclusive"
        ),
        "competitive_with_fast": passes("fast", 1.05),
    }
