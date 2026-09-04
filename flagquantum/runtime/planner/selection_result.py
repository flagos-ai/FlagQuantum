"""Final ranking and result assembly for Runtime selection."""

from __future__ import annotations

from collections.abc import Sequence

from .candidate_plans import CircuitAnalysisView
from .candidates import RuntimeCandidate
from .models import RuntimeSelectionPlan


def finalize_runtime_selection(
    *,
    analysis: CircuitAnalysisView,
    candidates: Sequence[RuntimeCandidate],
    objective: str,
    prefer_distributed: bool,
    world_size: int,
    local_world_size: int,
    node_count: int,
) -> RuntimeSelectionPlan:
    """Rank candidates while preventing blocked options from winning."""

    if not candidates:
        raise ValueError("runtime selection requires at least one candidate")

    ranked_candidates = tuple(
        sorted(candidates, key=lambda item: item.score, reverse=True)
    )
    available = tuple(
        candidate for candidate in ranked_candidates if candidate.available
    )
    recommended = (available or ranked_candidates)[0]
    distributed_tier = prefer_distributed or world_size > 1
    return RuntimeSelectionPlan(
        analysis=analysis,
        recommended_mode=recommended.mode,
        recommended_candidate=recommended,
        candidates=ranked_candidates,
        objective=objective,
        user_tier=("production_distributed" if distributed_tier else "single_device"),
        usability_contract=(
            "single_api_distributed_scale_out"
            if distributed_tier
            else "single_api_fast_path"
        ),
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=node_count,
    )
