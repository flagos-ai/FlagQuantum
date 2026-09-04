import pytest

from flagquantum.compilation.models import CircuitAnalysis
from flagquantum.runtime.planner.candidates import RuntimeCandidate
from flagquantum.runtime.planner.selection_result import finalize_runtime_selection

pytestmark = pytest.mark.unit


def _analysis() -> CircuitAnalysis:
    return CircuitAnalysis(
        n_wires=2,
        n_instructions=0,
        depth=0,
        gate_counts={},
        wire_usage=(0, 0),
        max_gate_width=0,
        two_qubit_gates=0,
        multi_qubit_gates=0,
    )


def _candidate(
    mode: str,
    score: int,
    *,
    blockers: tuple[str, ...] = (),
) -> RuntimeCandidate:
    return RuntimeCandidate(
        mode=mode,
        state_mode="statevector",
        execution_backend="native",
        distribution_semantics="single_device_fast_path",
        scalability_claim_allowed=False,
        estimated_memory_bytes=32,
        gradient_support="native_autograd",
        deployment_ready=True,
        score=score,
        blockers=blockers,
    )


def test_available_candidate_wins_over_higher_scored_blocked_candidate() -> None:
    blocked = _candidate("blocked", 100, blockers=("not_ready",))
    available = _candidate("available", 10)

    result = finalize_runtime_selection(
        analysis=_analysis(),
        candidates=(available, blocked),
        objective="training",
        prefer_distributed=False,
        world_size=1,
        local_world_size=1,
        node_count=1,
    )

    assert result.recommended_mode == "available"
    assert result.candidates == (blocked, available)
    assert result.user_tier == "single_device"
    assert result.usability_contract == "single_api_fast_path"


def test_all_blocked_candidates_fail_closed_to_highest_score() -> None:
    result = finalize_runtime_selection(
        analysis=_analysis(),
        candidates=(
            _candidate("lower", 1, blockers=("blocked",)),
            _candidate("higher", 2, blockers=("blocked",)),
        ),
        objective="inference",
        prefer_distributed=True,
        world_size=1,
        local_world_size=1,
        node_count=1,
    )

    assert result.recommended_mode == "higher"
    assert result.recommended_candidate.available is False
    assert result.user_tier == "production_distributed"
    assert result.usability_contract == "single_api_distributed_scale_out"


def test_empty_candidate_collection_is_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        finalize_runtime_selection(
            analysis=_analysis(),
            candidates=(),
            objective="training",
            prefer_distributed=False,
            world_size=1,
            local_world_size=1,
            node_count=1,
        )
