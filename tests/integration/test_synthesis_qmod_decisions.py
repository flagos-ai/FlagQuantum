"""Bounded decision decomposition and independent original acceptance."""

import pytest
import torch

from flagquantum import synth
from flagquantum.ecosystem._qmod import decisions
from tools.validate_qmod_decision_predicates import grover_power_matrix, validate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("case", ["grover_graph_coloring", "whitebox_fuzzing"])
def test_original_full_states(case: str) -> None:
    assert validate(case).workspace_nonzero_probability == 0


@pytest.mark.parametrize("rounds", [0, 1, 2, 10])
def test_reference_power_preserves_complete_operator(rounds: int) -> None:
    signs = torch.ones(16, dtype=torch.complex128)
    signs[[2, 5, 9]] = -1
    step = (
        torch.full((16, 16), 0.125, dtype=torch.complex128)
        - torch.eye(16, dtype=torch.complex128)
    ) * signs[None, :]
    torch.testing.assert_close(
        grover_power_matrix(4, (2, 5, 9), rounds),
        torch.linalg.matrix_power(step, rounds),
        atol=1e-12,
        rtol=0,
    )


SOURCE = "qstruct Data {x:qnum<5>;y:qnum<4>;} qperm pred(const d:Data,m:qbit) {m ^= (d.x+d.y)<3;} qfunc main(output d:Data) {allocate(d);hadamard_transform(d);phase_oracle(pred,d);}"


@pytest.mark.parametrize(
    "name,value",
    [
        ("MAX_DECISION_STEPS", 1),
        ("MAX_DECISION_NODES", 1),
        ("MAX_DECISION_CUBES", 1),
        ("MAX_DECISION_BITS", 8),
    ],
)
def test_decision_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch, name: str, value: int
) -> None:
    monkeypatch.setattr(decisions, name, value)
    with pytest.raises(ValueError, match="decision"):
        synth.synthesize(SOURCE, source_format="qmod")


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_qubits", 15),
        ("max_workspace", 0),
        ("max_operations", 100),
        ("max_depth", 100),
        ("memory_limit_bytes", 1024),
    ],
)
def test_actual_budgets(key: str, value: int) -> None:
    from tools.validate_qmod_decision_predicates import FIXTURES

    with pytest.raises(
        ValueError,
        match="budget|qubit|workspace|operation|depth|memory|limit|candidate",
    ):
        synth.synthesize(FIXTURES / "grover_graph_coloring.qmod", **{key: value})
