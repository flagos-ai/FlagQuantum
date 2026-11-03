"""Cross-frontend release evidence with the existing internal Boolean switch."""

from functools import partial

import pytest

from benchmarks.synthesis.performance_workloads import Prepared, Workload
from examples.synthesis_arithmetic_implementations import main
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.services import _synthesis_registers

pytestmark = pytest.mark.integration


@pytest.fixture
def enable_boolean(monkeypatch):
    """Exercise the real public entry and Runtime admission under trial policy."""
    selector = _synthesis_registers.select_register_program

    def enable() -> None:
        monkeypatch.setattr(
            _synthesis_registers,
            "select_register_program",
            partial(selector, include_boolean=True),
        )

    return enable


@pytest.mark.parametrize("frontend", ["qfunc", "qmod"])
@pytest.mark.parametrize(
    "case",
    [
        "energy-2x2",
        "energy-1x1",
        "multiply_add-1x1x1",
        "square-2",
        "square-3",
        "constant_product-1",
        "constant_product-3",
    ],
)
def test_trial_improves_complete_domain_cases(case, frontend, enable_boolean) -> None:
    workload = Prepared(Workload(case, frontend))
    legacy = workload.compile()
    enable_boolean()
    trial = workload.compile()
    assert trial.report["candidate"] == "boolean"
    assert (
        trial.report["resources"].n_operations < legacy.report["resources"].n_operations
    )
    verification = workload.verify(trial)
    assert verification["basis_domain_exhaustive"]
    assert verification["nonzero_targets"]
    assert verification["e1"]["status"] == "verified"
    if case == "energy-2x2":
        assert verification["basis_checks"] == 256
        assert trial.execution.state_bytes == 8192
        resource = trial.report["resources"]
        assert (
            resource.n_qubits,
            trial.report["workspace_qubits"],
            resource.n_operations,
            resource.depth,
        ) == (9, 1, 14, 9)


@pytest.mark.parametrize("frontend", ["qfunc", "qmod"])
@pytest.mark.parametrize("target", ["basis", "line"])
def test_trial_verifies_exact_basis_and_routing(
    frontend, target, enable_boolean
) -> None:
    workload = Prepared(Workload("energy-2x2", frontend, target=target))
    enable_boolean()
    trial = workload.compile()
    assert trial.report["candidate"] == "boolean"
    verification = workload.verify(trial)
    assert verification["e1"]["status"] == "verified"
    assert (
        trial.report["resources"].n_operations
        > trial.report["source_resources"].n_operations
    )


def test_trial_respects_whole_program_limits_and_runtime_memory(enable_boolean) -> None:
    enable_boolean()
    limits = dict(max_qubits=9, max_workspace=1, max_operations=24, max_depth=20)
    program = synth.synthesize(main, memory_limit_bytes=8192, **limits)
    assert program.execution.state_bytes == 8192
    with pytest.raises(PlanningError, match="memory"):
        synth.synthesize(main, memory_limit_bytes=8191, **limits)
