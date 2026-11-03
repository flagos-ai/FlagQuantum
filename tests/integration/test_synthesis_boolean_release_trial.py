"""Cross-frontend release evidence with default Boolean selection."""

import pytest

from benchmarks.synthesis.performance_workloads import Prepared, Workload
from examples.synthesis_arithmetic_implementations import main
from flagquantum import synth
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration


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
def test_trial_improves_complete_domain_cases(case, frontend, monkeypatch) -> None:
    workload = Prepared(Workload(case, frontend))
    with monkeypatch.context() as legacy_policy:
        from flagquantum.services import _synthesis_registers

        select = _synthesis_registers.select_register_program

        def select_legacy(*args, **kwargs):
            kwargs["include_boolean"] = False
            return select(*args, **kwargs)

        legacy_policy.setattr(
            _synthesis_registers, "select_register_program", select_legacy
        )
        legacy = workload.compile()
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
def test_trial_verifies_exact_basis_and_routing(frontend, target) -> None:
    workload = Prepared(Workload("energy-2x2", frontend, target=target))
    trial = workload.compile()
    assert trial.report["candidate"] == "boolean"
    verification = workload.verify(trial)
    assert verification["e1"]["status"] == "verified"
    assert (
        trial.report["resources"].n_operations
        > trial.report["source_resources"].n_operations
    )


def test_trial_respects_whole_program_limits_and_runtime_memory() -> None:
    limits = dict(max_qubits=9, max_workspace=1, max_operations=24, max_depth=20)
    program = synth.synthesize(main, memory_limit_bytes=8192, **limits)
    assert program.execution.state_bytes == 8192
    with pytest.raises(PlanningError, match="memory"):
        synth.synthesize(main, memory_limit_bytes=8191, **limits)
