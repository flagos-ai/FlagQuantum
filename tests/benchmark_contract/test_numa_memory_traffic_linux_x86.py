import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark_contract

ROOT = Path(__file__).parents[2]
RESULT = (
    ROOT
    / "benchmarks"
    / "results"
    / "comparison"
    / "numa_memory_traffic_linux_x86_20261004.json"
)
REPORT = RESULT.with_name("NUMA_MEMORY_TRAFFIC_LINUX_X86_20261004.md")


def _payload() -> dict[str, object]:
    return json.loads(RESULT.read_text())


def _cases() -> dict[tuple[str, str, str], dict[str, object]]:
    payload = _payload()
    return {
        (case["workload"], case["engine"], case["policy"]): case
        for case in payload["cases"]
    }


def test_numa_artifact_records_a_passing_auditable_run() -> None:
    payload = _payload()
    assert payload["schema"] == "flagquantum.numa_memory_traffic.v1"
    assert payload["runner"] == "numa_memory_traffic"
    assert payload["hostname"] == "redacted"
    assert payload["passed"] is True
    assert payload["assessment"] == {
        "automatic_bind_enabled": False,
        "correctness_passed": True,
        "forced_interleave_enabled": False,
        "material_speedup_threshold": 1.05,
        "maximum_relative_median_absolute_deviation": 0.2,
        "native_path_passed": True,
        "non_default_policy_universally_materially_faster": {
            "bind": False,
            "interleave": False,
        },
        "passed": True,
        "reason": (
            "keep Linux first-touch because neither forced bind nor forced "
            "interleave improves every measured workload by at least 5%"
        ),
        "recommended_policy": "default",
        "stability_passed": True,
    }
    assert len(payload["cases"]) == 8
    assert all(case["correctness"]["passed"] for case in payload["cases"])
    assert all(case["stable"] for case in payload["cases"])


@pytest.mark.parametrize("workload", ("hardware_efficient_vqe", "qaoa_path_maxcut"))
def test_forced_interleave_is_slower_and_moves_more_dram(workload: str) -> None:
    cases = _cases()
    default = cases[(workload, "flagquantum_adjoint", "default")]
    interleave = cases[(workload, "flagquantum_adjoint", "interleave")]
    assert (
        interleave["timing"]["total"]["median_seconds"]
        > default["timing"]["total"]["median_seconds"]
    )
    assert (
        interleave["traffic"]["bytes_per_call"] > default["traffic"]["bytes_per_call"]
    )


@pytest.mark.parametrize("workload", ("hardware_efficient_vqe", "qaoa_path_maxcut"))
def test_flagquantum_default_leads_lightning_on_same_semantics(workload: str) -> None:
    cases = _cases()
    native = cases[(workload, "flagquantum_adjoint", "default")]
    lightning = cases[(workload, "pennylane_lightning_adjoint", "default")]
    assert (
        lightning["timing"]["total"]["median_seconds"]
        / native["timing"]["total"]["median_seconds"]
        > 50.0
    )
    assert (
        lightning["traffic"]["bytes_per_call"] / native["traffic"]["bytes_per_call"]
        > 50.0
    )


def test_report_explains_meaning_example_reproduction_and_boundaries() -> None:
    report = REPORT.read_text()
    for required in (
        "FlagQuantum is **78.264x faster**",
        "PennyLane Lightning adjoint",
        "What this measures and why it matters",
        'differentiation="adjoint"',
        "flagquantum-benchmark run numa_memory_traffic",
        "Boundaries and stopping condition",
        "default first-touch policy",
    ):
        assert required in report
