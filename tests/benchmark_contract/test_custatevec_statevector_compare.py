import pytest

from benchmarks.custatevec_statevector_compare import workload

pytestmark = pytest.mark.benchmark_contract


def test_matched_workload_has_deterministic_gate_count_and_topology():
    gates = workload(5, 3)

    assert len(gates) == 3 * (2 * 5 + 4)
    assert [gate.name for gate in gates[:3]] == ["ry", "rz", "ry"]
    assert [gate.wires for gate in gates[-4:]] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert all(gate.angle is None for gate in gates if gate.name == "cx")


def test_matched_workload_rejects_no_dimensions_only_at_benchmark_boundary():
    assert workload(0, 0) == ()
