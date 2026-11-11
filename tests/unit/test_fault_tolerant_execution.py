"""Hand-solvable schedules and independent resource invariants for #1131."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.algorithms._fault_tolerant_execution import (
    _estimate_execution,
    _MachineAssumptions,
)
from flagquantum.algorithms.logical_resources import (
    SURFACE_CODE_MODEL,
    estimate_logical_resources,
)
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.unit


def machine(**changes: Any) -> _MachineAssumptions:
    # This test boundary deliberately accepts malformed external assumptions.
    base = _MachineAssumptions(
        name="hand-solvable",
        time_unit="declared_tick",
        operation_durations=(("h", 2), ("x", 3), ("t", 1), ("tdg", 1), ("cx", 4)),
        code_model=SURFACE_CODE_MODEL,
        code_distance=3,
        logical_data_capacity=2,
        factory_count=0,
        factory_physical_qubits=0,
        factory_startup=0,
        factory_period=5,
        inventory_capacity=4,
        initial_inventory=4,
        storage_physical_qubits_per_state=17,
        physical_qubit_budget=None,
    )
    return replace(base, **changes)


def test_unlimited_supply_preserves_dependencies_and_existing_reports() -> None:
    circuit = fq.Circuit(2).h(0).t(0).x(1).cx(0, 1).tdg(1)
    old_report = estimate_logical_resources(circuit, distance=3).to_dict()
    report = _estimate_execution(circuit, machine())
    times = {op.instruction_index: (op.start, op.end) for op in report.operations}
    assert times == {0: (0, 2), 1: (2, 3), 2: (0, 3), 3: (3, 7), 4: (7, 8)}
    assert report.completion_time == report.dependency_time_lower_bound == 8
    assert report.stalls == ()
    assert report.logical_resources.to_dict() == old_report
    assert report.logical_resources.estimate == estimate_resources(circuit)
    assert estimate_logical_resources(circuit, distance=3).to_dict() == old_report


def test_factory_bottleneck_and_ready_gate_progress() -> None:
    config = machine(
        factory_count=1,
        factory_physical_qubits=100,
        initial_inventory=0,
        inventory_capacity=1,
    )
    report = _estimate_execution(fq.Circuit(2).t(0).x(1).tdg(0), config)
    assert [
        (op.instruction_index, op.start, op.end, op.factory_wait)
        for op in report.operations
    ] == [
        (1, 0, 3, 0),
        (0, 5, 6, 5),
        (2, 10, 11, 4),
    ]
    assert report.completion_time == report.supply_time_lower_bound == 11
    assert report.dependency_time_lower_bound == 3
    assert sum(stall.end - stall.start for stall in report.stalls) == 9
    assert report.peak_inventory == 1
    assert report.physical_qubits == 2 * 17 + 100 + 17


def test_finite_inventory_discards_overflow_and_loses_parallelism() -> None:
    circuit = fq.Circuit(2).x(0).x(1).t(0).t(1)
    config = machine(
        operation_durations=(("x", 12), ("t", 1)),
        factory_count=2,
        factory_physical_qubits=100,
        inventory_capacity=1,
        initial_inventory=0,
    )
    report = _estimate_execution(circuit, config)
    assert [(op.start, op.end) for op in report.operations] == [
        (0, 12),
        (0, 12),
        (12, 13),
        (15, 16),
    ]
    assert report.completion_time == 16
    assert report.dependency_time_lower_bound == 13
    assert report.supply_time_lower_bound == 6  # separate lower bound, not a schedule
    assert report.peak_inventory == 1
    assert sum(event.discarded for event in report.inventory) == 4
    larger = _estimate_execution(circuit, replace(config, inventory_capacity=2))
    assert larger.completion_time == 13
    assert larger.peak_inventory == 2
    assert larger.physical_qubits == report.physical_qubits + 17


def test_factory_startup_and_same_tick_production_before_consumption() -> None:
    config = machine(
        factory_count=1,
        factory_physical_qubits=100,
        factory_startup=7,
        inventory_capacity=1,
        initial_inventory=0,
    )
    report = _estimate_execution(fq.Circuit(1).t(0), config)
    assert report.operations[0].start == 12
    assert [(e.kind, e.time, e.after) for e in report.inventory] == [
        ("initial", 0, 0),
        ("production", 12, 1),
        ("consume", 12, 0),
    ]


def test_full_storage_skips_arbitrarily_many_discarded_factory_outputs() -> None:
    long_duration = 10**12
    config = machine(
        operation_durations=(("h", long_duration), ("t", 1)),
        factory_count=1,
        factory_physical_qubits=100,
        factory_period=1,
        inventory_capacity=1,
        initial_inventory=0,
    )
    report = _estimate_execution(fq.Circuit(1).h(0).t(0), config)
    assert report.operations[1].start == long_duration
    assert report.completion_time == long_duration + 1
    assert sum(event.discarded for event in report.inventory) == long_duration - 1
    assert len(report.inventory) == 5


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"logical_data_capacity": 1}, "logical register"),
        ({"inventory_capacity": 0, "initial_inventory": 0}, "inventory slot"),
        ({"initial_inventory": 0}, "no factory"),
        ({"physical_qubit_budget": 101}, "footprints"),
    ],
)
def test_capacity_infeasibility_has_no_partial_schedule(
    changes: dict[str, int], reason: str
) -> None:
    report = _estimate_execution(fq.Circuit(2).t(0), machine(**changes))
    assert any(reason in item for item in report.infeasibility)
    assert report.completion_time is None
    assert report.peak_inventory is None
    if "no factory" in reason:
        assert report.supply_time_lower_bound is None
    assert not report.operations
    assert not report.inventory
    assert not report.stalls
    assert report.to_dict()["status"] == "capacity_infeasible"
    assert report.to_dict()["placement"] == "all_to_all_no_placement"


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"name": " "}, ValueError),
        ({"time_unit": 1}, ValueError),
        ({"factory_period": 0}, ValueError),
        ({"factory_startup": -1}, ValueError),
        ({"factory_count": True}, TypeError),
        ({"factory_count": 1.5}, TypeError),
        ({"inventory_capacity": 3}, ValueError),
        ({"storage_physical_qubits_per_state": 0}, ValueError),
        ({"factory_count": 1}, ValueError),
        ({"operation_durations": (("h", 0),)}, ValueError),
        ({"operation_durations": (("h", True),)}, TypeError),
        ({"operation_durations": (("h", 1), ("h", 2))}, ValueError),
        ({"operation_durations": [("h", 1)]}, TypeError),
        ({"operation_durations": (["h", 1],)}, TypeError),
        ({"code_distance": 4}, ValueError),
        ({"code_model": "unregistered_code"}, CapabilityError),
    ],
)
def test_invalid_assumptions_are_refused(
    changes: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        machine(**changes)


def test_missing_durations_and_unsupported_semantics_fail_closed() -> None:
    with pytest.raises(ValueError, match="durations.*h"):
        _estimate_execution(fq.Circuit(1).h(0), machine(operation_durations=()))
    for circuit in (fq.Circuit(1).rz(0, 0.1), fq.Circuit(3).ccx(0, 1, 2)):
        with pytest.raises(CapabilityError, match="decomposition"):
            _estimate_execution(circuit, machine())
    conditional = CircuitIR(
        1, (Instruction("x", (0,), metadata={"conditions": ((0, 1),)}),)
    )
    with pytest.raises(CapabilityError, match="static resource"):
        _estimate_execution(conditional, machine())
    measured = CircuitIR(1, (), measurements=(MeasurementNode("probs", (0,)),))
    with pytest.raises(CapabilityError, match="measurements"):
        _estimate_execution(measured, machine())
    matrix_gate = CircuitIR(1, (Instruction("x", (0,), matrix=[[1, 0], [0, 1]]),))
    with pytest.raises(CapabilityError, match="matrix"):
        _estimate_execution(matrix_gate, machine())


def test_seeded_schedules_obey_source_wire_order_and_inventory_conservation() -> None:
    rng = random.Random(1131)
    for _ in range(25):
        circuit = fq.Circuit(2)
        for _ in range(12):
            gate = rng.choice(("h", "t", "tdg", "cx"))
            if gate == "cx":
                circuit.cx(0, 1)
            else:
                getattr(circuit, gate)(rng.randrange(2))
        report = _estimate_execution(
            circuit,
            machine(
                factory_count=1,
                factory_physical_qubits=100,
                initial_inventory=0,
                inventory_capacity=rng.randint(1, 3),
            ),
        )
        by_index = {op.instruction_index: op for op in report.operations}
        assert len(by_index) == len(circuit.to_ir().instructions)
        last_end = [0, 0]
        for index, instruction in enumerate(circuit.to_ir()):
            timing = by_index[index]
            for wire in instruction.wires:
                assert timing.start >= last_end[wire]
                last_end[wire] = timing.end
        previous_inventory = 0
        for event in report.inventory:
            assert event.before == previous_inventory
            assert 0 <= event.after <= report.machine.inventory_capacity
            previous_inventory = event.after
        consumes = [e for e in report.inventory if e.kind == "consume"]
        assert len(consumes) == report.logical_resources.t_count
        assert all(e.before - e.after == 1 for e in consumes)
        assert report.completion_time is not None
        assert report.completion_time >= report.dependency_time_lower_bound
        assert report.supply_time_lower_bound is not None
        assert report.completion_time >= report.supply_time_lower_bound
        assert (
            _estimate_execution(circuit, report.machine).to_dict() == report.to_dict()
        )


def test_empty_program_and_unavailable_failure_budget() -> None:
    report = _estimate_execution(fq.Circuit(1), machine())
    assert report.completion_time == 0
    assert report.operations == ()
    assert report.to_dict()["failure_budget"] == {
        "status": "unavailable",
        "error_model": None,
        "provenance": None,
        "success_probability": None,
    }


@pytest.mark.integration
def test_example_json_replays_byte_for_byte(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    example = root / "examples/algorithms/fault_tolerant_execution.py"
    first = subprocess.run(
        [sys.executable, str(example)],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
        timeout=60,
    ).stdout
    payload = json.loads(first)
    assert payload["completion_time"] == 15
    saved = tmp_path / "estimate.json"
    saved.write_text(first, encoding="utf-8")
    second = subprocess.run(
        [sys.executable, str(example), "--replay", str(saved)],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
        timeout=60,
    ).stdout
    assert second == first
