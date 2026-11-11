"""Internal, deterministic timing composition for issue #1131.

The compiler owns dependencies; logical_resources owns code conventions. This
module adds integer-time factory and storage constraints to those records. It
does not execute a circuit or extend the stable API. See the interface proposal
and examples/algorithms/fault_tolerant_execution.py before using this preview.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from ..compiler.lowering.schedule_legalization import (
    CircuitSchedule,
    schedule_circuit_dependencies,
)
from ..core.ir import ensure_circuit_ir
from ..errors import CapabilityError
from .logical_resources import (
    CLIFFORD_T_OPCODES,
    SURFACE_CODE_MODEL,
    T_FAMILY_OPCODES,
    LogicalResourceReport,
    estimate_logical_resources,
    surface_code_qubits_per_logical,
)

_PLACEMENT = "all_to_all_no_placement"
_LIMITATIONS = (
    "Model-based estimate, not QPU execution or measured performance.",
    "All-to-all connectivity; placement, routing and communication costs are absent.",
    "Static unitary Clifford+T only; measurement, classical feed-forward and "
    "observable evaluation are unsupported. Decompose logical gates explicitly.",
    "One stored magic state is consumed at the start of each T or T-dagger. "
    "Injection work is included only in the caller's operation duration.",
    "Homogeneous independent factories, one state per cadence per factory; "
    "overflow is discarded. No aging, correlated failures or shared factory lanes.",
    "Deterministic source-index priority among ready operations; a feasible "
    "schedule is not an optimality claim.",
    "Failure budget unavailable: no calibrated error model or provenance is "
    "implemented; no success probability is inferred from code distance.",
)


@dataclass(frozen=True)
class _MachineAssumptions:
    """Caller-supplied units, code convention and provisioned capacities.

    Times are integer ticks of time_unit. First output is at startup + period;
    production is processed before starts at the same tick. Factories start at
    zero. Storage is provisioned separately from data and factory footprints.
    """

    name: str
    time_unit: str
    operation_durations: tuple[tuple[str, int], ...]
    code_model: str
    code_distance: int
    logical_data_capacity: int
    factory_count: int
    factory_physical_qubits: int
    factory_startup: int
    factory_period: int
    inventory_capacity: int
    initial_inventory: int
    storage_physical_qubits_per_state: int
    physical_qubit_budget: int | None

    def __post_init__(self) -> None:
        for field in ("name", "time_unit"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if self.code_model != SURFACE_CODE_MODEL:
            raise CapabilityError(f"unsupported code convention: {self.code_model!r}")
        surface_code_qubits_per_logical(self.code_distance)
        for field in (
            "logical_data_capacity",
            "factory_count",
            "factory_physical_qubits",
            "factory_startup",
            "inventory_capacity",
            "initial_inventory",
            "storage_physical_qubits_per_state",
        ):
            _require_integer(field, getattr(self, field), minimum=0)
        _require_integer("factory_period", self.factory_period, minimum=1)
        if self.physical_qubit_budget is not None:
            _require_integer(
                "physical_qubit_budget", self.physical_qubit_budget, minimum=0
            )
        if self.initial_inventory > self.inventory_capacity:
            raise ValueError("initial_inventory exceeds inventory_capacity")
        if self.factory_count and not self.factory_physical_qubits:
            raise ValueError("a factory requires a positive physical-qubit footprint")
        if self.inventory_capacity and not self.storage_physical_qubits_per_state:
            raise ValueError("storage requires a positive physical-qubit footprint")
        if not isinstance(self.operation_durations, tuple):
            raise TypeError("operation_durations must be an immutable tuple of pairs")
        names: set[str] = set()
        for pair in self.operation_durations:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "each duration must be an immutable (opcode, ticks) pair"
                )
            name, duration = pair
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("duration opcodes must be non-empty and unique")
            names.add(name)
            _require_integer(f"duration for {name}", duration, minimum=1)


def _require_integer(name: str, value: int, *, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


@dataclass(frozen=True)
class _OperationTiming:
    instruction_index: int
    start: int
    end: int
    factory_wait: int


@dataclass(frozen=True)
class _InventoryTransition:
    time: int
    kind: Literal["initial", "production", "consume"]
    before: int
    after: int
    discarded: int = 0
    instruction_index: int | None = None


@dataclass(frozen=True)
class _FactoryStall:
    start: int
    end: int
    instruction_indices: tuple[int, ...]


@dataclass(frozen=True)
class _ExecutionEstimate:
    machine: _MachineAssumptions
    logical_resources: LogicalResourceReport
    compiler_schedule: CircuitSchedule
    infeasibility: tuple[str, ...]
    dependency_time_lower_bound: int
    supply_time_lower_bound: int | None
    physical_qubits: int
    operations: tuple[_OperationTiming, ...]
    inventory: tuple[_InventoryTransition, ...]
    stalls: tuple[_FactoryStall, ...]
    peak_inventory: int | None
    completion_time: int | None

    def to_dict(self) -> dict[str, object]:
        """Serialize assumptions, authoritative facts and replayable timing."""
        return {
            "kind": "flagquantum.internal_fault_tolerant_execution_estimate",
            "status": "capacity_infeasible" if self.infeasibility else "feasible",
            "placement": _PLACEMENT,
            "limitations": list(_LIMITATIONS),
            "assumptions": asdict(self.machine),
            "program": self.compiler_schedule.program.to_dict(),
            "compiler_schedule_identity": self.compiler_schedule.schedule_identity,
            "logical_resources": self.logical_resources.to_dict(),
            "infeasibility": list(self.infeasibility),
            "lower_bounds": {
                "dependency_time": self.dependency_time_lower_bound,
                "supply_time": self.supply_time_lower_bound,
            },
            "physical_qubits": self.physical_qubits,
            "operations": [
                {
                    **asdict(timing),
                    "predecessors": self.compiler_schedule.instructions[
                        timing.instruction_index
                    ].predecessors,
                }
                for timing in self.operations
            ],
            "inventory": [asdict(event) for event in self.inventory],
            "stalls": [asdict(stall) for stall in self.stalls],
            "factory_stall_time": sum(stall.end - stall.start for stall in self.stalls),
            "peak_inventory": self.peak_inventory,
            "completion_time": self.completion_time,
            "failure_budget": {
                "status": "unavailable",
                "error_model": None,
                "provenance": None,
                "success_probability": None,
            },
        }


def _estimate_execution(
    program: object, machine: _MachineAssumptions
) -> _ExecutionEstimate:
    """Compose existing compiler facts with one declared resource model.

    Invalid assumptions raise TypeError/ValueError. Unsupported program semantics
    raise CapabilityError; insufficient machine capacity returns an infeasible
    report with no partial schedule. No circuit transformations are performed.
    """
    ir = ensure_circuit_ir(program)
    if ir.measurements or ir.observables:
        raise CapabilityError("terminal measurements and observables are unsupported")
    if any(instruction.matrix is not None for instruction in ir):
        raise CapabilityError(
            "custom matrix semantics require an explicit decomposition"
        )
    unsupported = {instruction.name for instruction in ir} - CLIFFORD_T_OPCODES
    if unsupported:
        raise CapabilityError(
            f"explicit Clifford+T decomposition required: {sorted(unsupported)}"
        )
    logical = estimate_logical_resources(ir, distance=machine.code_distance)
    schedule = schedule_circuit_dependencies(ir, target_snapshot_id=machine.name)
    durations = dict(machine.operation_durations)
    missing = set(logical.estimate.operation_counts) - durations.keys()
    if missing:
        raise ValueError(
            f"operation durations are required for: {', '.join(sorted(missing))}"
        )

    dependency_ends: list[int] = []
    for item in schedule.instructions:
        start = max((dependency_ends[p] for p in item.predecessors), default=0)
        dependency_ends.append(
            start + durations[ir.instructions[item.instruction_index].name]
        )
    dependency_bound = max(dependency_ends, default=0)
    needed = max(0, logical.t_count - machine.initial_inventory)
    supply_bound: int | None = None if needed else 0
    if needed and machine.factory_count:
        batches = (needed + machine.factory_count - 1) // machine.factory_count
        shortest_t = min(
            durations[name]
            for name in T_FAMILY_OPCODES
            if name in logical.estimate.operation_counts
        )
        supply_bound = (
            machine.factory_startup + batches * machine.factory_period + shortest_t
        )

    physical_qubits = (
        machine.logical_data_capacity
        * surface_code_qubits_per_logical(machine.code_distance)
        + machine.factory_count * machine.factory_physical_qubits
        + machine.inventory_capacity * machine.storage_physical_qubits_per_state
    )
    reasons: list[str] = []
    if ir.n_wires > machine.logical_data_capacity:
        reasons.append("logical register exceeds logical_data_capacity")
    if logical.t_count and not machine.inventory_capacity:
        reasons.append("T operations require at least one inventory slot")
    if needed and not machine.factory_count:
        reasons.append(
            "initial inventory is insufficient and no factory is provisioned"
        )
    if (
        machine.physical_qubit_budget is not None
        and physical_qubits > machine.physical_qubit_budget
    ):
        reasons.append(
            "data, factory and storage footprints exceed physical_qubit_budget"
        )
    if reasons:
        return _ExecutionEstimate(
            machine,
            logical,
            schedule,
            tuple(reasons),
            dependency_bound,
            supply_bound,
            physical_qubits,
            (),
            (),
            (),
            None,
            None,
        )

    operations, inventory, stalls = _time_operations(schedule, machine, durations)
    return _ExecutionEstimate(
        machine,
        logical,
        schedule,
        (),
        dependency_bound,
        supply_bound,
        physical_qubits,
        tuple(operations),
        tuple(inventory),
        tuple(stalls),
        max(event.after for event in inventory),
        max((operation.end for operation in operations), default=0),
    )


def _time_operations(
    schedule: CircuitSchedule, machine: _MachineAssumptions, durations: dict[str, int]
) -> tuple[list[_OperationTiming], list[_InventoryTransition], list[_FactoryStall]]:
    # Timings annotate compiler instruction indices; dependencies are never rebuilt.
    remaining = set(range(len(schedule.instructions)))
    ends: dict[int, int] = {}
    waiting_since: dict[int, int] = {}
    timings: list[_OperationTiming] = []
    transitions = [_InventoryTransition(0, "initial", 0, machine.initial_inventory)]
    stalls: list[_FactoryStall] = []
    inventory = machine.initial_inventory
    next_output = machine.factory_startup + machine.factory_period
    now = 0
    while True:
        if machine.factory_count and next_output <= now:
            batches = (now - next_output) // machine.factory_period + 1
            produced = batches * machine.factory_count
            accepted = min(machine.inventory_capacity - inventory, produced)
            transitions.append(
                _InventoryTransition(
                    now,
                    "production",
                    inventory,
                    inventory + accepted,
                    produced - accepted,
                )
            )
            inventory += accepted
            next_output += batches * machine.factory_period

        blocked: list[int] = []
        for index in sorted(remaining):
            item = schedule.instructions[index]
            if any(p not in ends or ends[p] > now for p in item.predecessors):
                continue
            name = schedule.program.instructions[index].name
            if name in T_FAMILY_OPCODES:
                if not inventory:
                    waiting_since.setdefault(index, now)
                    blocked.append(index)
                    continue
                transitions.append(
                    _InventoryTransition(
                        now,
                        "consume",
                        inventory,
                        inventory - 1,
                        instruction_index=index,
                    )
                )
                inventory -= 1
            end = now + durations[name]
            timings.append(
                _OperationTiming(index, now, end, now - waiting_since.pop(index, now))
            )
            ends[index] = end
            remaining.remove(index)

        future = [end for end in ends.values() if end > now]
        # Include production at the completion tick, then stop the factories.
        if not remaining and not future:
            break
        if machine.factory_count and inventory < machine.inventory_capacity:
            future.append(next_output)
        later = min(future)
        if blocked:
            stalls.append(_FactoryStall(now, later, tuple(blocked)))
        now = later
    return timings, transitions, stalls
