"""Exact arithmetic with retained or dependency-aware cleanup/recomputation.

Only verified unsigned expressions enter here. Both schedules compute into an
arbitrary XOR target and restore every supplied zero workspace qubit. The same
X/CX/CCX kernels serve forward and inverse computation without hidden phase.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ...core.ir import CircuitIR, Instruction
from .._arithmetic_kernels import comparison_xor, multiply_accumulate, ripple_add
from .._hybrid.model import INDEX, HybridProgram, Operation, ValueId
from .bindings import OutputMode, verify_bindings
from .lifetimes import LifetimeAnalysis, analyze_lifetimes
from .profile import _fail, _positive_integer
from .ranges import unsigned_width

_WireValue = tuple[int, ...] | int
Schedule = Literal["retain", "recompute"]


@dataclass(frozen=True)
class ScheduleEvent:
    action: Literal["compute", "uncompute"]
    value: ValueId
    allocation: tuple[int, ...]
    scratch: tuple[int, ...]
    live_workspace: int
    operation_count: int


@dataclass(frozen=True)
class ArithmeticCandidate:
    circuit: CircuitIR
    strategy: Schedule
    lifetimes: LifetimeAnalysis
    events: tuple[ScheduleEvent, ...]
    peak_workspace: int
    recomputations: int


@dataclass(frozen=True)
class _Allocation:
    owned: tuple[int, ...]
    result: tuple[int, ...]


def _copy(value: _WireValue, target: tuple[int, ...]) -> tuple[Instruction, ...]:
    """XOR a basis encoding or classical constant into a zero-extended target."""
    if isinstance(value, int):
        return tuple(
            Instruction("x", (wire,))
            for bit, wire in enumerate(reversed(target))
            if value & (1 << bit)
        )
    if len(value) > len(target):
        raise ValueError("internal arithmetic copy would truncate a quantum value")
    return tuple(
        Instruction("cx", (source, dest))
        for source, dest in zip(value, target[-len(value) :], strict=True)
    )


def schedule_arithmetic(
    functions: Sequence[HybridProgram],
    inputs: Mapping[ValueId, Sequence[int]],
    output: Sequence[int],
    *,
    entry: str,
    n_qubits: int,
    workspace: Sequence[int] = (),
    workspace_state: Literal[
        "fresh_zero", "caller_asserted_zero", "unknown"
    ] = "unknown",
    max_operations: int = 100_000,
    scheduling: Schedule = "retain",
    output_mode: OutputMode = "value",
) -> ArithmeticCandidate:
    """Emit one bounded candidate and its actual zero-pool allocation history.

    Retain is the compatibility default. Recompute eagerly erases non-input
    operands between forward uses, restores missing dependencies before cleanup,
    and retains final-result dependencies through the XOR copy. It is not an
    optimal pebbling search. External zero assertions remain caller obligations.
    """
    _positive_integer(n_qubits, "n_qubits")
    _positive_integer(max_operations, "max_operations")
    if n_qubits > 256 or max_operations > 100_000:
        _fail("lowering.budget", "safety caps are 256 qubits and 100000 operations")
    if scheduling not in {"retain", "recompute"}:
        _fail("lowering.schedule", "choose retain or recompute scheduling")
    verify_bindings(
        functions,
        inputs,
        output,
        entry=entry,
        n_wires=n_qubits,
        workspace=workspace,
        workspace_state=workspace_state,
        output_mode=output_mode,
    )
    analysis = analyze_lifetimes(functions, entry=entry)
    block = analysis.program.body.blocks[0]
    definitions = {op.results[0].id: op for op in block.operations[:-1]}
    available = tuple(workspace)
    occupied: set[int] = set()
    peak = 0
    instructions: list[Instruction] = []
    events: list[ScheduleEvent] = []
    active: dict[ValueId, _Allocation] = {}
    bound_inputs = {key: tuple(value) for key, value in inputs.items()}
    counts: dict[ValueId, int] = {}
    ensuring: set[ValueId] = set()
    returned = block.operations[-1].operands[0].id

    def allocate(count: int, op: Operation) -> tuple[int, ...]:
        nonlocal peak
        free = tuple(q for q in available if q not in occupied)
        if count > len(free):
            _fail(
                "lowering.workspace",
                f"{op.name} needs {count} more zero qubits; {len(free)} remain ({scheduling})",
                op,
            )
        result = free[:count]
        occupied.update(result)
        peak = max(peak, len(occupied))
        return result

    def emit(gates: Iterable[Instruction], op: Operation) -> int:
        start = len(instructions)
        for gate in gates:
            if len(instructions) >= max_operations:
                _fail(
                    "lowering.operations",
                    "compute/uncompute exceeds operation budget",
                    op,
                )
            instructions.append(gate)
        return len(instructions) - start

    def bound(key: ValueId) -> int:
        return analysis.values[key].upper_bound

    def encoding(key: ValueId) -> _WireValue:
        if key in bound_inputs:
            return bound_inputs[key]
        if analysis.values[key].value.type == INDEX:
            return bound(key)
        return active[key].result

    def operands(op: Operation) -> tuple[_WireValue, _WireValue]:
        left, right = (encoding(value.id) for value in op.operands)
        if op.name == "qint.add" and isinstance(left, int):
            left, right = right, left
        return left, right

    def reserve(key: ValueId, op: Operation) -> _Allocation:
        if op.name in {"qint.equal", "qint.less", "qint.multiply"}:
            owned = allocate(unsigned_width(bound(key)), op)
            return _Allocation(owned, owned)
        left, _ = operands(op)
        assert not isinstance(left, int)
        width = max(unsigned_width(bound(v.id)) for v in op.operands)
        owned = allocate(width - len(left) + width + 2, op)
        b = owned[-width - 2 : -2]
        result = (owned[-1], *b)[-unsigned_width(bound(key)) :]
        return _Allocation(owned, result)

    def kernel(
        key: ValueId, allocation: _Allocation, op: Operation, scratch: tuple[int, ...]
    ) -> Iterable[Instruction]:
        left, right = operands(op)
        if op.name in {"qint.equal", "qint.less"}:
            return comparison_xor(
                left,
                right,
                allocation.result[0],
                scratch[:-1],
                scratch[-1],
                operation="eq" if op.name == "qint.equal" else "lt",
            )
        if op.name == "qint.multiply":
            return multiply_accumulate(left, right, allocation.result, scratch)
        assert not isinstance(left, int)
        width = max(unsigned_width(bound(v.id)) for v in op.operands)
        padding = allocation.owned[: width - len(left)]
        b = allocation.owned[-width - 2 : -2]
        carry, carry_out = allocation.owned[-2:]
        return (*_copy(right, b), *ripple_add((*padding, *left), b, carry, carry_out))

    def apply(key: ValueId, *, inverse: bool) -> None:
        op = definitions[key]
        allocation = active[key]
        scratch_count = 0
        if op.name == "qint.multiply":
            scratch_count = len(allocation.result)
        elif op.name in {"qint.equal", "qint.less"}:
            scratch_count = max(unsigned_width(bound(v.id)) for v in op.operands) + 2
        scratch = allocate(scratch_count, op) if scratch_count else ()
        # Scratch is always zero and local to this invocation. A restored operand
        # may now occupy different physical qubits; regenerate the kernel using
        # its current binding instead of replaying stale controls.
        gates = kernel(key, allocation, op, scratch)
        if inverse:
            buffered: list[Instruction] = []
            for gate in gates:
                if len(instructions) + len(buffered) >= max_operations:
                    _fail("lowering.operations", "inverse exceeds operation budget", op)
                buffered.append(gate)
            gates = reversed(buffered)
        count = emit(gates, op)
        if len(events) >= 100_000:
            _fail(
                "lowering.schedule.budget",
                "schedule exceeds 100000 compute/cleanup events",
                op,
            )
        events.append(
            ScheduleEvent(
                "uncompute" if inverse else "compute",
                key,
                allocation.owned,
                scratch,
                len(occupied),
                count,
            )
        )
        occupied.difference_update(scratch)

    def ensure(key: ValueId) -> None:
        if (
            key in bound_inputs
            or key in active
            or analysis.values[key].value.type == INDEX
        ):
            return
        if key in ensuring or len(ensuring) >= 64:
            _fail(
                "lowering.schedule.depth",
                "rematerialization dependency depth exceeds 64",
            )
        ensuring.add(key)
        for dependency in analysis.values[key].dependencies:
            ensure(dependency)
        active[key] = reserve(key, definitions[key])
        apply(key, inverse=False)
        counts[key] = counts.get(key, 0) + 1
        ensuring.remove(key)

    def erase(key: ValueId) -> None:
        if key not in active:
            return
        before = set(active)
        for dependency in analysis.values[key].dependencies:
            ensure(dependency)
        restored = tuple(value for value in active if value not in before)
        apply(key, inverse=True)
        occupied.difference_update(active.pop(key).owned)
        # Only newly rematerialized dependencies belong to this cleanup call.
        # Other live values must survive, even if they share these dependencies.
        for value in reversed(restored):
            erase(value)

    for position, op in enumerate(block.operations[:-1]):
        key = op.results[0].id
        if op.name == "arith.constant":
            continue
        ensure(key)
        if scheduling == "recompute" and key != returned:
            for dependency in analysis.values[key].dependencies:
                erase(dependency)
            uses = analysis.values[key].uses
            if not uses or uses[-1] <= position:
                erase(key)
    ensure(returned)
    if output_mode == "phase":
        predicate = encoding(returned)
        assert not isinstance(predicate, int) and len(predicate) == 1
        prefix = tuple(instructions)
        forward_events = tuple(events)
        if any(
            gate.name not in {"x", "cx", "ccx"}
            or gate.matrix is not None
            or gate.params
            or gate.metadata
            for gate in prefix
        ):
            _fail(
                "phase.adjoint",
                "phase prefix requires admitted exact self-adjoint gates",
            )
        if 2 * len(forward_events) > 100_000:
            _fail(
                "lowering.schedule.budget",
                "phase inverse exceeds schedule event budget",
            )
        emit((Instruction("z", predicate),), block.operations[-1])
        # Invert the actual selected prefix, including its earlier cleanups and
        # physical rebindings. Preparation is composed later, outside this U.
        emit(reversed(prefix), block.operations[-1])
        for event in reversed(forward_events):
            action: Literal["compute", "uncompute"] = (
                "uncompute" if event.action == "compute" else "compute"
            )
            events.append(
                ScheduleEvent(
                    action,
                    event.value,
                    event.allocation,
                    event.scratch,
                    event.live_workspace,
                    event.operation_count,
                )
            )
            if action == "compute":
                counts[event.value] = counts.get(event.value, 0) + 1
    else:
        emit(_copy(encoding(returned), tuple(output)), block.operations[-1])
        # Restoration order is not dependency order. Clean descendants first so a
        # still-live ancestor is not erased only to be rebuilt for the next inverse.
        for key in reversed(definitions):
            erase(key)
        assert not active and not occupied
    circuit = CircuitIR(n_wires=n_qubits, instructions=tuple(instructions))
    return ArithmeticCandidate(
        circuit,
        scheduling,
        analysis,
        tuple(events),
        peak,
        sum(count - 1 for count in counts.values()),
    )


def lower_arithmetic(
    functions: Sequence[HybridProgram],
    inputs: Mapping[ValueId, Sequence[int]],
    output: Sequence[int],
    *,
    entry: str,
    n_qubits: int,
    workspace: Sequence[int] = (),
    workspace_state: Literal[
        "fresh_zero", "caller_asserted_zero", "unknown"
    ] = "unknown",
    max_operations: int = 100_000,
    scheduling: Schedule = "retain",
    output_mode: OutputMode = "value",
) -> CircuitIR:
    """Return existing CoreIR from the shared value/phase construction emitter."""
    return schedule_arithmetic(
        functions,
        inputs,
        output,
        entry=entry,
        n_qubits=n_qubits,
        workspace=workspace,
        workspace_state=workspace_state,
        max_operations=max_operations,
        scheduling=scheduling,
        output_mode=output_mode,
    ).circuit
