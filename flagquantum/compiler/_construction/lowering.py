"""Exact arithmetic with retained or dependency-aware cleanup/recomputation.

Only verified unsigned expressions enter here. Both schedules compute into an
arbitrary XOR target and restore every supplied zero workspace qubit. The same
X/CX/CCX kernels serve forward and inverse computation without hidden phase.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from ...core.ir import CircuitIR, Instruction
from .._arithmetic_kernels import (
    borrowed_less_xor,
    comparison_xor,
    constant_multiply_accumulate,
    multiply_accumulate,
    ripple_add,
    ripple_subtract_mod,
    square_accumulate,
    xor_add_parts,
)
from .._hybrid.model import INDEX, HybridProgram, Operation, ValueId
from .._hybrid.verifier import Diagnostic, HybridVerificationError
from ..resource_estimation import estimate_resources
from .arithmetic_rewrites import ArithmeticNormalization, normalize_arithmetic
from .bindings import OutputMode, verify_bindings
from .lifetimes import LifetimeAnalysis, analyze_lifetimes
from .profile import _fail, _positive_integer, uint_type
from .ranges import unsigned_width

_WireValue = tuple[int, ...] | int
Schedule = Literal["retain", "recompute"]
InstructionRole = Literal[
    "compute", "uncompute", "xor_write", "copy", "phase", "phase_inverse"
]

# Bound optional duplicate circuit construction; SSA facts remain available.
_NORMALIZATION_PROBE_MAX_OPERATIONS = 10_000


@dataclass(frozen=True)
class ScheduleEvent:
    action: Literal["compute", "uncompute"]
    value: ValueId
    allocation: tuple[int, ...]
    scratch: tuple[int, ...]
    live_workspace: int
    operation_count: int
    instruction_start: int = 0
    instruction_stop: int = 0


@dataclass(frozen=True)
class InstructionSpan:
    """A non-overlapping half-open instruction range and its lowering role."""

    start: int
    stop: int
    role: InstructionRole
    value: ValueId | None


@dataclass(frozen=True)
class OwnershipTransition:
    """Acquire or release physical workspace at an instruction boundary."""

    instruction_index: int
    event_index: int
    action: Literal["allocate", "release"]
    resource: Literal["value", "scratch"]
    value: ValueId
    wires: tuple[int, ...]


@dataclass(frozen=True)
class AllocationReplay:
    """Validated final ownership and peak live workspace for one candidate."""

    peak_workspace: int
    live_allocations: Mapping[ValueId, tuple[int, ...]]
    live_scratch: tuple[int, ...]


@dataclass(frozen=True)
class ArithmeticCandidate:
    circuit: CircuitIR
    strategy: Schedule
    lifetimes: LifetimeAnalysis
    events: tuple[ScheduleEvent, ...]
    peak_workspace: int
    recomputations: int
    instruction_spans: tuple[InstructionSpan, ...] = ()
    ownership_transitions: tuple[OwnershipTransition, ...] = ()
    normalization: ArithmeticNormalization | None = field(default=None, repr=False)
    baseline: ArithmeticCandidate | None = field(default=None, repr=False)
    normalized_cost: tuple[int, int, int] | None = None
    normalization_errors: tuple[Diagnostic, ...] = ()
    baseline_errors: tuple[Diagnostic, ...] = ()
    implementation: Literal["conservative", "arithmetic", "boolean"] = "conservative"
    implementation_rules: tuple[tuple[ValueId, str], ...] = ()
    conservative: ArithmeticCandidate | None = field(default=None, repr=False)
    implementation_cost: tuple[int, int, int] | None = None
    implementation_errors: tuple[Diagnostic, ...] = ()
    conservative_errors: tuple[Diagnostic, ...] = ()


def _ownership_transitions(
    events: Sequence[ScheduleEvent],
) -> tuple[OwnershipTransition, ...]:
    transitions: list[OwnershipTransition] = []
    held_terminal_scratch: tuple[ValueId, tuple[int, ...]] | None = None
    for index, event in enumerate(events):
        terminal_pair = (
            event.action == "compute"
            and not event.allocation
            and index + 1 < len(events)
            and events[index + 1].value == event.value
            and events[index + 1].action == "uncompute"
            and events[index + 1].scratch == event.scratch
        )
        scratch_is_held = held_terminal_scratch == (event.value, event.scratch)
        if event.action == "compute" and event.allocation:
            transitions.append(
                OwnershipTransition(
                    event.instruction_start,
                    index,
                    "allocate",
                    "value",
                    event.value,
                    event.allocation,
                )
            )
        if event.scratch and not scratch_is_held:
            transitions.append(
                OwnershipTransition(
                    event.instruction_start,
                    index,
                    "allocate",
                    "scratch",
                    event.value,
                    event.scratch,
                )
            )
        if terminal_pair:
            held_terminal_scratch = (event.value, event.scratch)
        elif event.scratch:
            transitions.append(
                OwnershipTransition(
                    event.instruction_stop,
                    index,
                    "release",
                    "scratch",
                    event.value,
                    event.scratch,
                )
            )
            held_terminal_scratch = None
        if event.action == "uncompute" and event.allocation:
            transitions.append(
                OwnershipTransition(
                    event.instruction_stop,
                    index,
                    "release",
                    "value",
                    event.value,
                    event.allocation,
                )
            )
    return tuple(transitions)


def replay_allocation_trace(candidate: ArithmeticCandidate) -> AllocationReplay:
    """Validate instruction coverage and replay workspace ownership transitions.

    Instruction spans are disjoint, ordered, and cover the circuit exactly.
    Event spans may overlap classification spans because they separately identify
    the operation associated with each allocation lifecycle transition.
    """
    cursor = 0
    for span in candidate.instruction_spans:
        if span.start != cursor or span.stop < span.start:
            raise ValueError("instruction spans must be ordered and non-overlapping")
        cursor = span.stop
    if cursor != len(candidate.circuit.instructions):
        raise ValueError("instruction spans must cover the generated arithmetic block")
    for event in candidate.events:
        if not 0 <= event.instruction_start <= event.instruction_stop <= cursor:
            raise ValueError("schedule event instruction span is out of bounds")
        if event.instruction_stop - event.instruction_start != event.operation_count:
            raise ValueError("schedule event span disagrees with operation count")

    expected = _ownership_transitions(candidate.events)
    if candidate.ownership_transitions != expected:
        raise ValueError("ownership transitions do not match schedule events")
    active: dict[ValueId, tuple[int, ...]] = {}
    scratch: set[int] = set()
    owners: dict[int, tuple[str, ValueId]] = {}
    peak = 0
    for index, event in enumerate(candidate.events):
        transitions = tuple(
            transition
            for transition in candidate.ownership_transitions
            if transition.event_index == index
        )
        for transition in transitions:
            if transition.action != "allocate":
                continue
            if transition.resource == "value":
                if event.action != "compute" or transition.value in active:
                    raise ValueError("allocation overlaps a live value binding")
            elif transition.value not in active and event.allocation:
                raise ValueError("scratch allocation has no owning value")
            for wire in transition.wires:
                if wire in owners:
                    raise ValueError("workspace wire has overlapping live allocations")
                owners[wire] = (transition.resource, transition.value)
            if transition.resource == "value":
                active[transition.value] = transition.wires
            else:
                scratch.update(transition.wires)
        if event.action == "uncompute" and event.allocation:
            if active.get(event.value) != event.allocation:
                raise ValueError("uncompute uses a stale physical allocation")
        if len(owners) != event.live_workspace:
            raise ValueError("event live workspace disagrees with replayed ownership")
        peak = max(peak, len(owners))
        for transition in transitions:
            if transition.action != "release":
                continue
            for wire in transition.wires:
                if owners.get(wire) != (transition.resource, transition.value):
                    raise ValueError("release does not match a live allocation")
                del owners[wire]
            if transition.resource == "scratch":
                scratch.difference_update(transition.wires)
            elif active.pop(transition.value, None) != transition.wires:
                raise ValueError("release does not match the active value binding")
    if active or scratch or owners:
        raise ValueError("arithmetic workspace is not clean after replay")
    if peak != candidate.peak_workspace:
        raise ValueError("replayed peak workspace disagrees with candidate metric")
    return AllocationReplay(peak, active.copy(), tuple(sorted(scratch)))


@dataclass(frozen=True)
class BoundArithmetic:
    """A HybridProgram invocation on owned or externally prepared qubits.

    Entry and definitions describe the same verified arithmetic language for
    scalar Python and register effects. Output mode records whether to XOR
    the result into storage or apply a one-bit phase and clean the workspace.
    """

    functions: Sequence[HybridProgram]
    entry: str
    inputs: Mapping[ValueId, Sequence[int]]
    output: Sequence[int]
    workspace: Sequence[int]
    output_mode: OutputMode = "value"

    def lower(
        self, *, n_qubits: int, max_operations: int, scheduling: Schedule
    ) -> ArithmeticCandidate:
        return schedule_arithmetic(
            self.functions,
            self.inputs,
            self.output,
            entry=self.entry,
            n_qubits=n_qubits,
            workspace=self.workspace,
            workspace_state="fresh_zero",
            max_operations=max_operations,
            scheduling=scheduling,
            output_mode=self.output_mode,
        )


def _self_adjoint_gates(gates: Sequence[Instruction]) -> bool:
    return all(
        gate.name in {"x", "cx", "ccx"}
        and gate.matrix is None
        and not gate.params
        and not gate.metadata
        for gate in gates
    )


def adjoint_value_candidate(candidate: ArithmeticCandidate) -> ArithmeticCandidate:
    """Reverse the selected clean value circuit and its actual allocation events."""
    computes = Counter(e.value for e in candidate.events if e.action == "compute")
    cleans = Counter(e.value for e in candidate.events if e.action == "uncompute")
    if not _self_adjoint_gates(candidate.circuit.instructions) or computes != cleans:
        _fail("register.adjoint", "cleanup requires a clean exact value candidate")
    events = tuple(
        replace(
            event,
            action="uncompute" if event.action == "compute" else "compute",
            instruction_start=len(candidate.circuit.instructions)
            - event.instruction_stop,
            instruction_stop=len(candidate.circuit.instructions)
            - event.instruction_start,
        )
        for event in reversed(candidate.events)
    )
    instruction_spans = tuple(
        replace(
            span,
            start=len(candidate.circuit.instructions) - span.stop,
            stop=len(candidate.circuit.instructions) - span.start,
            role=(
                "uncompute"
                if span.role == "compute"
                else "compute" if span.role == "uncompute" else span.role
            ),
        )
        for span in reversed(candidate.instruction_spans)
    )
    return replace(
        candidate,
        circuit=replace(
            candidate.circuit,
            instructions=tuple(reversed(candidate.circuit.instructions)),
        ),
        events=events,
        instruction_spans=instruction_spans,
        ownership_transitions=_ownership_transitions(events),
        recomputations=(
            0
            if candidate.implementation == "boolean"
            else sum(count - 1 for count in cleans.values())
        ),
        conservative=(
            adjoint_value_candidate(candidate.conservative)
            if candidate.conservative is not None
            else None
        ),
        baseline=(
            adjoint_value_candidate(candidate.baseline)
            if candidate.baseline is not None
            else None
        ),
    )


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


def _bitwise(
    operator: str,
    words: tuple[_WireValue, ...],
    target: tuple[int, ...],
    signed: tuple[bool, ...],
) -> Iterable[Instruction]:
    words = tuple(
        (
            (word[0],) * max(0, len(target) - len(word)) + word
            if sign and isinstance(word, tuple)
            else word
        )
        for word, sign in zip(words, signed, strict=True)
    )
    if operator in {"~", "|", "^"}:
        for word in words:
            yield from _copy(word, target)
    if operator == "~":
        for wire in target:
            yield Instruction("x", (wire,))
    if operator not in {"&", "|"}:
        return

    for bit, wire in enumerate(reversed(target)):
        controls = []
        for word in words:
            if isinstance(word, int):
                if not word & (1 << bit):
                    break
            elif bit < len(word):
                controls.append(word[-1 - bit])
            else:
                break
        else:
            # A repeated control is idempotent, including x & x and x | x.
            ports = (*dict.fromkeys(controls), wire)
            yield Instruction(("x", "cx", "ccx")[len(ports) - 1], ports)


def _emit_arithmetic(
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
    implementation: Literal["conservative", "arithmetic"] = "conservative",
) -> ArithmeticCandidate:
    """Emit one bounded candidate and its actual zero-pool allocation history.

    Retain is the compatibility default. Recompute eagerly erases non-input
    operands between forward uses, restores missing dependencies before cleanup,
    and retains final-result dependencies through the XOR copy. It is not an
    optimal pebbling search. A terminal one-bit phase product marks its live
    operands directly, keeping both through the marker and exact prefix inverse.
    External zero assertions remain caller obligations.
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
    instruction_spans: list[InstructionSpan] = []
    active: dict[ValueId, _Allocation] = {}
    bound_inputs = {key: tuple(value) for key, value in inputs.items()}
    counts: dict[ValueId, int] = {}
    ensuring: set[ValueId] = set()
    returned = block.operations[-1].operands[0].id
    final = block.operations[-2] if len(block.operations) > 1 else None
    direct_output = (
        implementation == "arithmetic"
        and output_mode == "value"
        and final is not None
        and final.name == "qint.add"
        and final.results[0].id == returned
    )
    rules: list[tuple[ValueId, str]] = []
    phase_product = (
        final
        if output_mode == "phase"
        and final is not None
        and final.name == "qint.multiply"
        and final.results[0].id == returned
        and all(operand.type == uint_type(1) for operand in final.operands)
        else None
    )
    phase_operands = (
        tuple(operand.id for operand in phase_product.operands)
        if phase_product is not None
        else (returned,)
    )

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

    def emit(
        gates: Iterable[Instruction],
        op: Operation,
        *,
        role: InstructionRole,
        value: ValueId | None = None,
    ) -> tuple[int, int, int]:
        start = len(instructions)
        for gate in gates:
            if len(instructions) >= max_operations:
                _fail(
                    "lowering.operations",
                    "compute/uncompute exceeds operation budget",
                    op,
                )
            instructions.append(gate)
        stop = len(instructions)
        if stop > start:
            instruction_spans.append(InstructionSpan(start, stop, role, value))
        return start, stop, stop - start

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

    def carry_free_add(op: Operation) -> bool:
        if op.name != "qint.add":
            return False
        left, right = operands(op)
        if not isinstance(left, tuple) or not isinstance(right, int) or right <= 0:
            return False
        variable = op.operands[1] if op.operands[0].type == INDEX else op.operands[0]
        mask = (1 << unsigned_width(bound(variable.id))) - 1
        return right & mask == 0

    def shifted_multiply(op: Operation) -> tuple[tuple[int, ...], int] | None:
        if op.name != "qint.multiply":
            return None
        left, right = operands(op)
        if isinstance(left, int):
            left, right = right, left
        if (
            not isinstance(left, tuple)
            or not isinstance(right, int)
            or right <= 1
            or right & (right - 1)
        ):
            return None
        variable = op.operands[1] if op.operands[0].type == INDEX else op.operands[0]
        if bound(variable.id) == 0:
            return None
        return left, right.bit_length() - 1

    def multiplication_rule(op: Operation) -> str | None:
        if implementation != "arithmetic" or op.name != "qint.multiply":
            return None
        left, right = operands(op)
        if isinstance(left, tuple) and op.operands[0].id == op.operands[1].id:
            return "square_symmetry"
        if isinstance(left, int):
            left, right = right, left
        if (
            isinstance(left, tuple)
            and isinstance(right, int)
            and right > 1
            and right & (right - 1)
        ):
            return "constant_shift_add"
        return None

    def reserve(key: ValueId, op: Operation) -> _Allocation:
        if op.name in {
            "qint.equal",
            "qint.less",
            "qint.multiply",
            "qint.select",
            "qint.bitwise",
            "qint.subtract_mod",
        } or carry_free_add(op):
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
        if op.name == "qint.bitwise":
            return _bitwise(
                op.attributes["operator"],
                tuple(encoding(value.id) for value in op.operands),
                allocation.result,
                op.attributes.get("signed", (False,) * len(op.operands)),
            )
        if op.name == "qint.subtract_mod":
            left, right = operands(op)
            subtrahend, carry = scratch[:-1], scratch[-1]
            width = len(allocation.result)
            if isinstance(left, tuple):
                left = left[-width:]
            if isinstance(right, tuple):
                right = right[-width:]
            copy_right = _copy(right, subtrahend)
            return (
                *_copy(left, allocation.result),
                *copy_right,
                *ripple_subtract_mod(subtrahend, allocation.result, carry),
                *reversed(copy_right),
            )
        if op.name == "qint.select":
            predicate, yes, no = (encoding(value.id) for value in op.operands)
            assert isinstance(predicate, tuple) and len(predicate) == 1
            control = predicate[0]
            # b XOR (p AND b) XOR (p AND a) selects a/b without flipping p.
            gates = list(_copy(no, allocation.result))
            for branch in (no, yes):
                for gate in _copy(branch, allocation.result):
                    if gate.name == "x":
                        gates.append(Instruction("cx", (control, *gate.wires)))
                    elif gate.wires[0] == control:
                        gates.append(gate)
                    else:
                        gates.append(Instruction("ccx", (control, *gate.wires)))
            return gates
        left, right = operands(op)
        if carry_free_add(op):
            return (*_copy(left, allocation.result), *_copy(right, allocation.result))
        shifted = shifted_multiply(op)
        if shifted is not None:
            variable, shift = shifted
            return _copy(variable, allocation.result[:-shift])
        if op.name in {"qint.equal", "qint.less"}:
            borrowed = borrowed_comparison(op)
            if borrowed is not None:
                return borrowed_less_xor(
                    *borrowed, target=allocation.result[0], carry=scratch[0]
                )
            return comparison_xor(
                left,
                right,
                allocation.result[0],
                scratch[:-1],
                scratch[-1],
                operation="eq" if op.name == "qint.equal" else "lt",
            )
        if op.name == "qint.multiply":
            rule = multiplication_rule(op)
            if rule == "square_symmetry":
                assert isinstance(left, tuple)
                return square_accumulate(left, allocation.result, scratch)
            if rule == "constant_shift_add":
                if isinstance(left, int):
                    left, right = right, left
                assert isinstance(left, tuple) and isinstance(right, int)
                return constant_multiply_accumulate(
                    left, right, allocation.result, scratch
                )
            return multiply_accumulate(left, right, allocation.result, scratch)
        assert not isinstance(left, int)
        width = max(unsigned_width(bound(v.id)) for v in op.operands)
        padding = allocation.owned[: width - len(left)]
        b = allocation.owned[-width - 2 : -2]
        carry, carry_out = allocation.owned[-2:]
        return (*_copy(right, b), *ripple_add((*padding, *left), b, carry, carry_out))

    def borrowed_comparison(
        op: Operation,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        if op.name == "qint.less":
            left, right = operands(op)
            if (
                isinstance(left, tuple)
                and isinstance(right, tuple)
                and len(left) == len(right)
                and len(set((*left, *right))) == len(left) + len(right)
            ):
                return left, right
        return None

    def apply(key: ValueId, *, inverse: bool) -> None:
        op = definitions[key]
        allocation = active[key]
        scratch_count = 0
        if op.name == "qint.multiply" and shifted_multiply(op) is None:
            scratch_count = len(allocation.result)
            rule = multiplication_rule(op)
            if rule is not None:
                rules.append((key, rule))
            if rule == "constant_shift_add":
                left, right = operands(op)
                variable = right if isinstance(left, int) else left
                assert isinstance(variable, tuple)
                scratch_count = max(0, len(allocation.result) - len(variable)) + 1
        elif op.name == "qint.subtract_mod":
            scratch_count = len(allocation.result) + 1
        elif op.name in {"qint.equal", "qint.less"}:
            scratch_count = (
                1
                if borrowed_comparison(op) is not None
                else max(unsigned_width(bound(v.id)) for v in op.operands) + 2
            )
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
        start, stop, count = emit(
            gates,
            op,
            role="uncompute" if inverse else "compute",
            value=key,
        )
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
                start,
                stop,
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
        if op.name == "arith.constant" or (
            (phase_product is not None or direct_output) and key == returned
        ):
            continue
        ensure(key)
        if scheduling == "recompute" and key != returned:
            for dependency in analysis.values[key].dependencies:
                erase(dependency)
            uses = analysis.values[key].uses
            if not uses or uses[-1] <= position:
                erase(key)
    if not direct_output:
        for key in phase_operands:
            ensure(key)
    if output_mode == "phase":
        marker_wires: list[int] = []
        for key in phase_operands:
            predicate = encoding(key)
            assert not isinstance(predicate, int) and len(predicate) == 1
            if predicate[0] not in marker_wires:
                marker_wires.append(predicate[0])
        prefix = tuple(instructions)
        forward_events = tuple(events)
        if not _self_adjoint_gates(prefix):
            _fail(
                "phase.adjoint",
                "phase prefix requires admitted exact self-adjoint gates",
            )
        if 2 * len(forward_events) > 100_000:
            _fail(
                "lowering.schedule.budget",
                "phase inverse exceeds schedule event budget",
            )
        emit(
            (
                Instruction(
                    "z" if len(marker_wires) == 1 else "cz", tuple(marker_wires)
                ),
            ),
            block.operations[-1],
            role="phase",
            value=returned,
        )
        # Invert the actual selected prefix, including its earlier cleanups and
        # physical rebindings. Preparation is composed later, outside this U.
        inverse_start, inverse_stop, _ = emit(
            reversed(prefix), block.operations[-1], role="phase_inverse", value=returned
        )
        for event in reversed(forward_events):
            action: Literal["compute", "uncompute"] = (
                "uncompute" if event.action == "compute" else "compute"
            )
            event_start = inverse_start + len(prefix) - event.instruction_stop
            event_stop = inverse_start + len(prefix) - event.instruction_start
            events.append(
                ScheduleEvent(
                    action,
                    event.value,
                    event.allocation,
                    event.scratch,
                    event.live_workspace,
                    event.operation_count,
                    event_start,
                    event_stop,
                )
            )
            if action == "compute":
                counts[event.value] = counts.get(event.value, 0) + 1
    else:
        if direct_output:
            assert final is not None
            for dependency in analysis.values[returned].dependencies:
                ensure(dependency)
            left, right = operands(final)
            target = tuple(output)[-unsigned_width(bound(returned)) :]
            disjoint = carry_free_add(final)
            scratch_count = 0 if left == right or disjoint else len(target) - 1
            scratch = allocate(scratch_count, final) if scratch_count else ()
            if disjoint:
                prefix, writes = (), (*_copy(left, target), *_copy(right, target))
            else:
                prefix, writes = xor_add_parts(left, right, target, scratch)
            if len(events) + 2 > 100_000:
                _fail(
                    "lowering.schedule.budget",
                    "terminal XOR exceeds event budget",
                    final,
                )
            start, _, prefix_count = emit(prefix, final, role="compute", value=returned)
            _, stop, write_count = emit(writes, final, role="xor_write", value=returned)
            count = prefix_count + write_count
            # Empty owned allocation denotes an external XOR result. The paired
            # events account for carry computation/write and exact carry cleanup.
            events.append(
                ScheduleEvent(
                    "compute", returned, (), scratch, len(occupied), count, start, stop
                )
            )
            start, stop, count = emit(
                reversed(prefix), final, role="uncompute", value=returned
            )
            events.append(
                ScheduleEvent(
                    "uncompute",
                    returned,
                    (),
                    scratch,
                    len(occupied),
                    count,
                    start,
                    stop,
                )
            )
            occupied.difference_update(scratch)
            counts[returned] = 1
            rules.append((returned, "terminal_xor_add"))
        else:
            emit(
                _copy(encoding(returned), tuple(output)),
                block.operations[-1],
                role="copy",
                value=returned,
            )
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
        instruction_spans=tuple(instruction_spans),
        ownership_transitions=_ownership_transitions(events),
        implementation=implementation,
        implementation_rules=tuple(dict.fromkeys(rules)),
    )


def _logical_cost(candidate: ArithmeticCandidate) -> tuple[int, int, int]:
    resource = estimate_resources(candidate.circuit)
    return candidate.peak_workspace, resource.n_operations, resource.depth


def _schedule_normalized_arithmetic(
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
    """Keep the original schedule; adopt a proven non-worse pure value graph.

    This is a bounded local rewrite guard, not target-aware implementation search.
    The existing two schedule checks and incomplete-search contract are unchanged.
    Phase keeps its existing exact prefix/inverse path. A proved rewrite can fit
    where original construction exhausted workspace/operations; that refusal is
    retained rather than inventing an original candidate that was never built.
    """

    def emit(graphs: Sequence[HybridProgram]) -> ArithmeticCandidate:
        return _emit_arithmetic(
            graphs,
            inputs,
            output,
            entry=entry,
            n_qubits=n_qubits,
            workspace=workspace,
            workspace_state=workspace_state,
            max_operations=max_operations,
            scheduling=scheduling,
            output_mode=output_mode,
        )

    if output_mode != "value":
        return emit(functions)
    original = None
    baseline_failure = None
    try:
        original = emit(functions)
    except HybridVerificationError as error:
        if not all(
            d.code.startswith("construction.lowering.") for d in error.diagnostics
        ):
            raise
        baseline_failure = error
    analysis = (
        original.lifetimes
        if original is not None
        else analyze_lifetimes(functions, entry=entry)
    )
    try:
        normalization = normalize_arithmetic(analysis)
    except HybridVerificationError as error:
        if original is None or not all(
            d.code.startswith("construction.budget.") for d in error.diagnostics
        ):
            raise
        return replace(original, normalization_errors=error.diagnostics)
    if (
        normalization.program.semantic_identity
        == normalization.original.semantic_identity
    ):
        if original is None:
            assert baseline_failure is not None
            raise baseline_failure
        return replace(original, normalization=normalization)
    if (
        original is not None
        and len(original.circuit.instructions) > _NORMALIZATION_PROBE_MAX_OPERATIONS
    ):
        return replace(
            original,
            normalization=normalization,
            baseline=original,
            normalization_errors=(
                Diagnostic(
                    "construction.normalization.budget",
                    "optional rewritten circuit emission skipped: original exceeds "
                    f"{_NORMALIZATION_PROBE_MAX_OPERATIONS} operations",
                ),
            ),
        )
    try:
        rewritten = emit((normalization.program,))
    except HybridVerificationError as error:
        if original is None or not all(
            d.code.startswith("construction.lowering.") for d in error.diagnostics
        ):
            raise
        return replace(
            original,
            normalization=normalization,
            baseline=original,
            normalization_errors=error.diagnostics,
        )
    new_cost = _logical_cost(rewritten)
    if original is None:
        assert baseline_failure is not None
        return replace(
            rewritten,
            normalization=normalization,
            normalized_cost=new_cost,
            baseline_errors=baseline_failure.diagnostics,
        )
    old_cost = _logical_cost(original)
    chosen = (
        rewritten
        if all(new <= old for new, old in zip(new_cost, old_cost, strict=True))
        else original
    )
    return replace(
        chosen, normalization=normalization, baseline=original, normalized_cost=new_cost
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
    """Retain AO-01 and check one bounded arithmetic implementation family.

    Local adoption requires nonincreasing workspace/operations/depth and a strict
    gain. Final addition is a genuine XOR writer, even on a nonzero target.
    Phase mode, public selector rows and target-aware choice remain unchanged.
    """
    conservative = None
    refusal = None
    try:
        conservative = _schedule_normalized_arithmetic(
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
        )
    except HybridVerificationError as error:
        if not all(
            d.code.startswith("construction.lowering.") for d in error.diagnostics
        ):
            raise
        refusal = error
    if output_mode != "value":
        if conservative is None:
            assert refusal is not None
            raise refusal
        return conservative
    if conservative is not None:
        normalization = conservative.normalization
        if normalization is None:
            return conservative
        if len(conservative.circuit.instructions) > _NORMALIZATION_PROBE_MAX_OPERATIONS:
            return replace(
                conservative,
                implementation_errors=(
                    Diagnostic(
                        "construction.implementation.budget",
                        "optional arithmetic emission exceeds probe budget",
                    ),
                ),
            )
    else:
        normalization = normalize_arithmetic(analyze_lifetimes(functions, entry=entry))
    operations = normalization.program.body.blocks[0].operations
    eligible = operations[-2].name == "qint.add" if len(operations) > 1 else False
    eligible |= any(
        op.name == "qint.multiply"
        and (
            op.operands[0].id == op.operands[1].id
            or any(value.type == INDEX for value in op.operands)
        )
        for op in operations[:-1]
    )
    if not eligible and conservative is not None:
        return conservative
    try:
        optimized = _emit_arithmetic(
            (normalization.program,),
            inputs,
            output,
            implementation="arithmetic",
            entry=entry,
            n_qubits=n_qubits,
            workspace=workspace,
            workspace_state=workspace_state,
            max_operations=max_operations,
            scheduling=scheduling,
            output_mode=output_mode,
        )
    except HybridVerificationError as error:
        if conservative is None or not all(
            d.code.startswith("construction.lowering.") for d in error.diagnostics
        ):
            raise
        return replace(
            conservative,
            conservative=conservative,
            implementation_errors=error.diagnostics,
        )
    new_cost = _logical_cost(optimized)
    if conservative is None:
        assert refusal is not None
        return replace(
            optimized,
            normalization=normalization,
            implementation_cost=new_cost,
            conservative_errors=refusal.diagnostics,
        )
    old_cost = _logical_cost(conservative)
    chosen = (
        optimized
        if all(a <= b for a, b in zip(new_cost, old_cost, strict=True))
        and new_cost != old_cost
        else conservative
    )
    return replace(
        chosen,
        normalization=normalization,
        baseline=(
            (conservative.baseline or conservative)
            if chosen is optimized
            else conservative.baseline
        ),
        conservative=conservative,
        implementation_cost=new_cost,
        normalized_cost=conservative.normalized_cost,
        normalization_errors=conservative.normalization_errors,
        baseline_errors=conservative.baseline_errors,
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
