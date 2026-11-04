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
    comparison_xor,
    constant_multiply_accumulate,
    multiply_accumulate,
    ripple_add,
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


@dataclass(frozen=True)
class ArithmeticCandidate:
    circuit: CircuitIR
    strategy: Schedule
    lifetimes: LifetimeAnalysis
    events: tuple[ScheduleEvent, ...]
    peak_workspace: int
    recomputations: int
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


@dataclass(frozen=True)
class BoundArithmetic:
    """A HybridProgram invocation on owned or externally prepared wires.

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
        replace(event, action="uncompute" if event.action == "compute" else "compute")
        for event in reversed(candidate.events)
    )
    return replace(
        candidate,
        circuit=replace(
            candidate.circuit,
            instructions=tuple(reversed(candidate.circuit.instructions)),
        ),
        events=events,
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
        )
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
            count = emit((*prefix, *writes), final)
            # Empty owned allocation denotes an external XOR result. The paired
            # events account for carry computation/write and exact carry cleanup.
            events.append(
                ScheduleEvent("compute", returned, (), scratch, len(occupied), count)
            )
            count = emit(reversed(prefix), final)
            events.append(
                ScheduleEvent("uncompute", returned, (), scratch, len(occupied), count)
            )
            occupied.difference_update(scratch)
            counts[returned] = 1
            rules.append((returned, "terminal_xor_add"))
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
