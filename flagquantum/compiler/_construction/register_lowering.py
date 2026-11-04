"""Compose fresh-register effects with the existing arithmetic scheduler.

All declared output storage stays live. Arithmetic and phase segments clean their
own workspace immediately; only that proven-zero tail is reused by later calls.
The returned circuit requires an all-zero entry state; this is an allocation
precondition, not a claim inferred from an arbitrary external state.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from ...core._composition import controlled_instruction
from ...core.ir import CircuitIR, Instruction
from .._hybrid.model import Block, HybridProgram, Operation, Region, Value, ValueId
from .boolean_lowering import BooleanCandidateError, lower_boolean_value
from .lowering import (
    ArithmeticCandidate,
    BoundArithmetic,
    Schedule,
    ScheduleEvent,
    _copy,
    adjoint_value_candidate,
)
from .profile import _fail, _positive_integer, _width, uint_type
from .register_expansion import expand_register_calls, is_register_function
from .register_profile import (
    _ANGLE_GATES,
    register_ports,
    register_width,
    verify_register_program,
)


@dataclass(frozen=True)
class RegisterCandidate:
    """One composed circuit with actual per-segment cleanup/allocation evidence."""

    circuit: CircuitIR
    strategy: Schedule
    registers: Mapping[str, tuple[int, ...]]
    segments: tuple[ArithmeticCandidate, ...]
    peak_workspace: int
    recomputations: int


def _uniform_interval(
    register: tuple[int, ...], start: int, stop: int
) -> tuple[Instruction, ...]:
    """Split the real amplitudes by the populations in each binary prefix."""
    gates: list[Instruction] = []
    for depth, target in enumerate(register):
        half = 1 << (len(register) - depth - 1)
        controls = register[:depth]
        for prefix in range(start // (2 * half), (stop - 1) // (2 * half) + 1):
            left = 2 * half * prefix
            middle = left + half
            zeros = max(0, min(stop, middle) - max(start, left))
            ones = max(0, min(stop, middle + half) - max(start, middle))
            if not ones:
                continue
            angle = 2 * math.atan2(math.sqrt(ones), math.sqrt(zeros))
            rotation = Instruction("ry", (target,), {"theta": angle})
            flips = tuple(
                Instruction("x", (bit,))
                for index, bit in enumerate(controls)
                if not prefix & (1 << (depth - index - 1))
            )
            gates.extend(flips)
            gates.extend(
                controlled_instruction(rotation, controls) if controls else (rotation,)
            )
            gates.extend(flips)
    return tuple(gates)


def schedule_register_program(
    program: HybridProgram,
    functions: Sequence[HybridProgram] = (),
    *,
    n_qubits: int = 256,
    max_operations: int = 100_000,
    scheduling: Schedule = "retain",
    boolean_values: bool = False,
) -> RegisterCandidate:
    """Lower verified fresh outputs without executing or inspecting any state."""
    _positive_integer(max_operations, "max_operations")
    if max_operations > 100_000 or scheduling not in {"retain", "recompute"}:
        _fail("register.budget", "use a known schedule and at most 100000 operations")
    verify_register_program(program, functions, max_qubits=n_qubits)
    program = expand_register_calls(program, functions)
    functions = tuple(f for f in functions if not is_register_function(f))
    labels, bindings, output_width = register_ports(program)
    position = max(
        (max(ports) + 1 for ports in bindings.values()), default=output_width
    )
    local_width = position - output_width
    instructions: list[Instruction] = []
    segments: list[ArithmeticCandidate] = []
    computed: dict[ValueId, ArithmeticCandidate] = {}
    used_boolean = False
    symbols = {f.name: f for f in functions}
    for ordinal, op in enumerate(program.body.blocks[0].operations):
        gates: tuple[Instruction, ...] = ()
        if op.name == "qreg.initialize":
            gates = _copy(op.attributes["value"], bindings[op.operands[1].id])
        elif op.name == "qreg.control_h":
            control, target = (bindings[value.id][0] for value in op.operands[1:])
            gates = controlled_instruction(Instruction("h", (target,)), (control,))
        elif op.name == "qreg.control_phase":
            condition = bindings[op.operands[1].id][0]
            angle = op.attributes["angle"]
            if len(op.operands) == 2:
                gates = (Instruction("phase", (condition,), {"theta": angle}),)
            else:
                target_bits = bindings[op.operands[2].id]
                gates = tuple(
                    Instruction(
                        "cphase", (condition, bit), {"theta": math.ldexp(angle, index)}
                    )
                    for index, bit in enumerate(reversed(target_bits))
                )
        elif op.name == "qreg.uniform_interval":
            gates = _uniform_interval(
                bindings[op.operands[1].id],
                op.attributes["start"],
                op.attributes["stop"],
            )
        elif op.name == "qreg.pair_gate":
            gates = (
                Instruction(
                    op.attributes["gate"],
                    tuple(bindings[value.id][0] for value in op.operands[1:]),
                    params=(
                        {"theta": op.attributes["angle"]}
                        if "angle" in op.attributes
                        else {}
                    ),
                ),
            )
        elif op.name == "qreg.gate":
            ports = bindings[op.operands[1].id]
            gates = (
                Instruction(
                    op.attributes["gate"],
                    (ports[-1 - op.attributes["index"]],),
                    params=(
                        {"theta": op.attributes["angle"]}
                        if op.attributes["gate"] in _ANGLE_GATES
                        else {}
                    ),
                ),
            )
        elif op.name == "qreg.uncompute":
            segment = adjoint_value_candidate(computed.pop(op.operands[1].id))
            segments.append(segment)
            gates = segment.circuit.instructions
        elif op.name == "qreg.phase" and "callee" not in op.attributes:
            angle = op.attributes["angle"]
            if angle:
                # P(theta) X P(theta) X = exp(i theta) I on any existing qubit.
                rotation = Instruction("phase", (0,), params={"theta": angle})
                flip = Instruction("x", (0,))
                gates = (flip, rotation, flip, rotation)
        elif op.name in {"qreg.compute", "qreg.phase"}:
            phase = op.name == "qreg.phase"
            actual = op.operands[1:] if phase else op.operands[2:]
            # A wrapper preserves repeated actual aliases while the original
            # binding verifier still requires disjoint entry argument ports.
            unique = tuple(dict.fromkeys(actual))
            name = f"{program.name}.{op.name.removeprefix('qreg.')}{ordinal}"
            while name in symbols:
                name += ".call"
            borrowed = tuple(
                Value(ValueId(name, index), uint_type(register_width(value)))
                for index, value in enumerate(unique)
            )
            replacements = dict(zip(unique, borrowed, strict=True))
            callee = symbols[op.attributes["callee"]]
            returned = callee.body.blocks[0].operations[-1].operands[0]
            result = Value(ValueId(name, len(unique)), returned.type)
            wrapper = HybridProgram(
                name,
                Region(
                    (
                        Block(
                            borrowed,
                            (
                                Operation(
                                    "qint.call",
                                    tuple(replacements[value] for value in actual),
                                    (result,),
                                    {"callee": callee.name},
                                    location=op.location,
                                ),
                                Operation(
                                    "qint.return", (result,), location=op.location
                                ),
                            ),
                        ),
                    )
                ),
            )
            remaining = max_operations - len(instructions)
            if remaining < 1:
                _fail(
                    "lowering.operations", "whole program exceeds operation budget", op
                )
            segment_functions = (*functions, wrapper)
            inputs = {new.id: bindings[old.id] for old, new in replacements.items()}
            output = (
                ()
                if phase
                else bindings[op.operands[1].id][-_width(returned.type, n_qubits) :]
            )
            workspace = tuple(range(position, n_qubits))
            if phase and "angle" in op.attributes:
                angle = op.attributes["angle"]
                if not angle:
                    continue
                width = _width(returned.type, n_qubits)
                if width > len(workspace):
                    _fail(
                        "lowering.workspace", "phase result exceeds clean workspace", op
                    )
                try:
                    angles = tuple(math.ldexp(angle, bit) for bit in range(width))
                except OverflowError:
                    _fail(
                        "register.phase",
                        "weighted phase angle exceeds finite range",
                        op,
                    )
                value_output = workspace[:width]
                segment = BoundArithmetic(
                    segment_functions, name, inputs, value_output, workspace[width:]
                ).lower(
                    n_qubits=n_qubits, max_operations=remaining, scheduling=scheduling
                )
                adjoint = adjoint_value_candidate(segment)
                rotations = tuple(
                    Instruction("phase", (wire,), params={"theta": radians})
                    for wire, radians in zip(
                        reversed(value_output), angles, strict=True
                    )
                )
                segment = replace(
                    segment,
                    circuit=replace(
                        segment.circuit,
                        instructions=(
                            *segment.circuit.instructions,
                            *rotations,
                            *adjoint.circuit.instructions,
                        ),
                    ),
                    events=(*segment.events, *adjoint.events),
                    peak_workspace=width + segment.peak_workspace,
                    recomputations=segment.recomputations + adjoint.recomputations,
                )
            elif boolean_values and not phase:
                boolean = lower_boolean_value(
                    segment_functions,
                    inputs,
                    output,
                    entry=name,
                    n_qubits=n_qubits,
                    workspace=workspace,
                    workspace_state="fresh_zero",
                    max_operations=remaining,
                )
                # Each cubic write has exactly one compute/write/uncompute triple.
                trace: list[ScheduleEvent] = []
                for gate in boolean.circuit.instructions:
                    if boolean.workspace and gate.wires[-1:] == boolean.workspace:
                        action: Literal["compute", "uncompute"] = (
                            "uncompute" if len(trace) % 2 else "compute"
                        )
                        trace.append(
                            ScheduleEvent(
                                action,
                                boolean.source_result,
                                (),
                                boolean.workspace,
                                1,
                                1,
                            )
                        )
                segment = ArithmeticCandidate(
                    boolean.circuit,
                    scheduling,
                    boolean.lifetimes,
                    tuple(trace),
                    len(boolean.workspace),
                    0,
                    implementation="boolean",
                    implementation_rules=((boolean.source_result, "bounded_anf_xor"),),
                )
                used_boolean = True
            else:
                segment = BoundArithmetic(
                    segment_functions,
                    name,
                    inputs,
                    output,
                    workspace,
                    "phase" if phase else "value",
                ).lower(
                    n_qubits=n_qubits, max_operations=remaining, scheduling=scheduling
                )
            segment = replace(
                segment,
                circuit=replace(
                    segment.circuit, n_wires=position + segment.peak_workspace, shape=()
                ),
            )
            segments.append(segment)
            if not phase:
                computed[op.operands[1].id] = segment
            gates = segment.circuit.instructions
        if len(instructions) + len(gates) > max_operations:
            _fail("lowering.operations", "whole program exceeds operation budget", op)
        instructions.extend(gates)
    if boolean_values and not used_boolean:
        raise BooleanCandidateError("unsupported: no unsigned value segment")
    return RegisterCandidate(
        CircuitIR(n_qubits, tuple(instructions), dtype="complex128"),
        scheduling,
        MappingProxyType(labels),
        tuple(segments),
        local_width + max((s.peak_workspace for s in segments), default=0),
        sum(s.recomputations for s in segments),
    )
