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

import torch

from ...core._composition import controlled_instruction, inverted_instruction
from ...core.controlled import _clean_mcx_instructions
from ...core.ir import CircuitIR, Instruction
from .._fourier_kernels import qft_instructions
from .._hybrid.model import Block, HybridProgram, Operation, Region, Value, ValueId
from ..state_preparation_synthesis import _synthesize_real_state
from .boolean_lowering import BooleanCandidateError, lower_boolean_value
from .lowering import (
    ArithmeticCandidate,
    BoundArithmetic,
    Schedule,
    ScheduleEvent,
    _copy,
    adjoint_value_candidate,
)
from .pauli_rotation import pauli_rotation_instructions
from .profile import _fail, _positive_integer, _width, uint_type
from .register_expansion import expand_register_calls, is_register_function
from .register_profile import (
    _ANGLE_GATES,
    register_ports,
    register_width,
    semantic_hidden_width,
    unitary_matrix,
    verify_register_program,
)
from .unitary import unitary_instructions


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


def _array_controlled_x(
    controls: tuple[int, ...],
    target: int,
    workspace_start: int,
    n_qubits: int,
    op: Operation,
) -> tuple[tuple[Instruction, ...], int]:
    """Reuse clean ladders and share their scratch across conjunction groups."""
    count = len(controls)
    scratch_width = max(count - 2, 0)
    group_size = count
    flags_width = 0
    # A flag holds each group result while its ladder scratch can be reused.
    # Use the direct ladder on a tie to avoid extra compute/uncompute.
    for size in range(2, count):
        flag_count = (count + size - 1) // size
        scratch = max(size - 2, flag_count - 2)
        if flag_count + scratch < flags_width + scratch_width:
            group_size, flags_width, scratch_width = size, flag_count, scratch

    workspace_width = flags_width + scratch_width
    if workspace_start + workspace_width > n_qubits:
        _fail("lowering.workspace", "array control exceeds clean workspace", op)
    flags = tuple(range(workspace_start, workspace_start + flags_width))
    scratch_wires = tuple(
        range(workspace_start + flags_width, workspace_start + workspace_width)
    )
    gates: list[Instruction] = []

    def append(source: tuple[int, ...], destination: int) -> None:
        gates.extend(
            _clean_mcx_instructions(
                source, destination, scratch_wires[: max(len(source) - 2, 0)]
            )
        )

    if flags:
        groups = tuple(
            controls[i : i + group_size] for i in range(0, count, group_size)
        )
        for group, flag in zip(groups, flags, strict=True):
            append(group, flag)
        append(flags, target)
        for group, flag in zip(reversed(groups), reversed(flags), strict=True):
            append(group, flag)
    else:
        append(controls, target)
    return tuple(gates), workspace_width


def _bounded_control_group(
    gate: Instruction,
    controls: tuple[int, ...],
    position: int,
    n_qubits: int,
) -> tuple[tuple[Instruction, ...], int]:
    # Group verified wide masks without allocating a full control ladder.
    group_count = len(controls) // 3
    if not group_count or position + group_count > n_qubits:
        return controlled_instruction(gate, controls), 0
    flags = tuple(range(position, position + group_count))
    prefix = tuple(
        entry
        for ordinal, flag in enumerate(flags)
        for entry in controlled_instruction(
            Instruction("x", (flag,)), controls[3 * ordinal : 3 * (ordinal + 1)]
        )
    )
    middle = controlled_instruction(gate, (*flags, *controls[3 * group_count :]))
    return (
        *prefix,
        *middle,
        *(inverted_instruction(entry) for entry in reversed(prefix)),
    ), group_count


def _modular_exp_instructions(
    target: tuple[int, ...],
    exponent: tuple[int, ...],
    workspace_start: int,
    n_qubits: int,
    op: Operation,
) -> tuple[tuple[Instruction, ...], int]:
    """Compose controlled modular permutations using adjacent basis swaps."""
    modulus, base = op.attributes["modulus"], op.attributes["base"]
    gates: list[Instruction] = []
    workspace = 0

    def swap(control: int, start: int, end: int) -> None:
        nonlocal workspace
        path = [start]
        while path[-1] != end:
            difference = path[-1] ^ end
            path.append(path[-1] ^ (difference & -difference))
        edges = list(zip(path[:-1], path[1:], strict=True))
        for left, right in (*edges, *reversed(edges[:-1])):
            bit = (left ^ right).bit_length() - 1
            wire = target[-1 - bit]
            controls = (control, *(q for q in target if q != wire))
            flips = tuple(
                Instruction("x", (q,))
                for index, q in enumerate(reversed(target))
                if q != wire and not left & (1 << index)
            )
            middle, used = _array_controlled_x(
                controls, wire, workspace_start, n_qubits, op
            )
            workspace = max(workspace, used)
            gates.extend((*flips, *middle, *flips))

    for index, control in enumerate(reversed(exponent)):
        multiplier = pow(base, 1 << index, modulus)
        seen: set[int] = set()
        for first in range(modulus):
            if first in seen:
                continue
            seen.add(first)
            current = first * multiplier % modulus
            while current != first:
                swap(control, first, current)
                seen.add(current)
                current = current * multiplier % modulus
    return tuple(gates), workspace


def _pattern_gate(
    gate: Instruction, wires: tuple[int, ...], pattern: int
) -> tuple[Instruction, ...]:
    """Apply a single-wire gate on a specified pattern of the other wires."""
    controls = tuple(wire for wire in wires if wire != gate.wires[0])
    flips = tuple(
        Instruction("x", (wire,))
        for index, wire in enumerate(wires)
        if wire in controls and not pattern & (1 << (len(wires) - 1 - index))
    )
    middle = controlled_instruction(gate, controls) if controls else (gate,)
    return (*flips, *middle, *flips)


def _type2_instructions(
    kind: str, data: tuple[int, ...], selector: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Factor a bounded real Type II transform into registered gates."""
    size = 1 << len(data)

    def matrix(*, sine: bool) -> list[list[float]]:
        trig = math.sin if sine else math.cos
        return [
            [
                math.sqrt((1 if row == (size - 1 if sine else 0) else 2) / size)
                * trig(math.pi * (row + int(sine)) * (column + 0.5) / size)
                for column in range(size)
            ]
            for row in range(size)
        ]

    if kind == "combined":
        zeros = [0.0] * size
        values = [row + zeros for row in matrix(sine=False)]
        values.extend(zeros + [-value for value in row] for row in matrix(sine=True))
    else:
        values = matrix(sine=kind == "sine")

    return _real_transform_instructions(values, (*selector, *data))


def _type1_instructions(wires: tuple[int, ...]) -> tuple[Instruction, ...]:
    half = 1 << (len(wires) - 1)
    boundary = half + 1
    size = half * 2
    scale = math.sqrt(2 / half)
    weights = [math.sqrt(0.5), *([1.0] * (half - 1)), math.sqrt(0.5)]
    values = [
        [
            scale
            * weights[row]
            * weights[column]
            * math.cos(math.pi * row * column / half)
            for column in range(boundary)
        ]
        + [0.0] * (size - boundary)
        for row in range(boundary)
    ]
    values.extend(
        [0.0] * boundary
        + [scale * math.sin(math.pi * row * column / half) for column in range(1, half)]
        for row in range(1, half)
    )
    return _real_transform_instructions(values, wires, imaginary_start=boundary)


def _real_transform_instructions(
    values: list[list[float]],
    wires: tuple[int, ...],
    *,
    imaginary_start: int | None = None,
) -> tuple[Instruction, ...]:
    """Factor a real orthogonal matrix, with optional phase i on its final block."""
    dimension = len(values)
    gray = tuple(index ^ (index >> 1) for index in range(dimension))
    rotations: list[tuple[int, int, float]] = []
    # Eliminating in Gray order makes each adjacent row rotation differ in one bit.
    for column_index, column in enumerate(gray):
        for index in range(dimension - 1, column_index, -1):
            upper, lower = gray[index - 1], gray[index]
            x, y = values[upper][column], values[lower][column]
            if abs(y) < 1e-15:
                continue
            angle = math.atan2(y, x)
            cosine, sine = math.cos(angle), math.sin(angle)
            for k, (left, right) in enumerate(
                zip(values[upper], values[lower], strict=True)
            ):
                values[upper][k] = cosine * left + sine * right
                values[lower][k] = -sine * left + cosine * right
            bit = upper ^ lower
            target = wires[len(wires) - bit.bit_length()]
            rotations.append((upper, target, -2 * angle if upper & bit else 2 * angle))

    gates: list[Instruction] = []
    for pattern in range(dimension):
        angle = math.pi if values[pattern][pattern] < 0 else 0.0
        # The final block phase commutes with the exact block-diagonal transform.
        if imaginary_start is not None and pattern >= imaginary_start:
            angle += math.pi / 2
        if not angle:
            continue
        target = wires[0]
        phase = Instruction("phase", (target,), {"theta": angle})
        flip = () if pattern & (dimension >> 1) else (Instruction("x", (target,)),)
        gates.extend((*flip, *_pattern_gate(phase, wires, pattern), *flip))
    for pattern, target, angle in reversed(rotations):
        gates.extend(
            _pattern_gate(
                Instruction("ry", (target,), {"theta": angle}), wires, pattern
            )
        )
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
    hidden_width = semantic_hidden_width(program)
    local_width = position - output_width - hidden_width
    instructions: list[Instruction] = []
    segments: list[ArithmeticCandidate] = []
    computed: dict[ValueId, ArithmeticCandidate] = {}
    used_boolean = False
    control_workspace = 0
    symbols = {f.name: f for f in functions}
    for ordinal, op in enumerate(program.body.blocks[0].operations):
        gates: tuple[Instruction, ...] = ()
        if op.name == "qreg.initialize":
            gates = _copy(op.attributes["value"], bindings[op.operands[1].id])
        elif op.name in {"qreg.prepare_amplitudes", "qreg.inplace_prepare_amplitudes"}:
            wires = (
                tuple(
                    bindings[value.id][-1 - index]
                    for value, index in zip(
                        op.operands[1:], op.attributes["indices"], strict=True
                    )
                )
                if "indices" in op.attributes
                else bindings[op.operands[1].id]
            )
            control_count = op.attributes.get("control_count", 0)
            controls = wires[:control_count]
            gates = _synthesize_real_state(
                op.attributes["amplitudes"], wires[control_count:]
            )
            if op.attributes.get("inverse"):
                gates = tuple(inverted_instruction(gate) for gate in reversed(gates))
            if controls:
                gates = tuple(
                    controlled
                    for gate in gates
                    for controlled in controlled_instruction(gate, controls)
                )
        elif op.name == "qreg.unitary":
            unitary_data = op.operands[1:]
            control = None
            if "control_index" in op.attributes:
                control = bindings[unitary_data[-1].id][
                    -1 - op.attributes["control_index"]
                ]
                unitary_data = unitary_data[:-1]
            if "indices" in op.attributes:
                wires = tuple(
                    bindings[value.id][-1 - index]
                    for value, index in zip(
                        unitary_data, op.attributes["indices"], strict=True
                    )
                )
            else:
                wires = bindings[unitary_data[0].id]
            matrix = unitary_matrix(op.attributes["matrix"], len(wires), op)
            if "power" in op.attributes:
                matrix = torch.linalg.matrix_power(matrix, op.attributes["power"])
            needed = 2 + (control is not None) if len(wires) == 4 else 0
            scratch = (
                tuple(range(position, position + needed))
                if position + needed <= n_qubits
                else ()
            )
            gates = unitary_instructions(
                matrix, wires, op, () if control is None else (control,), scratch
            )
            if scratch:
                used = {
                    wire for gate in gates for wire in gate.wires if wire in scratch
                }
                if used:
                    control_workspace = max(control_workspace, max(used) - position + 1)
        elif op.name in {"qreg.qft", "qreg.inverse_qft"}:
            gates = qft_instructions(bindings[op.operands[1].id])
            if op.name == "qreg.inverse_qft":
                gates = tuple(inverted_instruction(gate) for gate in reversed(gates))
        elif op.name == "qreg.type1_transform":
            gates = _type1_instructions(bindings[op.operands[1].id])
        elif op.name == "qreg.type2_transform":
            data, *selectors = (bindings[value.id] for value in op.operands[1:])
            gates = _type2_instructions(
                op.attributes["kind"], data, selectors[0] if selectors else ()
            )
        elif op.name == "qreg.prepare_ghz":
            first, *rest = bindings[op.operands[1].id]
            gates = (
                Instruction("h", (first,)),
                *(Instruction("cx", (first, target)) for target in rest),
            )
        elif op.name == "qreg.modular_exp":
            residue_ports, exponent = (bindings[v.id] for v in op.operands[1:])
            gates, used_workspace = _modular_exp_instructions(
                residue_ports, exponent, position, n_qubits, op
            )
            control_workspace = max(control_workspace, used_workspace)
        elif op.name == "qreg.control_qft":
            control_ports, data = (bindings[v.id] for v in op.operands[1:])
            transform = qft_instructions(data)
            if op.attributes.get("inverse"):
                transform = tuple(
                    inverted_instruction(gate) for gate in reversed(transform)
                )
            control_value = op.attributes.get("control_value", 1)
            flips = tuple(
                Instruction("x", (qubit,))
                for bit, qubit in enumerate(reversed(control_ports))
                if not (control_value >> bit) & 1
            )
            gates = (
                *flips,
                *(
                    leaf
                    for gate in transform
                    for leaf in controlled_instruction(gate, control_ports)
                ),
                *reversed(flips),
            )
        elif op.name == "qreg.control_h":
            control, target = (bindings[value.id][0] for value in op.operands[1:])
            gates = controlled_instruction(Instruction("h", (target,)), (control,))
        elif op.name == "qreg.pauli_rotation":
            pauli_target = op.operands[1]
            pauli_control = (
                bindings[op.operands[2].id][-1 - op.attributes["control_index"]]
                if "control_index" in op.attributes
                else None
            )
            gates = pauli_rotation_instructions(
                bindings[pauli_target.id],
                op.attributes["support"],
                op.attributes["angle"],
                control=pauli_control,
            )
        elif op.name == "qreg.control_x":
            if "indices" in op.attributes:
                ports = tuple(
                    bindings[value.id][-1 - index]
                    for value, index in zip(
                        op.operands[1:], op.attributes["indices"], strict=True
                    )
                )
                controls, target_wire = ports[:-1], ports[-1]
            else:
                controls, target_ports = (
                    bindings[value.id] for value in op.operands[1:]
                )
                target_wire = target_ports[0]
            if "control_state" in op.attributes and len(controls) >= 5:
                gates, used_workspace = _bounded_control_group(
                    Instruction("x", (target_wire,)), controls, position, n_qubits
                )
                control_workspace = max(control_workspace, used_workspace)
            else:
                gates, used_workspace = _array_controlled_x(
                    controls, target_wire, position, n_qubits, op
                )
                control_workspace = max(control_workspace, used_workspace)
        elif op.name == "qreg.array_phase_flip":
            ports = bindings[op.operands[1].id]
            controls, target = ports[:-1], ports[-1]
            if controls:
                middle, used_workspace = _array_controlled_x(
                    controls, target, position, n_qubits, op
                )
                flip = Instruction("h", (target,))
                gates = (flip, *middle, flip)
                control_workspace = max(control_workspace, used_workspace)
            else:
                gates = (Instruction("z", (target,)),)
        elif op.name == "qreg.control_phase" and "indices" in op.attributes:
            ports = tuple(
                bindings[value.id][-1 - index]
                for value, index in zip(
                    op.operands[1:], op.attributes["indices"], strict=True
                )
            )
            rotation = Instruction(
                "phase", (ports[-1],), {"theta": op.attributes["angle"]}
            )
            if "control_state" in op.attributes and len(ports) >= 6:
                gates, used_workspace = _bounded_control_group(
                    rotation, ports[:-1], position, n_qubits
                )
                control_workspace = max(control_workspace, used_workspace)
            else:
                gates = (
                    controlled_instruction(rotation, ports[:-1])
                    if len(ports) > 1
                    else (rotation,)
                )
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
        elif op.name == "qreg.rzz":
            gates = (
                Instruction(
                    "rzz",
                    bindings[op.operands[1].id],
                    {"theta": op.attributes["angle"]},
                ),
            )
        elif op.name == "qreg.pair_gate":
            operands = op.operands[1:]
            indices = op.attributes.get("indices", (0, 0))
            ports = tuple(
                bindings[value.id][-1 - index]
                for value, index in zip(operands, indices, strict=True)
            )
            gates = (
                Instruction(
                    op.attributes["gate"],
                    ports,
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
            if op.attributes.get("inverse"):
                gates = (inverted_instruction(gates[0]),)
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
                # Track individual ladder wires, including multi-wire cubes.
                trace: list[ScheduleEvent] = []
                live_scratch: set[int] = set()
                for gate in boolean.circuit.instructions:
                    if gate.wires[-1] in boolean.workspace:
                        wire = gate.wires[-1]
                        action: Literal["compute", "uncompute"] = (
                            "uncompute" if wire in live_scratch else "compute"
                        )
                        live_scratch.symmetric_difference_update((wire,))
                        trace.append(
                            ScheduleEvent(
                                action,
                                boolean.source_result,
                                (),
                                (wire,),
                                len(live_scratch),
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
        if "control_state" in op.attributes:
            operands = op.operands[1:]
            count = (
                len(operands) - 1
                if op.name == "qreg.control_x"
                else (
                    len(operands)
                    if op.name == "qreg.control_phase"
                    else op.attributes["control_count"]
                )
            )
            pattern = op.attributes["control_state"]
            flips = tuple(
                Instruction("x", (bindings[value.id][-1 - wire],))
                for bit, (value, wire) in enumerate(
                    zip(operands, op.attributes["indices"], strict=True)
                )
                if bit < count and not pattern & (1 << bit)
            )
            gates = (*flips, *gates, *reversed(flips))
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
        local_width
        + max(control_workspace, max((s.peak_workspace for s in segments), default=0)),
        sum(s.recomputations for s in segments),
    )
