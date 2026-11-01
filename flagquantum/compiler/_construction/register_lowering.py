"""Compose fresh-register effects with the existing arithmetic scheduler.

All declared output storage stays live. Arithmetic and phase segments clean their
own workspace immediately; only that proven-zero tail is reused by later calls.
The returned circuit requires an all-zero entry state; this is an allocation
precondition, not a claim inferred from an arbitrary external state.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from ...core.ir import CircuitIR, Instruction
from .._hybrid.model import Block, HybridProgram, Operation, Region, Value, ValueId
from .lowering import (
    ArithmeticCandidate,
    Schedule,
    _copy,
    adjoint_value_candidate,
    schedule_arithmetic,
)
from .profile import _fail, _positive_integer, _width, uint_type
from .register_expansion import expand_register_calls, is_register_function
from .register_profile import register_ports, register_width, verify_register_program


@dataclass(frozen=True)
class RegisterCandidate:
    """One composed circuit with actual per-segment cleanup/allocation evidence."""

    circuit: CircuitIR
    strategy: Schedule
    registers: Mapping[str, tuple[int, ...]]
    segments: tuple[ArithmeticCandidate, ...]
    peak_workspace: int
    recomputations: int


def schedule_register_program(
    program: HybridProgram,
    functions: Sequence[HybridProgram] = (),
    *,
    n_qubits: int = 256,
    max_operations: int = 100_000,
    scheduling: Schedule = "retain",
) -> RegisterCandidate:
    """Lower verified fresh outputs without executing or inspecting any state."""
    _positive_integer(max_operations, "max_operations")
    if max_operations > 100_000 or scheduling not in {"retain", "recompute"}:
        _fail("register.budget", "use a known schedule and at most 100000 operations")
    verify_register_program(program, functions, max_qubits=n_qubits)
    program = expand_register_calls(program, functions)
    functions = tuple(f for f in functions if not is_register_function(f))
    labels, bindings, position = register_ports(program)
    instructions: list[Instruction] = []
    segments: list[ArithmeticCandidate] = []
    computed: dict[ValueId, ArithmeticCandidate] = {}
    symbols = {f.name: f for f in functions}
    for ordinal, op in enumerate(program.body.blocks[0].operations):
        gates: tuple[Instruction, ...] = ()
        if op.name == "qreg.initialize":
            gates = _copy(op.attributes["value"], bindings[op.operands[1].id])
        elif op.name == "qreg.gate":
            ports = bindings[op.operands[1].id]
            gates = (
                Instruction(
                    op.attributes["gate"], (ports[-1 - op.attributes["index"]],)
                ),
            )
        elif op.name == "qreg.uncompute":
            segment = adjoint_value_candidate(computed.pop(op.operands[1].id))
            segments.append(segment)
            gates = segment.circuit.instructions
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
            segment = schedule_arithmetic(
                (*functions, wrapper),
                {new.id: bindings[old.id] for old, new in replacements.items()},
                # Value calls bind the fresh destination's low-weight slice;
                # phase calls have no externally bound value-result storage.
                (
                    ()
                    if phase
                    else bindings[op.operands[1].id][-_width(returned.type, n_qubits) :]
                ),
                entry=name,
                n_qubits=n_qubits,
                workspace=tuple(range(position, n_qubits)),
                workspace_state="fresh_zero",
                max_operations=remaining,
                scheduling=scheduling,
                output_mode="phase" if phase else "value",
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
    return RegisterCandidate(
        CircuitIR(n_qubits, tuple(instructions), dtype="complex128"),
        scheduling,
        MappingProxyType(labels),
        tuple(segments),
        max((s.peak_workspace for s in segments), default=0),
        sum(s.recomputations for s in segments),
    )
