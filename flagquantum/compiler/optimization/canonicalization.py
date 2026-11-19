"""Canonical circuit rewrites and their shared opcode facts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from math import isclose
from types import MappingProxyType
from typing import Any, cast

from ...core.ir import CircuitIR, Instruction

# The opcodes the pipeline treats as their own inverse. `i` is deliberately absent
# even though the operator schema declares it self-inverse: `remove_identity_gates`
# deletes it one pass earlier, so including it here would add a candidate that can
# never reach these passes. The difference is reported, not closed, by
# `benchmarks/compiler_inverse_cancellation.py`.
_SELF_INVERSE = frozenset(
    {"x", "y", "z", "h", "cx", "cy", "cz", "swap", "ccx", "cswap"}
)

# The opcodes whose single angle parameter is `theta`, which is the parameter their
# adjoint negates. Measured to be exactly the schema's `negate_parameters` opcodes.
_ROTATION_PARAM: Mapping[str, str] = MappingProxyType(
    {
        "rx": "theta",
        "ry": "theta",
        "rz": "theta",
        "phase": "theta",
        "u1": "theta",
        "rxx": "theta",
        "ryy": "theta",
        "rzz": "theta",
        "crx": "theta",
        "cry": "theta",
        "crz": "theta",
        "cphase": "theta",
    }
)


def _rotation_table(analyses: Mapping[str, object] | None) -> Mapping[str, str]:
    """The rotation-parameter table a pass was handed, or this module's own."""

    if analyses is None:
        return _ROTATION_PARAM
    return cast(Mapping[str, str], analyses["rotation_parameters"])


def _self_inverse_table(analyses: Mapping[str, object] | None) -> frozenset[str]:
    """The self-inverse opcode set a pass was handed, or this module's own."""

    if analyses is None:
        return _SELF_INVERSE
    return cast(frozenset[str], analyses["self_inverse_opcodes"])


def _is_zero(value: Any, atol: float = 0.0) -> bool:
    """Use exact scalar zero unless a numerical synthesis caller supplies a tolerance."""
    try:
        if bool(getattr(value, "requires_grad", False)):
            return False
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and value.numel() != 1:
            return False
        if hasattr(value, "item"):
            value = value.item()
        return isclose(float(value), 0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _add_values(left: Any, right: Any) -> Any:
    return left + right


def _replace_param(instruction: Instruction, key: str, value: Any) -> Instruction:
    params = dict(instruction.params)
    params[key] = value
    return Instruction(
        name=instruction.name,
        wires=instruction.wires,
        params=params,
        matrix=instruction.matrix,
        metadata=instruction.metadata,
    )


def remove_identity_gates(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    """Remove every gate that is exactly the identity operator.

    A single-qubit instruction is decided by `is_identity_one_qubit`, which reads
    the angle triple the operator schema declares rather than a table of opcode
    names kept here: that deletes the explicit `i`/`id` set this pass used to
    carry and admits `u3(0, phi, lam)` when `phi + lam` vanishes, a zero-angle
    `rx`, `ry`, `rz`, `phase` or `u1`, and every full turn that returns to `+I`.
    See that function for why a trainable angle refuses, why only exact multiples
    are considered, and why the fold is modulo `4*pi` rather than `2*pi` -- the
    shorter fold would remove `rz(2*pi)`, which is `-I`, and the IR has nowhere
    to record the sign. The import is local because `one_qubit_synthesis` reads
    `_is_zero` from this module.

    A multi-qubit instruction keeps the older rule, which reads the angle
    parameter `_ROTATION_PARAM` names for its opcode, because
    `canonical_euler_angles` describes the single-qubit group only. That leaves
    the `crz(0)`, `rxx(0)`, `cphase(0)` and `rzz(0)` removals this pass already
    performed untouched, and it declines a two-qubit half turn for the same reason
    the single-qubit branch does: `rzz(2*pi)` is `-I`.

    An instruction carrying its own matrix is never removed. The angle says what
    the opcode's matrix would be, not what the caller supplied, so a
    matrix-bearing `rz(0)` is not the identity and `is_identity_one_qubit`
    already declines it on that ground. This branch declines for the same reason
    rather than for a different one, so the two branches cannot disagree about
    the same instruction.
    """

    from .one_qubit_synthesis import is_identity_one_qubit

    rotations = _rotation_table(analyses)
    instructions = []
    for instruction in ir:
        if is_identity_one_qubit(instruction):
            continue
        if instruction.matrix is None:
            param_name = rotations.get(instruction.name)
            if param_name is not None and _is_zero(instruction.params.get(param_name)):
                continue
        instructions.append(instruction)
    return replace(ir, instructions=tuple(instructions))


class _WireLocalProgram:
    """Instructions in program order, indexed by the latest writer of each qubit.

    Both merge passes need the most recent instruction that touches a qubit set,
    and they need to remove or rewrite it. Scanning the output list for that
    instruction costs the length of the untouched prefix, which is quadratic on a
    circuit whose gates share no qubit -- the shape a routing pass produces on a
    wide device. Indexing by qubit instead makes a pass linear in the gates times
    their qubits.

    ``position`` is the insertion sequence number. Sequence numbers only
    increase, so the largest live one touching a qubit is also the latest one in
    program order, and positions do not shift when an earlier instruction is
    removed.
    """

    def __init__(self) -> None:
        self._instructions: dict[int, Instruction] = {}
        self._last_by_wire: dict[int, list[int]] = {}
        self._position = 0

    def last_touching(self, wires: tuple[int, ...]) -> int | None:
        """Return the position of the latest instruction touching any of ``wires``."""

        latest: int | None = None
        for wire in wires:
            stack = self._last_by_wire.get(wire)
            if stack and (latest is None or stack[-1] > latest):
                latest = stack[-1]
        return latest

    def append(self, instruction: Instruction) -> None:
        self._position += 1
        self._instructions[self._position] = instruction
        for wire in instruction.wires:
            self._last_by_wire.setdefault(wire, []).append(self._position)

    def pop(self, position: int) -> None:
        """Remove the instruction at ``position``, a qubit's latest writer.

        Both passes only ever remove the instruction they just looked up, so the
        removed one is by construction the latest writer of every qubit it touches
        and the per-qubit stack needs no repair beyond dropping its top entry.
        """

        instruction = self._instructions.pop(position)
        for wire in instruction.wires:
            stack = self._last_by_wire[wire]
            assert stack[-1] == position
            stack.pop()

    def rewrite(self, position: int, instruction: Instruction) -> None:
        """Replace the instruction at ``position``, keeping the qubits it touches."""

        assert instruction.wires == self._instructions[position].wires
        self._instructions[position] = instruction

    def __getitem__(self, position: int) -> Instruction:
        return self._instructions[position]

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions.values())


def merge_self_inverse(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    """Remove identical self-inverse gates adjacent on their qubits."""

    self_inverse = _self_inverse_table(analyses)
    program = _WireLocalProgram()
    for instruction in ir:
        position = program.last_touching(instruction.wires)
        previous = None if position is None else program[position]
        if (
            instruction.name in self_inverse
            and not (
                instruction.name == "cx"
                and instruction.metadata.get("routing_phase") is not None
            )
            and not (
                previous is not None
                and previous.name == "cx"
                and previous.metadata.get("routing_phase") is not None
            )
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and not instruction.params
            and not previous.params
        ):
            assert position is not None
            program.pop(position)
        else:
            program.append(instruction)
    return replace(ir, instructions=tuple(program))


def merge_adjacent_rotations(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    """Merge rotations adjacent on their qubits."""

    rotations = _rotation_table(analyses)
    program = _WireLocalProgram()
    for instruction in ir:
        param_name = rotations.get(instruction.name)
        position = program.last_touching(instruction.wires)
        previous = None if position is None else program[position]
        if (
            param_name is not None
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and previous.matrix is None
            and instruction.matrix is None
            and param_name in previous.params
            and param_name in instruction.params
        ):
            assert position is not None
            merged_value = _add_values(
                previous.params[param_name], instruction.params[param_name]
            )
            if _is_zero(merged_value):
                program.pop(position)
            else:
                program.rewrite(
                    position,
                    _replace_param(previous, param_name, merged_value),
                )
        else:
            program.append(instruction)
    return replace(ir, instructions=tuple(program))
