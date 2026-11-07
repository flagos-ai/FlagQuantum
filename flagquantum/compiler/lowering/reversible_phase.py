"""Legalize the restricted reversible-phase instruction set."""

from ...core.classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from ...core.ir import CircuitIR, Instruction
from ...errors import CapabilityError
from ..artifacts.reversible_phase import (
    REVERSIBLE_OPCODE_ARITIES,
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from .reversible_classical import lower_classical_assignment, positive_condition_slots

_REVERSIBLE_OPCODES = frozenset(REVERSIBLE_OPCODE_ARITIES) - {
    "hmr",
    "r",
    "neg",
    "push_condition",
    "pop_condition",
    "z_pow",
}


def _plain(instruction: Instruction) -> bool:
    return (
        instruction.matrix is None
        and not instruction.params
        and not (instruction.metadata.get("is_channel"))
    )


def _demolition_measurement(span: tuple[Instruction, ...]) -> int | None:
    if len(span) != 3:
        return None
    basis, measure, clear = span
    if (
        (basis.name, measure.name, clear.name) != ("h", "measure", "x")
        or not all(_plain(item) for item in span)
        or basis.wires != measure.wires
        or basis.wires != clear.wires
        or set(basis.metadata) - {"conditions", "condition_clauses"}
        or set(measure.metadata)
        - {
            "is_dynamic",
            "classical_bit",
            "guarded_effect",
            "conditions",
            "condition_clauses",
        }
        or set(clear.metadata) - {"conditions", "condition_clauses"}
        or measure.metadata.get("is_dynamic") is not True
    ):
        return None
    guards = positive_condition_slots(basis)
    if instruction_condition_clauses(measure) != instruction_condition_clauses(basis):
        return None
    bit = measure.metadata["classical_bit"]
    expected = tuple(sorted({*((slot, 1) for slot in guards), (bit, 1)}))
    if instruction_condition_clauses(clear) != (expected,):
        return None
    return int(bit)


def _x_basis_reset(span: tuple[Instruction, ...]) -> bool:
    if len(span) != 2:
        return False
    basis, reset = span
    return (
        (basis.name, reset.name) == ("h", "reset")
        and all(_plain(item) for item in span)
        and basis.wires == reset.wires
        and not (set(basis.metadata) - {"conditions", "condition_clauses"})
        and not (
            set(reset.metadata)
            - {"is_dynamic", "guarded_effect", "conditions", "condition_clauses"}
        )
        and reset.metadata.get("is_dynamic") is True
        and instruction_condition_clauses(basis) == instruction_condition_clauses(reset)
    )


def _is_scalar_sign(span: tuple[Instruction, ...]) -> bool:
    if tuple(instruction.name for instruction in span) != ("x", "z", "x", "z"):
        return False
    anchor = span[0]
    return (
        anchor.metadata.get("exact_scalar_sign") is True
        and len(anchor.wires) == 1
        and all(
            _plain(instruction)
            and instruction.wires == anchor.wires
            and instruction.metadata == anchor.metadata
            for instruction in span
        )
    )


def _ccz_conjugation(span: tuple[Instruction, ...]) -> Instruction | None:
    if tuple(instruction.name for instruction in span) != ("h", "ccx", "h"):
        return None
    before, controlled_x, after = span
    if (
        not before.metadata
        and not after.metadata
        and all(_plain(instruction) for instruction in span)
        and len(controlled_x.wires) == 3
        and before.wires == after.wires == controlled_x.wires[-1:]
    ):
        return controlled_x
    return None


def lower_reversible_phase(ir: CircuitIR) -> ReversiblePhaseProgram:
    """Lower exact reversible operations and adjacent X-basis clear sequences.

    Validates classical dependencies before forming the target stream. Other
    measurement patterns and parameterized phases require their own rules.
    """
    if ir.observables or ir.measurements:
        raise CapabilityError("reversible-phase lowering needs explicit instructions")
    n_bits = validate_classical_control(ir.instructions, n_qubits=ir.n_wires)
    operations: list[ReversiblePhaseOperation] = []
    index = 0
    while index < len(ir.instructions):
        instruction = ir.instructions[index]
        opcode = instruction.name
        consumed = 1
        result_bit = None
        if instruction.name == "classical_xor":
            operations.extend(lower_classical_assignment(instruction))
            index += 1
            continue
        if instruction.name == "h":
            span = ir.instructions[index : index + 3]
            result_bit = _demolition_measurement(span)
            if result_bit is not None:
                opcode = "hmr"
                consumed = 3
            elif _x_basis_reset(ir.instructions[index : index + 2]):
                opcode = "r"
                consumed = 2
            else:
                controlled_x = _ccz_conjugation(span)
                if controlled_x is not None:
                    instruction = controlled_x
                    opcode = "ccz"
                    consumed = 3
        if instruction.name == "x" and _is_scalar_sign(
            ir.instructions[index : index + 4]
        ):
            opcode = "neg"
            consumed = 4
        supported_metadata = {"conditions", "condition_clauses"}
        if opcode == "neg" and consumed == 4:
            supported_metadata.add("exact_scalar_sign")
        if (
            (consumed == 1 and opcode not in _REVERSIBLE_OPCODES)
            or not _plain(instruction)
            or set(instruction.metadata) - supported_metadata
        ):
            raise CapabilityError(
                f"reversible-phase lowering cannot represent {instruction.name!r} "
                f"at instruction {index}"
            )
        slots = positive_condition_slots(instruction)
        scopes, bit = slots[:-1], slots[-1] if slots else None
        operations.extend(
            ReversiblePhaseOperation("push_condition", (), condition_bit=scope)
            for scope in scopes
        )
        operations.append(
            ReversiblePhaseOperation(
                opcode,
                () if opcode == "neg" else instruction.wires,
                condition_bit=bit,
                result_bit=result_bit,
            )
        )
        operations.extend(ReversiblePhaseOperation("pop_condition", ()) for _ in scopes)
        index += consumed
    written = {
        int(instruction.metadata["classical_bit"])
        for instruction in ir.instructions
        if instruction.name in {"measure", "classical_xor"}
    }
    holes = tuple(bit for bit in range(n_bits) if bit not in written)
    return ReversiblePhaseProgram(ir.n_wires, n_bits, tuple(operations), holes)
