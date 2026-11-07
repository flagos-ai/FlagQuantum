"""Legalize the restricted reversible-phase instruction set."""

from ...core.classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from ...core.classical_dataflow import parse_classical_xor
from ...core.ir import CircuitIR, Instruction
from ...errors import CapabilityError
from ..artifacts.reversible_phase import (
    REVERSIBLE_OPCODE_ARITIES,
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)

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


def _demolition_measurement(
    span: tuple[Instruction, ...],
) -> ReversiblePhaseOperation | None:
    if len(span) != 3:
        return None
    basis, measure, clear = span
    if (
        (basis.name, measure.name, clear.name) != ("h", "measure", "x")
        or not all(_plain(item) for item in span)
        or basis.wires != measure.wires
        or basis.wires != clear.wires
        or basis.metadata
        or set(measure.metadata) - {"is_dynamic", "classical_bit"}
        or set(clear.metadata) - {"conditions", "condition_clauses"}
    ):
        return None
    bit = measure.metadata["classical_bit"]
    if instruction_condition_clauses(clear) != (((bit, 1),),):
        return None
    return ReversiblePhaseOperation("hmr", basis.wires, result_bit=bit)


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
        if instruction.name == "classical_xor":
            if instruction_condition_clauses(instruction):
                raise CapabilityError(
                    "reversible-phase lowering requires a conditional classical result merge"
                )
            assignment = parse_classical_xor(instruction)
            if assignment.sources.count(assignment.target) % 2 == 0:
                operations.append(
                    ReversiblePhaseOperation(
                        "bit_store0", (), result_bit=assignment.target
                    )
                )
            operations.extend(
                ReversiblePhaseOperation(
                    "bit_invert", (), result_bit=assignment.target, condition_bit=source
                )
                for source in assignment.sources
                if source != assignment.target
            )
            if assignment.constant:
                operations.append(
                    ReversiblePhaseOperation(
                        "bit_invert", (), result_bit=assignment.target
                    )
                )
            index += 1
            continue
        if instruction.name == "h":
            span = ir.instructions[index : index + 3]
            hmr = _demolition_measurement(span)
            if hmr is not None:
                operations.append(hmr)
                index += 3
                continue
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
        clauses = instruction_condition_clauses(instruction)
        scopes: tuple[int, ...] = ()
        bit = None
        if clauses:
            if len(clauses) != 1 or any(value != 1 for _, value in clauses[0]):
                raise CapabilityError(
                    f"reversible-phase lowering requires one positive condition clause at {index}"
                )
            slots = tuple(slot for slot, _ in clauses[0])
            scopes, bit = slots[:-1], slots[-1]
            operations.extend(
                ReversiblePhaseOperation("push_condition", (), condition_bit=scope)
                for scope in scopes
            )
        operations.append(
            ReversiblePhaseOperation(
                opcode,
                () if opcode == "neg" else instruction.wires,
                condition_bit=bit,
            )
        )
        operations.extend(ReversiblePhaseOperation("pop_condition", ()) for _ in scopes)
        index += consumed
    return ReversiblePhaseProgram(ir.n_wires, n_bits, tuple(operations))
