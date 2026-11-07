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
    "push_condition",
    "pop_condition",
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
        if instruction.name == "classical_xor":
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
            index += 1
            continue
        if instruction.name == "h":
            hmr = _demolition_measurement(ir.instructions[index : index + 3])
            if hmr is not None:
                operations.append(hmr)
                index += 3
                continue
        if (
            instruction.name not in _REVERSIBLE_OPCODES
            or not _plain(instruction)
            or set(instruction.metadata) - {"conditions", "condition_clauses"}
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
                instruction.name, instruction.wires, condition_bit=bit
            )
        )
        operations.extend(ReversiblePhaseOperation("pop_condition", ()) for _ in scopes)
        index += 1
    return ReversiblePhaseProgram(ir.n_wires, n_bits, tuple(operations))
