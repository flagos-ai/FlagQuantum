"""Text encoding for the restricted reversible-phase operation stream."""

from ..artifacts.reversible_phase import ReversiblePhaseProgram


def encode_kmx(program: ReversiblePhaseProgram) -> str:
    """Encode the operations in logical wire order."""
    lines = []
    for operation in program.operations:
        fields = [operation.opcode.upper(), *(f"q{wire}" for wire in operation.wires)]
        if operation.result_bit is not None:
            fields.append(f"b{operation.result_bit}")
        if operation.condition_bit is not None:
            fields.extend(("if", f"b{operation.condition_bit}"))
        lines.append(" ".join(fields))
    return "".join(f"{line}\n" for line in lines)
