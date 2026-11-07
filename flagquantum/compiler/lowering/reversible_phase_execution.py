"""Bind the reversible-phase target to local dynamic instructions."""

from ...core.classical_dataflow import ClassicalXor
from ...core.ir import Instruction
from ..artifacts.reversible_phase import ReversiblePhaseProgram
from ..registers.global_phase import ccz_ir, scalar_sign_ir


def bind_reversible_phase_execution(
    program: ReversiblePhaseProgram, *, classical_offset: int = 0
) -> tuple[Instruction, ...]:
    """Expand target operations with an exclusively owned classical slot range.

    Slots start at classical_offset. A scope snapshots its guard into a slot
    above the declared result range, reusing it after scope exit. The target
    program remains the source for native opcode counts.
    """
    if type(classical_offset) is not int or classical_offset < 0:
        raise ValueError("Classical offset must be a nonnegative integer")
    first_scratch = classical_offset + program.n_bits
    gates = [
        ClassicalXor((), bit).instruction()
        for bit in range(classical_offset, first_scratch)
    ]
    scopes: list[int] = []
    for op in program.operations:
        predicate = (
            None if op.condition_bit is None else classical_offset + op.condition_bit
        )
        if op.opcode == "push_condition":
            assert predicate is not None
            scratch = first_scratch + len(scopes)
            gates.append(ClassicalXor((predicate,), scratch).instruction())
            scopes.append(scratch)
            continue
        if op.opcode == "pop_condition":
            scopes.pop()
            continue

        result = None if op.result_bit is None else classical_offset + op.result_bit
        if op.opcode in {"bit_store0", "bit_invert"}:
            assert result is not None
            sources = () if op.opcode == "bit_store0" else (result,)
            if predicate is not None:
                sources = (*sources, predicate)
            gates.append(
                ClassicalXor(
                    sources,
                    result,
                    constant=op.opcode == "bit_invert" and predicate is None,
                ).instruction()
            )
            continue

        # An inline HMR predicate can alias its own output.
        if op.opcode == "hmr" and predicate is not None and predicate == result:
            scratch = first_scratch + len(scopes)
            gates.append(ClassicalXor((predicate,), scratch).instruction())
            predicate = scratch
        guards = (*scopes, *((predicate,) if predicate is not None else ()))
        conditions = tuple(sorted((bit, 1) for bit in guards))
        metadata: dict[str, object] = {"conditions": conditions} if conditions else {}
        if op.opcode in {"hmr", "r"}:
            gates.append(Instruction("h", op.qubits, metadata=metadata))
            effect = {**metadata, "is_dynamic": True}
            if conditions:
                effect["guarded_effect"] = True
            if op.opcode == "hmr":
                assert result is not None
                gates.append(
                    Instruction(
                        "measure",
                        op.qubits,
                        metadata={**effect, "classical_bit": result},
                    )
                )
                clear = tuple(sorted((*conditions, (result, 1))))
                gates.append(
                    Instruction("x", op.qubits, metadata={"conditions": clear})
                )
            else:
                gates.append(Instruction("reset", op.qubits, metadata=effect))
        elif op.opcode == "ccz":
            a, b, c = op.qubits
            gates.extend(ccz_ir((a, b, c), conditions=conditions))
        elif op.opcode == "neg":
            gates.extend(scalar_sign_ir(0, conditions=conditions))
        else:
            gates.append(Instruction(op.opcode, op.qubits, metadata=metadata))
    return tuple(gates)
