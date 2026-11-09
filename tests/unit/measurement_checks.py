"""Exact unnormalized branch projection shared by register-component tests."""

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.core.classical_control import instruction_condition_clauses
from flagquantum.core.classical_dataflow import ClassicalXor, parse_classical_xor
from flagquantum.core.ir import Instruction


def forced_branch(
    state: torch.Tensor, gates: tuple[Instruction, ...], outcomes: tuple[int, ...]
) -> torch.Tensor:
    """Postselect actual primitive IR, retaining unnormalized branch amplitudes."""
    width = state.numel().bit_length() - 1
    data = torch.arange(state.numel(), dtype=torch.int64)
    classical: dict[int, int] = {}
    pending: list[Instruction] = []
    measurements = iter(outcomes)

    def flush() -> None:
        nonlocal state
        if pending:
            state = (
                Circuit(width, inputs=state, dtype=torch.complex128)
                .append(FieldOperation(width, tuple(pending)))
                .state()
                .flatten()
            )
            pending.clear()

    for gate in gates:
        clauses = instruction_condition_clauses(gate)
        active = not clauses or any(
            all(classical[bit] == value for bit, value in clause) for clause in clauses
        )
        if gate.name == "classical_xor":
            flush()
            assignment = parse_classical_xor(gate)
            if active:
                classical[assignment.target] = assignment.evaluate(classical)
        elif gate.name == "measure":
            flush()
            outcome = next(measurements)
            if not active:
                if outcome:
                    state = torch.zeros_like(state)
                continue
            classical[int(gate.metadata["classical_bit"])] = outcome
            mask = (data >> (width - 1 - gate.wires[0])) & 1
            state = state * (mask == outcome)
        elif gate.name == "reset":
            flush()
            if not active:
                continue
            mask = 1 << (width - 1 - gate.wires[0])
            # Reset follows a projective measurement; sum the disjoint sectors.
            state = torch.where((data & mask) == 0, state + state[data ^ mask], 0)
        elif active:
            pending.append(
                Instruction(gate.name, gate.wires, gate.params, matrix=gate.matrix)
            )
    flush()
    return state


def certified_and_channel_map(
    gates: tuple[Instruction, ...], basis_bits: tuple[tuple[int, ...], ...]
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Prove all erasure branches on basis inputs using certified primitive blocks.

    Each exact And-compute block has positive unit amplitude on a zero target.
    Each exact erase block has identical normalized outputs in both outcomes
    iff its target is the controls' And. Check that invariant for every basis
    input and every block. Linearity then gives the same map / sqrt(2**events)
    for every branch on every superposition of these inputs. Only unconditional
    X/CX/CCX and literal known And/exact-CSwap blocks are accepted. CSwap columns
    are independently proved on all eight inputs before composition. This is a test certificate,
    not an execution backend or a sampled branch check.
    """
    from flagquantum.compiler.registers.primitives.and_compute import (
        and_compute_ir,
        and_uncompute_ir,
    )
    from flagquantum.compiler.registers.primitives.controlled_swap import (
        controlled_swap_bit_ir,
    )

    if not basis_bits or not basis_bits[0]:
        raise AssertionError("Channel certificate needs nonempty basis inputs")
    width = len(basis_bits[0])
    assert all(len(word) == width and set(word) <= {0, 1} for word in basis_bits)
    values = torch.tensor(basis_bits, dtype=torch.uint8).T.contiguous()
    i, events = 0, 0
    while i < len(gates):
        gate = gates[i]
        if (
            gate.name == "cx"
            and i + 2 < len(gates)
            and gates[i + 1].name == "cx"
            and gates[i + 2].name == "h"
        ):
            right, left = gate.wires
            control = gates[i + 1].wires[0]
            if len({control, left, right}) == 3:
                swap = controlled_swap_bit_ir(control, left, right)
                if gates[i : i + len(swap)] == swap:
                    a, b = values[left].clone(), values[right].clone()
                    values[left] = torch.where(values[control].bool(), b, a)
                    values[right] = torch.where(values[control].bool(), a, b)
                    i += len(swap)
                    continue
        if gate.name == "h" and i + 3 < len(gates):
            target = gate.wires[0]
            if gates[i + 1].name == "t":
                assert gates[i + 2].name == gates[i + 3].name == "cx"
                x, y = gates[i + 2].wires[0], gates[i + 3].wires[0]
                block = and_compute_ir(x, y, target)
                assert gates[i : i + len(block)] == block
                assert torch.count_nonzero(values[target]) == 0
                values[target] = values[x] & values[y]
            else:
                assert gates[i + 1].name == "measure"
                bit = int(gates[i + 1].metadata["classical_bit"])
                x, y = gates[i + 2].wires
                block = and_uncompute_ir(x, y, target, bit)
                assert gates[i : i + len(block)] == block
                assert torch.equal(values[target], values[x] & values[y])
                values[target] = 0
                events += 1
            i += len(block)
            continue
        assert gate.name in {"x", "cx", "ccx"} and not gate.metadata and not gate.params
        if gate.name == "x":
            values[gate.wires[0]] ^= 1
        elif gate.name == "cx":
            values[gate.wires[1]] ^= values[gate.wires[0]]
        else:
            a, b, target = gate.wires
            values[target] ^= values[a] & values[b]
        i += 1
    return tuple(tuple(word) for word in values.T.tolist()), events


def expand_reset_branches(
    gates: tuple[Instruction, ...], *, first_slot: int
) -> tuple[Instruction, ...]:
    """Expose reset Kraus outcomes; inactive effects use identity label zero."""
    result: list[Instruction] = []
    slot = first_slot
    for gate in gates:
        if gate.name != "reset":
            result.append(gate)
            continue
        clauses = instruction_condition_clauses(gate)
        if len(clauses) > 1:
            raise ValueError("Reset branch expansion requires one conjunction")
        conditions = clauses[0] if clauses else ()
        metadata: dict[str, object] = {"classical_bit": slot, "is_dynamic": True}
        if conditions:
            result.append(ClassicalXor((), slot).instruction())
            metadata.update(conditions=conditions, guarded_effect=True)
        result.extend(
            (
                Instruction("measure", gate.wires, metadata=metadata),
                Instruction(
                    "x",
                    gate.wires,
                    metadata={"conditions": tuple(sorted((*conditions, (slot, 1))))},
                ),
            )
        )
        slot += 1
    return tuple(result)
