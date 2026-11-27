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
            reset_mask = 1 << (width - 1 - gate.wires[0])
            # Reset follows a projective measurement; sum the disjoint sectors.
            state = torch.where(
                (data & reset_mask) == 0, state + state[data ^ reset_mask], 0
            )
        elif active:
            pending.append(
                Instruction(gate.name, gate.wires, gate.params, matrix=gate.matrix)
            )
    flush()
    return state


def certified_and_channel_map(
    gates: tuple[Instruction, ...], basis_bits: tuple[tuple[int, ...], ...]
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Prove the positive-amplitude permutation and every exact-erasure history."""
    outputs, phases, events = certified_and_monomial_map(gates, basis_bits)
    torch.testing.assert_close(phases, torch.ones_like(phases), atol=1e-14, rtol=0)
    return outputs, events


def certified_and_monomial_map(
    gates: tuple[Instruction, ...], basis_bits: tuple[tuple[int, ...], ...]
) -> tuple[tuple[tuple[int, ...], ...], torch.Tensor, int]:
    """Prove a phased permutation from actual primitives and exact And blocks.

    Literal And compute/erase and exact CSwap blocks have already-proved positive
    local branch scalars. Track every input's bits and complex phase across them
    and unconditional monomial gates. Erase requires the actual And invariant;
    its two histories give the same map divided by sqrt(2). Linearity extends
    the resulting columns to coherent states and every combined raw history.
    X-basis lookup erasure retains each outcome's Boolean phase coefficient
    until actual Z/CZ feedback cancels it on every supplied input column.
    Classical XOR slots retain affine functions of earlier measurement outcomes;
    conditional Z/CZ feedback cancels each outcome's phase coefficient separately.
    Unknown gates, unbalanced erase targets and other conditional atoms fail closed.
    """
    from flagquantum.compiler.registers.primitives.and_compute import (
        and_compute_ir,
        and_uncompute_ir,
    )
    from flagquantum.compiler.registers.primitives.controlled_swap import (
        controlled_swap_bit_ir,
    )
    from flagquantum.compiler.registers.primitives.relative_controlled_swap import (
        _relative_swap_gates,
    )

    if not basis_bits or not basis_bits[0]:
        raise AssertionError("Channel certificate needs nonempty basis inputs")
    width = len(basis_bits[0])
    assert all(len(word) == width and set(word) <= {0, 1} for word in basis_bits)
    values = torch.tensor(basis_bits, dtype=torch.uint8).T.contiguous()
    phases = torch.ones(len(basis_bits), dtype=torch.complex128)
    measurement_phases: dict[int, torch.Tensor] = {}
    classical_affine: dict[int, int] = {}
    i, events = 0, 0
    while i < len(gates):
        gate = gates[i]
        if gate.name == "cx":
            end = i
            while end < len(gates) and gates[end].name == "cx":
                end += 1
            pairs = tuple((g.wires[1], g.wires[0]) for g in gates[i:end])
            size = len(pairs)
            fanout = i + 13 * size - 1
            if fanout < len(gates) and gates[fanout].name == "cx":
                control = gates[fanout].wires[0]
                wires = (control, *(w for pair in pairs for w in pair))
                if len(set(wires)) == len(wires):
                    for inverse, layered in (
                        (False, False),
                        (True, False),
                        (False, True),
                        (True, True),
                    ):
                        ordered = pairs[::-1] if inverse else pairs
                        relative_left, relative_right = tuple(
                            zip(*ordered, strict=True)
                        )
                        block = _relative_swap_gates(
                            control,
                            relative_left,
                            relative_right,
                            adjoint=inverse,
                            layer_rotations=layered,
                        )
                        if gates[i : i + len(block)] != block:
                            continue
                        for x, y in ordered:
                            a, b = values[x].clone(), values[y].clone()
                            # The dirty 111 column carries the source minus sign.
                            phases *= 1 - 2 * (values[control] & a & b).to(torch.int64)
                            values[x] = torch.where(values[control].bool(), b, a)
                            values[y] = torch.where(values[control].bool(), a, b)
                        i += len(block)
                        break
                    else:
                        block = ()
                    if block:
                        continue
        if gate.name == "h" and i + 2 < len(gates):
            target = gate.wires[0]
            measured, correction = gates[i + 1 : i + 3]
            if (
                measured.name == "measure"
                and measured.wires == (target,)
                and correction.name == "x"
                and correction.wires == (target,)
            ):
                bit = int(measured.metadata["classical_bit"])
                assert gate == Instruction("h", (target,))
                assert measured == Instruction(
                    "measure",
                    (target,),
                    metadata={"is_dynamic": True, "classical_bit": bit},
                )
                assert correction == Instruction(
                    "x", (target,), metadata={"conditions": ((bit, 1),)}
                )
                assert bit not in measurement_phases
                # X-basis erasure contributes (-1)**(outcome * old_bit).
                # Keep that coefficient until actual later feedback cancels it.
                classical_affine[bit] = 1 << (len(measurement_phases) + 1)
                measurement_phases[bit] = values[target].clone()
                values[target] = 0
                events += 1
                i += 3
                continue
        clauses = instruction_condition_clauses(gate)
        if gate.name == "classical_xor":
            assert not clauses, "Affine proof refuses guarded classical writes"
            assignment = parse_classical_xor(gate)
            expression = int(assignment.constant)
            for bit in assignment.sources:
                assert bit in classical_affine, "XOR reads undefined outcome"
                expression ^= classical_affine[bit]
            classical_affine[assignment.target] = expression
            i += 1
            continue
        if clauses and gate.name in {"z", "cz"}:
            assert gate.matrix is None and not gate.params
            assert set(gate.metadata) == {"conditions"}
            assert len(clauses) == 1 and len(clauses[0]) == 1
            bit, activation = clauses[0][0]
            assert bit in classical_affine
            assert len(gate.wires) == (1 if gate.name == "z" else 2)
            assert len(set(gate.wires)) == len(gate.wires)
            assert all(0 <= wire < width for wire in gate.wires)
            coefficient = values[gate.wires[0]].clone()
            if gate.name == "cz":
                coefficient &= values[gate.wires[1]]
            expression = classical_affine[bit] ^ int(activation == 0)
            if expression & 1:
                phases *= 1 - 2 * coefficient.to(torch.int64)
            for index, phase in enumerate(measurement_phases.values(), start=1):
                if expression >> index & 1:
                    phase ^= coefficient
            i += 1
            continue
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
        assert not instruction_condition_clauses(gate)
        assert gate.matrix is None
        assert len(set(gate.wires)) == len(gate.wires)
        assert all(0 <= wire < width for wire in gate.wires)
        assert set(gate.metadata) <= {"exact_scalar_sign"}
        arities = {
            "i": 1,
            "x": 1,
            "y": 1,
            "z": 1,
            "s": 1,
            "sdg": 1,
            "t": 1,
            "tdg": 1,
            "phase": 1,
            "rz": 1,
            "cx": 2,
            "cy": 2,
            "cz": 2,
            "swap": 2,
            "ccx": 3,
        }
        assert len(gate.wires) == arities.get(gate.name, -1)
        assert set(gate.params) == (
            {"theta"} if gate.name in {"phase", "rz"} else set()
        )
        if gate.name == "x":
            values[gate.wires[0]] ^= 1
        elif gate.name in {"cx", "cy"}:
            control, target = gate.wires
            if gate.name == "cy":
                phases *= torch.where(
                    values[control].bool(),
                    1j * (1 - 2 * values[target].to(torch.int64)),
                    1,
                )
            values[target] ^= values[control]
        elif gate.name == "ccx":
            left, right, target = gate.wires
            values[target] ^= values[left] & values[right]
        elif gate.name == "y":
            target = gate.wires[0]
            phases *= 1j * (1 - 2 * values[target].to(torch.int64))
            values[target] ^= 1
        elif gate.name == "swap":
            left, right = gate.wires
            values[left], values[right] = values[right].clone(), values[left].clone()
        elif gate.name == "cz":
            left, right = gate.wires
            phases *= 1 - 2 * (values[left] & values[right]).to(torch.int64)
        elif gate.name in {"z", "s", "sdg", "t", "tdg", "phase", "rz"}:
            from math import pi

            target_values = values[gate.wires[0]].to(torch.float64)
            angles = {"z": pi, "s": pi / 2, "sdg": -pi / 2, "t": pi / 4, "tdg": -pi / 4}
            angle = (
                angles[gate.name]
                if gate.name in angles
                else float(gate.params["theta"])
            )
            exponent = (
                angle * (target_values - 0.5)
                if gate.name == "rz"
                else angle * target_values
            )
            phases *= torch.exp(1j * exponent)
        elif gate.name != "i":
            raise AssertionError(f"Uncertified monomial gate: {gate.name}")
        i += 1
    assert all(
        torch.count_nonzero(coefficient) == 0
        for coefficient in measurement_phases.values()
    ), "Uncancelled measured lookup phase"
    return tuple(tuple(word) for word in values.T.tolist()), phases, events


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
