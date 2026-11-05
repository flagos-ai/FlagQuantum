"""Exact unnormalized branch projection shared by register-component tests."""

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
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
        if gate.name == "measure":
            flush()
            outcome = next(measurements)
            classical[int(gate.metadata["classical_bit"])] = outcome
            mask = (data >> (width - 1 - gate.wires[0])) & 1
            state = state * (mask == outcome)
        elif gate.name == "reset":
            flush()
            mask = 1 << (width - 1 - gate.wires[0])
            # Reset follows a projective measurement; sum the disjoint sectors.
            state = torch.where((data & mask) == 0, state + state[data ^ mask], 0)
        elif all(
            classical[int(bit)] == value
            for bit, value in gate.metadata.get("conditions", ())
        ):
            pending.append(Instruction(gate.name, gate.wires, gate.params))
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
    X/CX and literal known blocks are accepted. This is a test certificate,
    not an execution backend or a sampled branch check.
    """
    import numpy as np

    from flagquantum.compiler.registers.and_compute import (
        and_compute_ir,
        and_uncompute_ir,
    )

    if not basis_bits or not basis_bits[0]:
        raise AssertionError("Channel certificate needs nonempty basis inputs")
    width = len(basis_bits[0])
    assert all(len(word) == width and set(word) <= {0, 1} for word in basis_bits)
    values = np.asarray(basis_bits, dtype=np.uint8).T.copy()
    i, events = 0, 0
    while i < len(gates):
        gate = gates[i]
        if gate.name == "h" and i + 3 < len(gates):
            target = gate.wires[0]
            if gates[i + 1].name == "t":
                x, y = gates[i + 2].wires[0], gates[i + 3].wires[0]
                block = and_compute_ir(x, y, target)
                assert gates[i : i + len(block)] == block
                np.testing.assert_array_equal(values[target], 0)
                values[target] = values[x] & values[y]
            else:
                assert gates[i + 1].name == "measure"
                bit = int(gates[i + 1].metadata["classical_bit"])
                x, y = gates[i + 2].wires
                block = and_uncompute_ir(x, y, target, bit)
                assert gates[i : i + len(block)] == block
                np.testing.assert_array_equal(values[target], values[x] & values[y])
                values[target] = 0
                events += 1
            i += len(block)
            continue
        assert gate.name in {"x", "cx"} and not gate.metadata and not gate.params
        if gate.name == "x":
            values[gate.wires[0]] ^= 1
        else:
            values[gate.wires[1]] ^= values[gate.wires[0]]
        i += 1
    return tuple(tuple(int(v) for v in word) for word in values.T), events
