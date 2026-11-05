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
