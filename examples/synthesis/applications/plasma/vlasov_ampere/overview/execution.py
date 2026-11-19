"""Execute the complete emitted sequence through bounded public Circuit segments."""

import time
from dataclasses import replace

import torch

import flagquantum as fq


def execute(circuit: fq.Circuit, *, segment_size: int = 64) -> torch.Tensor:
    if type(segment_size) is not int or segment_size < 1:
        raise ValueError("Segment size must be a positive integer")
    ir = circuit.to_ir()
    state = torch.zeros(1 << circuit.n_qubits, dtype=torch.complex128)
    state[0] = 1
    started = time.monotonic()
    for first in range(0, len(ir.instructions), segment_size):
        stop = min(first + segment_size, len(ir.instructions))
        segment = fq.Circuit(circuit.n_qubits, dtype=torch.complex128, inputs=state)
        segment.compose(replace(ir, instructions=ir.instructions[first:stop]))
        state = segment.state().reshape(-1)
        if first % 1024 == 0 or stop == len(ir.instructions):
            print(
                f"Executed {stop}/{len(ir.instructions)} original instructions in {time.monotonic() - started:.1f}s",
                flush=True,
            )
    return state
