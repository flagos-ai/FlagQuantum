"""Local statevector trajectory execution for dynamic circuits."""

from collections import Counter
from time import perf_counter

import torch

from ...circuit import Circuit
from ...core.ir import Instruction
from ._conditions import classical_width, instruction_conditions
from .circuit import DynamicCircuit
from .result import DynamicExecutionResult


def _measure_wire(
    state: torch.Tensor,
    wire: int,
    n_wires: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    shaped = state.reshape((2,) * n_wires)
    axes = tuple(axis for axis in range(n_wires) if axis != wire)
    magnitudes = torch.abs(shaped) ** 2
    probabilities = magnitudes.sum(dim=axes) if axes else magnitudes
    bit = int(torch.multinomial(probabilities, 1, generator=generator).item())
    selector = torch.arange(2, device=state.device) == bit
    view_shape = [1] * n_wires
    view_shape[wire] = 2
    collapsed = shaped * selector.reshape(view_shape)
    norm = torch.linalg.vector_norm(collapsed)
    return (collapsed / norm).reshape(1, -1), bit


def _apply_instruction(
    state: torch.Tensor,
    instruction: Instruction,
    *,
    n_wires: int,
) -> torch.Tensor:
    circuit = Circuit(
        n_qubits=n_wires,
        bsz=1,
        device=state.device,
        dtype=state.dtype,
        inputs=state,
    )
    circuit._instructions.append(instruction)
    return circuit.state()


def run_dynamic(
    circuit: DynamicCircuit,
    *,
    shots: int,
    seed: int | None = None,
) -> DynamicExecutionResult:
    """Execute independent statevector trajectories with classical feedback."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("run_dynamic requires an experimental DynamicCircuit")
    if int(shots) <= 0:
        raise ValueError("shots must be a positive integer")
    for instruction in circuit._instructions:
        for value in instruction.params.values():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise RuntimeError("dynamic trajectory execution is not differentiable")

    width = classical_width(circuit)
    started = perf_counter()
    branch_counts: Counter[str] = Counter()
    measurement_count = 0
    reset_count = 0
    conditional_applied = 0
    conditional_skipped = 0
    generator = torch.Generator(device=torch.device(circuit.device).type)
    if seed is not None:
        generator.manual_seed(int(seed))
    initial_states = circuit.initial_state()
    batched_states = []
    batched_samples = []
    batched_classical = []
    for batch_index in range(circuit.bsz):
        final_states = []
        final_samples = []
        classical_rows = []
        for _ in range(int(shots)):
            state = initial_states[batch_index : batch_index + 1]
            classical = [-1] * width
            branch_trace = []
            for instruction_index, instruction in enumerate(circuit._instructions):
                conditions = instruction_conditions(instruction)
                skip = False
                for bit_index, expected in conditions:
                    if classical[bit_index] < 0:
                        raise RuntimeError(
                            f"classical bit {bit_index} was read before measurement"
                        )
                    if classical[bit_index] != expected:
                        skip = True
                        break
                if skip:
                    conditional_skipped += 1
                    continue
                if conditions:
                    conditional_applied += 1
                if instruction.name == "measure":
                    state, bit = _measure_wire(
                        state,
                        instruction.wires[0],
                        circuit.n_wires,
                        generator=generator,
                    )
                    classical[int(instruction.metadata["classical_bit"])] = bit
                    measurement_count += 1
                    branch_trace.append((instruction_index, bit))
                elif instruction.name == "reset":
                    state, bit = _measure_wire(
                        state,
                        instruction.wires[0],
                        circuit.n_wires,
                        generator=generator,
                    )
                    if bit:
                        state = _apply_instruction(
                            state,
                            Instruction("x", instruction.wires),
                            n_wires=circuit.n_wires,
                        )
                    reset_count += 1
                    branch_trace.append((instruction_index, bit))
                else:
                    state = _apply_instruction(
                        state,
                        instruction,
                        n_wires=circuit.n_wires,
                    )
            index = int(
                torch.multinomial(
                    torch.abs(state.reshape(-1)) ** 2,
                    1,
                    generator=generator,
                ).item()
            )
            final_states.append(state.reshape(-1))
            final_samples.append(
                [
                    (index >> (circuit.n_wires - wire - 1)) & 1
                    for wire in range(circuit.n_wires)
                ]
            )
            classical_rows.append(classical)
            branch_counts[",".join(f"{index}:{bit}" for index, bit in branch_trace)] += 1
        batched_states.append(torch.stack(final_states))
        batched_samples.append(
            torch.tensor(final_samples, dtype=torch.int64, device=circuit.device)
        )
        batched_classical.append(
            torch.tensor(classical_rows, dtype=torch.int64, device=circuit.device)
        )
    states_tensor = torch.stack(batched_states)
    samples_tensor = torch.stack(batched_samples)
    classical_tensor = torch.stack(batched_classical)
    if circuit.bsz == 1:
        states_tensor = states_tensor[0]
        samples_tensor = samples_tensor[0]
        classical_tensor = classical_tensor[0]
    return DynamicExecutionResult(
        samples=samples_tensor,
        classical_bits=classical_tensor,
        final_states=states_tensor,
        shots=int(shots),
        seed=seed,
        provider_metadata={
            "provider": "flagquantum",
            "device": str(circuit.device),
        },
        statistics={
            "batch_size": circuit.bsz,
            "trajectory_count": circuit.bsz * int(shots),
            "measurement_count": measurement_count,
            "reset_count": reset_count,
            "conditional_applied_count": conditional_applied,
            "conditional_skipped_count": conditional_skipped,
            "branch_count": len(branch_counts),
            "branch_shots": dict(sorted(branch_counts.items())),
            "elapsed_seconds": perf_counter() - started,
            "device": str(circuit.device),
            "seed": seed,
        },
    )

__all__ = ("run_dynamic",)
