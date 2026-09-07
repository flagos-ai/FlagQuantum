"""Local statevector trajectory execution for dynamic circuits."""

from collections import Counter
from time import perf_counter

import torch

from ...core.ir import Instruction
from ...core.operator_schema import canonical_opcode
from ...simulation.gate_matrix import gate_matrix
from ...simulation.statevector.operations import (
    _DIAGONAL_STATEVECTOR_GATES,
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_single_qubit_fixed,
    _statevector_layout,
)
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
    """Apply one static instruction without constructing a temporary circuit."""

    name = canonical_opcode(instruction.name)
    if name in {"x", "cx", "swap"}:
        return _apply_fixed_permutation(state, name, instruction.wires, n_wires)
    if name == "y":
        return _apply_single_qubit_fixed(
            state,
            name,
            instruction.wires[0],
            n_wires,
        )
    matrix = gate_matrix(
        instruction,
        bsz=state.shape[0],
        device=state.device,
        dtype=state.dtype,
        parameter_bindings=None,
    )
    apply_gate = (
        _apply_diagonal_matrix if name in _DIAGONAL_STATEVECTOR_GATES else _apply_matrix
    )
    return apply_gate(
        state,
        matrix,
        instruction.wires,
        n_wires,
        layout=_statevector_layout(n_wires, instruction.wires),
    )


def _run_dynamic_trajectory(
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
            branch_counts[
                ",".join(f"{index}:{bit}" for index, bit in branch_trace)
            ] += 1
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
            "gate_execution_strategy": "trajectory_direct_statevector_kernel",
            "device": str(circuit.device),
            "seed": seed,
        },
    )


def _measure_wires_batched(
    state: torch.Tensor,
    wire: int,
    n_wires: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    shaped = state.reshape((state.shape[0],) + (2,) * n_wires)
    axes = tuple(axis + 1 for axis in range(n_wires) if axis != wire)
    probabilities = (
        (torch.abs(shaped) ** 2).sum(dim=axes) if axes else (torch.abs(shaped) ** 2)
    )
    bits = torch.multinomial(probabilities, 1, generator=generator).reshape(-1)
    selector = torch.arange(2, device=state.device).reshape(1, 2) == bits.reshape(-1, 1)
    view_shape = [state.shape[0]] + [1] * n_wires
    view_shape[wire + 1] = 2
    collapsed = shaped * selector.reshape(view_shape)
    norms = torch.linalg.vector_norm(collapsed.reshape(state.shape[0], -1), dim=1)
    return collapsed.reshape_as(state) / norms.reshape(-1, 1), bits


def _apply_masked_instruction(
    state: torch.Tensor,
    active: torch.Tensor,
    instruction: Instruction,
    *,
    n_wires: int,
) -> torch.Tensor:
    if bool(torch.all(active)):
        return _apply_instruction(state, instruction, n_wires=n_wires)
    indices = torch.nonzero(active, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        return state
    selected = state.index_select(0, indices)
    updated = _apply_instruction(selected, instruction, n_wires=n_wires)
    return state.index_copy(0, indices, updated)


def _run_dynamic_batched(
    circuit: DynamicCircuit,
    *,
    shots: int,
    seed: int | None,
) -> DynamicExecutionResult:
    width = classical_width(circuit)
    started = perf_counter()
    generator = torch.Generator(device=torch.device(circuit.device).type)
    if seed is not None:
        generator.manual_seed(int(seed))

    state = circuit.initial_state().repeat_interleave(int(shots), dim=0)
    classical = torch.full(
        (int(shots), width),
        -1,
        dtype=torch.int64,
        device=circuit.device,
    )
    branch_values: list[tuple[int, torch.Tensor]] = []
    measurement_count = 0
    reset_count = 0
    conditional_applied = 0
    conditional_skipped = 0

    for instruction_index, instruction in enumerate(circuit._instructions):
        active = torch.ones(int(shots), dtype=torch.bool, device=circuit.device)
        conditions = instruction_conditions(instruction)
        for bit_index, expected in conditions:
            if bool(torch.any(classical[:, bit_index] < 0)):
                raise RuntimeError(
                    f"classical bit {bit_index} was read before measurement"
                )
            active &= classical[:, bit_index] == expected
        active_count = int(torch.count_nonzero(active).item())
        if conditions:
            conditional_applied += active_count
            conditional_skipped += int(shots) - active_count
        if active_count == 0:
            continue

        if instruction.name in {"measure", "reset"}:
            indices = torch.nonzero(active, as_tuple=False).reshape(-1)
            selected, bits = _measure_wires_batched(
                state.index_select(0, indices),
                instruction.wires[0],
                circuit.n_wires,
                generator=generator,
            )
            state = state.index_copy(0, indices, selected)
            recorded_bits = torch.full(
                (int(shots),), -1, dtype=torch.int64, device=circuit.device
            )
            recorded_bits[indices] = bits
            branch_values.append((instruction_index, recorded_bits))
            if instruction.name == "measure":
                classical[indices, int(instruction.metadata["classical_bit"])] = bits
                measurement_count += active_count
            else:
                reset_count += active_count
                one_indices = indices[bits == 1]
                if one_indices.numel():
                    reset_mask = torch.zeros(
                        int(shots), dtype=torch.bool, device=circuit.device
                    )
                    reset_mask[one_indices] = True
                    state = _apply_masked_instruction(
                        state,
                        reset_mask,
                        Instruction("x", instruction.wires),
                        n_wires=circuit.n_wires,
                    )
        else:
            state = _apply_masked_instruction(
                state,
                active,
                instruction,
                n_wires=circuit.n_wires,
            )

    sampled_indices = torch.multinomial(
        torch.abs(state) ** 2,
        1,
        generator=generator,
    ).reshape(-1)
    samples = torch.stack(
        tuple(
            (sampled_indices >> (circuit.n_wires - wire - 1)) & 1
            for wire in range(circuit.n_wires)
        ),
        dim=1,
    )
    branch_counts: Counter[str] = Counter()
    branch_rows = tuple((index, values.tolist()) for index, values in branch_values)
    for shot_index in range(int(shots)):
        branch_counts[
            ",".join(
                f"{index}:{values[shot_index]}"
                for index, values in branch_rows
                if values[shot_index] >= 0
            )
        ] += 1
    return DynamicExecutionResult(
        samples=samples,
        classical_bits=classical,
        final_states=state,
        shots=int(shots),
        seed=seed,
        provider_metadata={
            "provider": "flagquantum",
            "device": str(circuit.device),
        },
        statistics={
            "batch_size": 1,
            "trajectory_count": int(shots),
            "measurement_count": measurement_count,
            "reset_count": reset_count,
            "conditional_applied_count": conditional_applied,
            "conditional_skipped_count": conditional_skipped,
            "branch_count": len(branch_counts),
            "branch_shots": dict(sorted(branch_counts.items())),
            "elapsed_seconds": perf_counter() - started,
            "gate_execution_strategy": "batched_statevector_kernel",
            "device": str(circuit.device),
            "seed": seed,
        },
    )


def run_dynamic(
    circuit: DynamicCircuit,
    *,
    shots: int,
    seed: int | None = None,
    strategy: str = "auto",
    max_batched_bytes: int = 256 * 1024**2,
) -> DynamicExecutionResult:
    """Execute dynamic shots using a reference or vectorized trajectory path."""

    if strategy not in {"auto", "trajectory", "batched"}:
        raise ValueError("strategy must be 'auto', 'trajectory', or 'batched'")
    if int(max_batched_bytes) <= 0:
        raise ValueError("max_batched_bytes must be positive")
    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("run_dynamic requires an experimental DynamicCircuit")
    if int(shots) <= 0:
        raise ValueError("shots must be a positive integer")
    for instruction in circuit._instructions:
        for value in instruction.params.values():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise RuntimeError("dynamic trajectory execution is not differentiable")

    element_size = torch.empty((), dtype=circuit.dtype).element_size()
    estimated_bytes = int(shots) * (2**circuit.n_wires) * element_size * 3
    batched_compatible = circuit.bsz == 1 and estimated_bytes <= max_batched_bytes
    if strategy == "batched" and not batched_compatible:
        reason = (
            "batch size must be one" if circuit.bsz != 1 else "memory budget exceeded"
        )
        raise ValueError(f"batched dynamic execution unavailable: {reason}")
    use_batched = strategy == "batched" or (
        strategy == "auto" and int(shots) >= 32 and batched_compatible
    )
    if use_batched:
        return _run_dynamic_batched(circuit, shots=int(shots), seed=seed)
    return _run_dynamic_trajectory(circuit, shots=int(shots), seed=seed)


__all__ = ("run_dynamic",)
