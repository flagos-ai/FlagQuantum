"""Local statevector trajectory execution for dynamic circuits."""

from collections import Counter
from time import perf_counter

import torch

from ...core.ir import Instruction
from ...core.operator_schema import canonical_opcode
from ...noise import NoiseModel
from ...simulation.gate_matrix import gate_matrix
from ...simulation.statevector.operations import (
    _DIAGONAL_STATEVECTOR_GATES,
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_single_qubit_fixed,
    _statevector_layout,
)
from ._conditions import classical_width, instruction_condition_clauses
from ._feedback import (
    DynamicFeedbackAction,
    DynamicFeedbackDecision,
    DynamicFeedbackObservation,
    DynamicFeedbackPlan,
    DynamicFeedbackTrace,
)
from ._noise import (
    apply_noise_after_instruction as _apply_noise_after_instruction,
)
from ._noise import apply_readout_error as _apply_readout_error
from ._noise import validate_dynamic_noise as _validate_dynamic_noise
from .circuit import DynamicCircuit
from .result import DynamicExecutionResult


def _validate_feedback_plan(
    circuit: DynamicCircuit, feedback_plan: DynamicFeedbackPlan
) -> None:
    measurement_positions = {
        int(instruction.metadata["classical_bit"]): instruction_index
        for instruction_index, instruction in enumerate(circuit._instructions)
        if instruction.name == "measure"
    }
    for point in feedback_plan.points:
        if point.trigger_classical_bit not in measurement_positions:
            raise ValueError("feedback trigger does not name a measured bit")
        trigger_position = measurement_positions[point.trigger_classical_bit]
        if any(bit not in measurement_positions for bit in point.classical_bits):
            raise ValueError("feedback point references an unmeasured bit")
        if any(
            measurement_positions[bit] > trigger_position
            for bit in point.classical_bits
        ):
            raise ValueError("feedback point reads a measurement after its trigger")
    if any(wire >= circuit.n_wires for wire in feedback_plan.allowed_wires):
        raise ValueError("feedback allowed wire is outside the circuit")


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
    noise_model: NoiseModel | None = None,
    feedback_plan: DynamicFeedbackPlan | None = None,
) -> DynamicExecutionResult:
    """Execute independent statevector trajectories with classical feedback."""

    width = classical_width(circuit)
    started = perf_counter()
    branch_counts: Counter[str] = Counter()
    measurement_count = 0
    reset_count = 0
    conditional_applied = 0
    conditional_skipped = 0
    noise_channel_applications = 0
    bit_flip_events = 0
    readout_errors = 0
    generator = torch.Generator(device=torch.device(circuit.device).type)
    if seed is not None:
        generator.manual_seed(int(seed))
    initial_states = circuit.initial_state()
    batched_states = []
    batched_samples = []
    batched_classical = []
    feedback_traces = []
    points_by_trigger = (
        {}
        if feedback_plan is None
        else {point.trigger_classical_bit: point for point in feedback_plan.points}
    )
    for batch_index in range(circuit.bsz):
        final_states = []
        final_samples = []
        classical_rows = []
        for _ in range(int(shots)):
            state = initial_states[batch_index : batch_index + 1]
            classical = [-1] * width
            true_classical = [-1] * width
            branch_trace = []
            feedback_observations: list[DynamicFeedbackObservation] = []
            feedback_decisions = []
            frame_x_wires: set[int] = set()
            for instruction_index, instruction in enumerate(circuit._instructions):
                clauses = instruction_condition_clauses(instruction)
                for bit_index, _expected in {
                    term for clause in clauses for term in clause
                }:
                    if classical[bit_index] < 0:
                        raise RuntimeError(
                            f"classical bit {bit_index} was read before measurement"
                        )
                skip = bool(clauses) and not any(
                    all(classical[bit] == expected for bit, expected in clause)
                    for clause in clauses
                )
                if skip:
                    conditional_skipped += 1
                    continue
                if clauses:
                    conditional_applied += 1
                if instruction.name == "measure":
                    state, true_bit = _measure_wire(
                        state,
                        instruction.wires[0],
                        circuit.n_wires,
                        generator=generator,
                    )
                    observed, count = _apply_readout_error(
                        torch.tensor(
                            [true_bit], dtype=torch.int64, device=state.device
                        ),
                        instruction.wires[0],
                        noise_model,
                        generator=generator,
                    )
                    bit = int(observed.item())
                    readout_errors += count
                    classical_bit = int(instruction.metadata["classical_bit"])
                    classical[classical_bit] = bit
                    true_classical[classical_bit] = true_bit
                    measurement_count += 1
                    branch_trace.append((instruction_index, bit))
                    point = points_by_trigger.get(classical_bit)
                    if feedback_plan is not None and point is not None:
                        if any(classical[item] < 0 for item in point.classical_bits):
                            raise RuntimeError(
                                f"feedback point {point.name!r} reads an unmeasured bit"
                            )
                        observation = DynamicFeedbackObservation(
                            point_name=point.name,
                            decision_index=len(feedback_observations),
                            classical_bits=point.classical_bits,
                            true_bits=tuple(
                                true_classical[item] for item in point.classical_bits
                            ),
                            observed_bits=tuple(
                                classical[item] for item in point.classical_bits
                            ),
                            frame_x_wires_before=tuple(sorted(frame_x_wires)),
                        )
                        feedback_observations.append(observation)
                        action = feedback_plan.controller.decide(
                            tuple(feedback_observations)
                        )
                        if not isinstance(action, DynamicFeedbackAction):
                            raise TypeError(
                                "feedback controller must return DynamicFeedbackAction"
                            )
                        if action.mode != "none":
                            if action.mode not in feedback_plan.allowed_action_modes:
                                raise ValueError(
                                    "feedback action mode is outside the plan"
                                )
                            if (
                                action.wire is None
                                or action.wire not in feedback_plan.allowed_wires
                            ):
                                raise ValueError("feedback wire is outside the plan")
                            if action.wire >= circuit.n_wires:
                                raise ValueError("feedback wire is outside the circuit")
                            if action.mode == "physical_x":
                                feedback_instruction = Instruction("x", (action.wire,))
                                state = _apply_instruction(
                                    state,
                                    feedback_instruction,
                                    n_wires=circuit.n_wires,
                                )
                                state, applications, events = (
                                    _apply_noise_after_instruction(
                                        state,
                                        feedback_instruction,
                                        noise_model,
                                        n_wires=circuit.n_wires,
                                        generator=generator,
                                    )
                                )
                                noise_channel_applications += applications
                                bit_flip_events += events
                            elif action.mode == "frame_x":
                                if action.wire in frame_x_wires:
                                    frame_x_wires.remove(action.wire)
                                else:
                                    frame_x_wires.add(action.wire)
                        feedback_decisions.append(
                            DynamicFeedbackDecision(
                                observation=observation,
                                action=action,
                                frame_x_wires_after=tuple(sorted(frame_x_wires)),
                            )
                        )
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
                    state, applications, events = _apply_noise_after_instruction(
                        state,
                        instruction,
                        noise_model,
                        n_wires=circuit.n_wires,
                        generator=generator,
                    )
                    noise_channel_applications += applications
                    bit_flip_events += events
            index = int(
                torch.multinomial(
                    torch.abs(state.reshape(-1)) ** 2,
                    1,
                    generator=generator,
                ).item()
            )
            final_states.append(state.reshape(-1))
            sample = torch.tensor(
                [
                    (index >> (circuit.n_wires - wire - 1)) & 1
                    for wire in range(circuit.n_wires)
                ],
                dtype=torch.int64,
                device=state.device,
            )
            for wire in range(circuit.n_wires):
                observed, count = _apply_readout_error(
                    sample[wire : wire + 1],
                    wire,
                    noise_model,
                    generator=generator,
                )
                sample[wire] = observed[0]
                readout_errors += count
            for wire in frame_x_wires:
                sample[wire] ^= 1
            final_samples.append(sample.tolist())
            classical_rows.append(classical)
            if feedback_plan is not None:
                feedback_traces.append(DynamicFeedbackTrace(tuple(feedback_decisions)))
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
            "noise_model_identity": (
                None if noise_model is None else noise_model.identity
            ),
            "noise_channel_application_count": noise_channel_applications,
            "bit_flip_event_count": bit_flip_events,
            "readout_error_count": readout_errors,
            "feedback_decision_count": sum(
                len(trace.decisions) for trace in feedback_traces
            ),
        },
        feedback_traces=tuple(feedback_traces),
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
    noise_model: NoiseModel | None,
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
    noise_channel_applications = 0
    bit_flip_events = 0
    readout_errors = 0

    for instruction_index, instruction in enumerate(circuit._instructions):
        active = torch.ones(int(shots), dtype=torch.bool, device=circuit.device)
        clauses = instruction_condition_clauses(instruction)
        for bit_index in {bit for clause in clauses for bit, _ in clause}:
            if bool(torch.any(classical[:, bit_index] < 0)):
                raise RuntimeError(
                    f"classical bit {bit_index} was read before measurement"
                )
        if clauses:
            active = torch.zeros(int(shots), dtype=torch.bool, device=circuit.device)
            for clause in clauses:
                clause_active = torch.ones(
                    int(shots), dtype=torch.bool, device=circuit.device
                )
                for bit_index, expected in clause:
                    clause_active &= classical[:, bit_index] == expected
                active |= clause_active
        active_count = int(torch.count_nonzero(active).item())
        if clauses:
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
            recorded = bits
            if instruction.name == "measure":
                recorded, count = _apply_readout_error(
                    bits,
                    instruction.wires[0],
                    noise_model,
                    generator=generator,
                )
                readout_errors += count
            recorded_bits = torch.full(
                (int(shots),), -1, dtype=torch.int64, device=circuit.device
            )
            recorded_bits[indices] = recorded
            branch_values.append((instruction_index, recorded_bits))
            if instruction.name == "measure":
                classical[indices, int(instruction.metadata["classical_bit"])] = (
                    recorded
                )
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
            indices = torch.nonzero(active, as_tuple=False).reshape(-1)
            selected, applications, events = _apply_noise_after_instruction(
                state.index_select(0, indices),
                instruction,
                noise_model,
                n_wires=circuit.n_wires,
                generator=generator,
            )
            state = state.index_copy(0, indices, selected)
            noise_channel_applications += applications
            bit_flip_events += events

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
    for wire in range(circuit.n_wires):
        observed, count = _apply_readout_error(
            samples[:, wire],
            wire,
            noise_model,
            generator=generator,
        )
        samples[:, wire] = observed
        readout_errors += count
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
            "noise_model_identity": (
                None if noise_model is None else noise_model.identity
            ),
            "noise_channel_application_count": noise_channel_applications,
            "bit_flip_event_count": bit_flip_events,
            "readout_error_count": readout_errors,
        },
    )


def run_dynamic(
    circuit: DynamicCircuit,
    *,
    shots: int,
    seed: int | None = None,
    strategy: str = "auto",
    max_batched_bytes: int = 256 * 1024**2,
    noise_model: NoiseModel | None = None,
    _feedback_plan: DynamicFeedbackPlan | None = None,
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
    _validate_dynamic_noise(circuit, noise_model)
    if _feedback_plan is not None:
        if not isinstance(_feedback_plan, DynamicFeedbackPlan):
            raise TypeError("_feedback_plan must be a DynamicFeedbackPlan or None")
        if strategy == "batched":
            raise ValueError(
                "decoder feedback is available only with trajectory execution"
            )
        _validate_feedback_plan(circuit, _feedback_plan)

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
    if _feedback_plan is not None:
        use_batched = False
    if use_batched:
        return _run_dynamic_batched(
            circuit,
            shots=int(shots),
            seed=seed,
            noise_model=noise_model,
        )
    return _run_dynamic_trajectory(
        circuit,
        shots=int(shots),
        seed=seed,
        noise_model=noise_model,
        feedback_plan=_feedback_plan,
    )


__all__ = ("run_dynamic",)
