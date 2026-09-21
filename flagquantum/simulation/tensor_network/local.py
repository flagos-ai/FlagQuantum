"""Local tensor-network plan construction and numerical execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..gate_matrix import gate_matrix
from .models import (
    CompiledTNProgram,
    ContractionPathStep,
    TensorNetworkContractionPlan,
    TensorNetworkNode,
)
from .path_search import _as_ir
from .state import TensorNetworkState


def _initial_state_tensors(
    circuit_or_ir: Any,
    *,
    bsz: int,
    n_wires: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], bool]:
    """Keep explicit inputs joint and represent the default zero state sparsely."""

    if getattr(circuit_or_ir, "_inputs", None) is not None:
        state = circuit_or_ir.initial_state().to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        reshaped = state.reshape((state.shape[0],) + (2,) * n_wires)
        return (reshaped,), True
    zero = torch.zeros(int(bsz), 2, dtype=dtype, device=device)
    zero[:, 0] = 1
    return tuple(zero for _ in range(n_wires)), False


def _build_tensor_network_plan(
    circuit_or_ir: Any,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype | None,
    write_cache: bool,
) -> TensorNetworkContractionPlan:
    """Walk the program once, materializing node tensors on ``device``."""

    ir = _as_ir(circuit_or_ir)
    dtype = dtype or torch.complex64
    initial_tensors, joint_initial_state = _initial_state_tensors(
        circuit_or_ir, bsz=bsz, n_wires=ir.n_wires, device=device, dtype=dtype
    )
    bsz = int(initial_tensors[0].shape[0]) if initial_tensors else int(bsz)
    batch_label = 0
    current_labels = list(range(1, ir.n_wires + 1))
    next_label = ir.n_wires + 1
    nodes: list[TensorNetworkNode] = []
    dynamic_tensors = list(initial_tensors)
    binding_source = getattr(circuit_or_ir, "_parameter_bindings", None)
    parameter_bindings = None if binding_source is None else binding_source.values()
    # A shape-only build must not read or write the circuit's program cache: the
    # compiled template describes a device the caller never asked to execute on.
    program_cache = (
        getattr(circuit_or_ir, "_backend_programs", None) if write_cache else None
    )
    program_key = ("tensor_network", bsz)
    compiled_structure = (
        None if program_cache is None else program_cache.get(program_key)
    )
    if compiled_structure is not None and not isinstance(
        compiled_structure, CompiledTNProgram
    ):
        raise TypeError("cached tensor-network program must be a CompiledTNProgram")
    # Binding many small node objects costs more than rebuilding descriptors on
    # CPU. Reuse the compiled template only with the fused CUDA path.
    active_structure = (
        compiled_structure if torch.device(device).type == "cuda" else None
    )

    if active_structure is None:
        if joint_initial_state:
            nodes.append(
                TensorNetworkNode(
                    tensor=initial_tensors[0],
                    labels=(batch_label, *current_labels),
                    name="initial_state",
                )
            )
        else:
            nodes.extend(
                TensorNetworkNode(
                    tensor=tensor,
                    labels=(batch_label, current_labels[wire]),
                    name=f"init_{wire}",
                )
                for wire, tensor in enumerate(initial_tensors)
            )

    for index, instruction in enumerate(ir.instructions):
        if instruction.metadata.get("is_channel"):
            raise NotImplementedError(
                "Tensor network mode currently supports unitary circuit instructions."
            )
        matrix = gate_matrix(
            instruction,
            bsz=bsz,
            device=device,
            dtype=dtype,
            parameter_bindings=parameter_bindings,
        )
        if matrix.ndim == 2:
            tensor = matrix.reshape((2,) * len(instruction.wires) * 2)
            input_labels = tuple(current_labels[wire] for wire in instruction.wires)
            output_labels = tuple(
                range(next_label, next_label + len(instruction.wires))
            )
            next_label += len(instruction.wires)
            labels = output_labels + input_labels
        else:
            tensor = matrix.reshape((bsz,) + (2,) * len(instruction.wires) * 2)
            input_labels = tuple(current_labels[wire] for wire in instruction.wires)
            output_labels = tuple(
                range(next_label, next_label + len(instruction.wires))
            )
            next_label += len(instruction.wires)
            labels = (batch_label,) + output_labels + input_labels
        for wire, label in zip(instruction.wires, output_labels, strict=True):
            current_labels[wire] = label
        dynamic_tensors.append(tensor)
        if active_structure is None:
            nodes.append(
                TensorNetworkNode(
                    tensor=tensor,
                    labels=labels,
                    name=f"{index}:{instruction.name}",
                    metadata={"wires": instruction.wires},
                )
            )

    if active_structure is not None:
        return active_structure.bind(dynamic_tensors, program_cache=program_cache)

    output_labels = (batch_label,) + tuple(current_labels)
    path = tuple(
        ContractionPathStep(
            node_index=index,
            name=node.name,
            labels=node.labels,
            shape=tuple(node.tensor.shape),
        )
        for index, node in enumerate(nodes)
    )
    plan = TensorNetworkContractionPlan(
        n_wires=ir.n_wires,
        bsz=bsz,
        nodes=tuple(nodes),
        output_labels=output_labels,
        path=path,
        program_cache=program_cache,
    )
    if program_cache is not None and compiled_structure is None:
        program_cache[program_key] = CompiledTNProgram.compile(plan)
    return plan


def build_local_tensor_network(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> TensorNetworkContractionPlan:
    """Build a local tensor-network contraction plan from a circuit or IR."""

    return _build_tensor_network_plan(
        circuit_or_ir,
        bsz=bsz,
        device=device,
        dtype=dtype,
        write_cache=True,
    )


def build_shape_only_tensor_network(
    circuit_or_ir: Any,
    *,
    bsz: int | None = None,
    dtype: torch.dtype | None = None,
) -> TensorNetworkContractionPlan:
    """Build the node structure of the numerical build while holding no state.

    Every node carries a meta-device tensor whose shape, labels, dtype and element
    size match what the numerical build would produce, so a contraction peak can be
    computed without allocating it. No program is compiled into the circuit's
    backend cache and no device is touched, so a caller can ask this during
    planning and leave the program exactly as it was found. ``bsz`` and ``dtype``
    default to the circuit's own, which is how the local runner resolves them.
    """

    if hasattr(circuit_or_ir, "to_ir"):
        if bsz is None:
            bsz = circuit_or_ir.bsz
        if dtype is None:
            dtype = circuit_or_ir.dtype
    return _build_tensor_network_plan(
        circuit_or_ir,
        bsz=1 if bsz is None else int(bsz),
        device="meta",
        dtype=dtype,
        write_cache=False,
    )


def tensor_network_contraction_peak_bytes(
    circuit_or_ir: Any,
    *,
    bsz: int | None = None,
    dtype: torch.dtype | None = None,
    contraction_strategy: str = "greedy",
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    sliced_labels: Sequence[int] | None = None,
) -> int:
    """Return the peak working-set bytes of one contraction order for a program.

    The peak is a property of the contraction order rather than of the program, so
    it is reported for one explicitly selected order. The default is the local
    runner's own default order: the runner's peak and this number describe the same
    contraction only when the two orders agree, and a caller that changes one must
    change the other. Passing a byte budget reports the peak of the sliced order
    that budget selects, which is how a caller can check a budget is actually met.

    Shape-only and side-effect free; see
    :func:`build_shape_only_tensor_network`. The element size is the widest node
    element size, which is how the slicing planner sizes the same peak.
    """

    plan = build_shape_only_tensor_network(circuit_or_ir, bsz=bsz, dtype=dtype)
    element_size = max(
        (int(node.tensor.element_size()) for node in plan.nodes), default=1
    )
    profile = plan.contraction_profile(
        contraction_strategy,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        sliced_labels=sliced_labels,
    )
    return int(profile.peak_size) * element_size


def ensure_local_tensor_network_plan(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
) -> TensorNetworkContractionPlan:
    """Return an existing plan or build one with the caller's local settings."""

    if isinstance(plan_or_circuit, TensorNetworkContractionPlan):
        return plan_or_circuit
    if hasattr(plan_or_circuit, "to_ir"):
        return build_local_tensor_network(
            plan_or_circuit,
            bsz=plan_or_circuit.bsz,
            device=plan_or_circuit.device,
            dtype=plan_or_circuit.dtype,
        )
    return build_local_tensor_network(plan_or_circuit)


def run_local_tensor_network(
    circuit_or_ir: Any,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype | None,
    contraction_strategy: str,
    max_intermediate_size: int | None,
    sliced_labels: Sequence[int] | None,
    dense_observable_wires: int,
    max_intermediate_bytes: int | None = None,
) -> TensorNetworkState:
    """Build and return one local tensor-network numerical state.

    ``max_intermediate_bytes`` is a hard peak budget for every contraction this
    state performs, in bytes, and ``None`` applies no budget. Only slicing can
    lower a peak, so the state resolves the budget against the requested order
    when it contracts rather than here; see
    :meth:`TensorNetworkState.state`. A budget no slicing can satisfy raises from
    the slicing planner instead of being overrun.
    """

    if max_intermediate_bytes is not None and max_intermediate_bytes < 1:
        raise ValueError("max_intermediate_bytes must be a positive byte count")
    plan = build_local_tensor_network(
        circuit_or_ir, bsz=bsz, device=device, dtype=dtype
    )
    return TensorNetworkState(
        plan,
        contraction_strategy=contraction_strategy,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=sliced_labels,
        dense_observable_wires=dense_observable_wires,
        max_intermediate_bytes=max_intermediate_bytes,
    )
