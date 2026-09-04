"""Local tensor-network plan construction and numerical execution."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ..circuit import Circuit
from ..ops.gate_matrix import gate_matrix
from .tensor_contraction import _as_ir
from .tensor_models import (
    CompiledTNProgram,
    ContractionPathStep,
    TensorNetworkContractionPlan,
    TensorNetworkNode,
)
from .tensor_state import TensorNetworkState


def _initial_wire_tensors(
    circuit_or_ir: Any,
    *,
    bsz: int,
    n_wires: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Represent the default |0...0> state without a dense allocation."""

    if isinstance(circuit_or_ir, Circuit) and circuit_or_ir._inputs is not None:
        state = circuit_or_ir.initial_state().to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        reshaped = state.reshape((state.shape[0],) + (2,) * n_wires)
        tensors = []
        for wire in range(n_wires):
            selector = [slice(None)] + [0] * n_wires
            selector[wire + 1] = slice(None)
            tensors.append(reshaped[tuple(selector)])
        return tuple(tensors)
    zero = torch.zeros(int(bsz), 2, dtype=dtype, device=device)
    zero[:, 0] = 1
    return tuple(zero for _ in range(n_wires))


def build_local_tensor_network(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> TensorNetworkContractionPlan:
    """Build a local tensor-network contraction plan from a circuit or IR."""

    ir = _as_ir(circuit_or_ir)
    dtype = dtype or torch.complex64
    initial_wires = _initial_wire_tensors(
        circuit_or_ir, bsz=bsz, n_wires=ir.n_wires, device=device, dtype=dtype
    )
    bsz = int(initial_wires[0].shape[0]) if initial_wires else int(bsz)
    batch_label = 0
    next_label = 1
    current_labels = []
    nodes: list[TensorNetworkNode] = []
    dynamic_tensors: list[torch.Tensor] = []
    binding_source = getattr(circuit_or_ir, "_parameter_bindings", None)
    parameter_bindings = None if binding_source is None else binding_source.values()
    program_cache = getattr(circuit_or_ir, "_backend_programs", None)
    program_key = ("tensor_network", bsz)
    compiled_structure = (
        None if program_cache is None else program_cache.get(program_key)
    )
    # Binding many small node objects costs more than rebuilding descriptors on
    # CPU. Reuse the compiled template only with the fused CUDA path.
    active_structure = (
        compiled_structure if torch.device(device).type == "cuda" else None
    )

    for wire in range(ir.n_wires):
        label = next_label
        next_label += 1
        current_labels.append(label)
        tensor = initial_wires[wire]
        dynamic_tensors.append(tensor)
        if active_structure is None:
            nodes.append(
                TensorNetworkNode(
                    tensor=tensor, labels=(batch_label, label), name=f"init_{wire}"
                )
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
        for wire, label in zip(instruction.wires, output_labels):
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
) -> TensorNetworkState:
    """Build and return one local tensor-network numerical state."""

    plan = build_local_tensor_network(
        circuit_or_ir, bsz=bsz, device=device, dtype=dtype
    )
    return TensorNetworkState(
        plan,
        contraction_strategy=contraction_strategy,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=sliced_labels,
        dense_observable_wires=dense_observable_wires,
    )
