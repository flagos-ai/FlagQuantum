"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ..circuit import Circuit, _gate_matrix
from ..ops.matrices import GATE_MAT_DICT

_CONTRACTION_PROFILE_CACHE: dict[tuple[Any, ...], TensorNetworkContractionProfile] = {}
_CONTRACTION_PATH_CACHE: dict[
    tuple[Any, ...], tuple[tuple[int, int, tuple[int, ...]], ...]
] = {}
_CONTRACTION_STAGE_CACHE: dict[tuple[Any, ...], CompiledTNStagePlan] = {}
_DENSE_Z_OBSERVABLE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}
_Z_OBSERVABLE_NODE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], tuple[torch.Tensor, ...]
] = {}


from .tensor_contraction import (  # noqa: E402
    _as_ir,
    _clone_nodes_with_offset,
)
from .tensor_models import (  # noqa: E402
    CompiledTNProgram,
    CompiledTNStagePlan,
    ContractionPathStep,
    TensorNetworkContractionPlan,
    TensorNetworkContractionProfile,
    TensorNetworkExpectationPlan,
    TensorNetworkNode,
)
from .tensor_state import (  # noqa: E402
    TensorNetworkState,
)


def _identity_matrix_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.eye(2, dtype=reference.dtype, device=reference.device)


def _identity_size_like(reference: torch.Tensor, size: int) -> torch.Tensor:
    return torch.eye(int(size), dtype=reference.dtype, device=reference.device)


def _initial_state_tensor(
    circuit_or_ir: Any,
    *,
    bsz: int,
    n_wires: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(circuit_or_ir, Circuit):
        return circuit_or_ir.initial_state().to(device=device, dtype=dtype)
    state = torch.zeros(bsz, 2**n_wires, dtype=dtype, device=device)
    state[:, 0] = 1
    return state


def build_tensor_network(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> TensorNetworkContractionPlan:
    """Build a general tensor-network contraction plan from a circuit or IR."""

    ir = _as_ir(circuit_or_ir)
    dtype = dtype or torch.complex64
    initial = _initial_state_tensor(
        circuit_or_ir, bsz=bsz, n_wires=ir.n_wires, device=device, dtype=dtype
    )
    bsz = int(initial.shape[0])
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
    # Binding many tiny Python node objects costs more than rebuilding their
    # descriptors on CPU. The compiled template pays off together with fused
    # CUDA contraction kernels; retain eager construction for CPU workloads.
    active_structure = (
        compiled_structure if torch.device(device).type == "cuda" else None
    )

    reshaped_initial = initial.reshape((bsz,) + (2,) * ir.n_wires)
    for wire in range(ir.n_wires):
        label = next_label
        next_label += 1
        current_labels.append(label)
        selector = [slice(None)] + [0] * ir.n_wires
        selector[wire + 1] = slice(None)
        tensor = reshaped_initial[tuple(selector)]
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
        matrix = _gate_matrix(
            instruction,
            bsz=bsz,
            device=device,
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

    if active_structure is None:
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
    else:
        return active_structure.bind(dynamic_tensors, program_cache=program_cache)


def build_tensor_network_expectation(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    *,
    z: Sequence[int] | None = None,
    x: Sequence[int] | None = None,
    y: Sequence[int] | None = None,
) -> TensorNetworkExpectationPlan:
    """Build a direct tensor-network expectation plan for a Pauli product."""

    if isinstance(plan_or_circuit, TensorNetworkContractionPlan):
        ket_plan = plan_or_circuit
    else:
        ket_plan = build_tensor_network(plan_or_circuit)

    x_set = set(x or ())
    y_set = set(y or ())
    z_set = set(z or ())
    if (x_set & y_set) or (x_set & z_set) or (y_set & z_set):
        raise ValueError("A wire can appear in only one of x, y, or z.")

    max_label = max(
        (label for node in ket_plan.nodes for label in node.labels), default=0
    )
    offset = max_label + 1
    ket_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=0, conjugate=False)
    bra_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=offset, conjugate=True)
    ket_outputs = ket_plan.output_labels
    bra_outputs = tuple(label + offset for label in ket_plan.output_labels)
    batch_label = ket_outputs[0]
    bra_batch_label = bra_outputs[0]
    nodes: list[TensorNetworkNode] = list(ket_nodes) + list(bra_nodes)
    reference = (
        ket_plan.nodes[0].tensor
        if ket_plan.nodes
        else torch.empty((), dtype=torch.complex64)
    )

    nodes.append(
        TensorNetworkNode(
            tensor=_identity_size_like(reference, ket_plan.bsz),
            labels=(batch_label, bra_batch_label),
            name="batch_identity",
        )
    )
    observable_wires = tuple(sorted(x_set | y_set | z_set))
    for wire in range(ket_plan.n_wires):
        if wire in x_set:
            matrix = GATE_MAT_DICT["x"].to(
                device=reference.device, dtype=reference.dtype
            )
        elif wire in y_set:
            matrix = GATE_MAT_DICT["y"].to(
                device=reference.device, dtype=reference.dtype
            )
        elif wire in z_set:
            matrix = GATE_MAT_DICT["z"].to(
                device=reference.device, dtype=reference.dtype
            )
        else:
            matrix = _identity_matrix_like(reference)
        nodes.append(
            TensorNetworkNode(
                tensor=matrix,
                labels=(bra_outputs[wire + 1], ket_outputs[wire + 1]),
                name=f"obs_{wire}",
                metadata={"wire": wire},
            )
        )

    path = tuple(
        ContractionPathStep(
            node_index=index,
            name=node.name,
            labels=node.labels,
            shape=tuple(node.tensor.shape),
        )
        for index, node in enumerate(nodes)
    )
    return TensorNetworkExpectationPlan(
        n_wires=ket_plan.n_wires,
        bsz=ket_plan.bsz,
        nodes=tuple(nodes),
        output_labels=(batch_label,),
        observable_wires=observable_wires,
        path=path,
    )


def tensor_network_expectation_ps(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    *,
    z: Sequence[int] | None = None,
    x: Sequence[int] | None = None,
    y: Sequence[int] | None = None,
) -> torch.Tensor:
    """Directly contract a Pauli-product expectation from a tensor network."""

    plan = build_tensor_network_expectation(plan_or_circuit, x=x, y=y, z=z)
    return torch.real(plan.contract())


def run_tensor_network(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    contraction_strategy: str = "greedy",
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    dense_observable_wires: int = 0,
) -> TensorNetworkState:
    """Run a circuit through the general tensor-network contraction engine."""

    if isinstance(circuit_or_ir, Circuit):
        bsz = circuit_or_ir.bsz
        dtype = circuit_or_ir.dtype
        if device is None:
            device = circuit_or_ir.device
    if device is None:
        device = "cpu"
    plan = build_tensor_network(circuit_or_ir, bsz=bsz, device=device, dtype=dtype)
    return TensorNetworkState(
        plan,
        contraction_strategy=contraction_strategy,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=sliced_labels,
        dense_observable_wires=dense_observable_wires,
    )
