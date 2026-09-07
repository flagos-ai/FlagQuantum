"""Tensor-network observable plans and Pauli MPO construction."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch

from ..ops.matrices import GATE_MAT_DICT
from .tensor_local import ensure_local_tensor_network_plan
from .tensor_network.contraction import (
    _build_slicing_plan,
    _clone_nodes_with_offset,
    _contract_nodes_greedy,
    _contract_nodes_sliced,
)
from .tensor_network.models import (
    ContractionPathStep,
    TensorNetworkContractionPlan,
    TensorNetworkExpectationPlan,
    TensorNetworkNode,
)

_PAULI_MPO_CORE_CACHE: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, ...]] = (
    OrderedDict()
)
_PAULI_MPO_CORE_CACHE_LIMIT = 32


def _identity_matrix_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.eye(2, dtype=reference.dtype, device=reference.device)


def _identity_size_like(reference: torch.Tensor, size: int) -> torch.Tensor:
    return torch.eye(int(size), dtype=reference.dtype, device=reference.device)


def _compress_pauli_sum_mpo(
    coefficients: torch.Tensor,
    local_products: Sequence[Sequence[torch.Tensor]],
    *,
    relative_tolerance: float = 1e-14,
    compression_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ...]:
    """Exactly compress a direct-sum Pauli MPO on an explicit device."""

    output_device = coefficients.device
    if compression_device is not None:
        compression_device = torch.device(compression_device)
        coefficients = coefficients.to(compression_device)
        local_products = tuple(
            tuple(matrix.to(compression_device) for matrix in product)
            for product in local_products
        )

    term_count = len(local_products)
    n_wires = len(local_products[0])
    physical_dim = 4
    local_by_wire = tuple(
        torch.stack(tuple(product[wire] for product in local_products))
        for wire in range(n_wires)
    )
    cores = [
        (coefficients[:, None, None] * local_by_wire[0]).reshape(
            1, term_count, physical_dim
        )
    ]
    diagonal = torch.eye(
        term_count, dtype=coefficients.dtype, device=coefficients.device
    )
    for wire in range(1, n_wires - 1):
        core = diagonal[:, :, None, None] * local_by_wire[wire][:, None, :, :]
        cores.append(core.reshape(term_count, term_count, physical_dim))
    cores.append(local_by_wire[-1].reshape(term_count, 1, physical_dim))

    for wire in range(n_wires - 1):
        left_rank, right_rank, _ = cores[wire].shape
        matrix = (
            cores[wire].permute(0, 2, 1).reshape(left_rank * physical_dim, right_rank)
        )
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        threshold = relative_tolerance * singular[0]
        rank = max(1, int(torch.count_nonzero(singular > threshold)))
        left = left[:, :rank]
        transfer = singular[:rank, None] * right[:rank, :]
        cores[wire] = left.reshape(left_rank, physical_dim, rank).permute(0, 2, 1)
        next_right = cores[wire + 1].shape[1]
        cores[wire + 1] = torch.matmul(
            transfer, cores[wire + 1].reshape(right_rank, -1)
        ).reshape(rank, next_right, physical_dim)
    return tuple(
        core.reshape(core.shape[0], core.shape[1], 2, 2).to(output_device)
        for core in cores
    )


def _cached_pauli_sum_mpo(
    coefficients: torch.Tensor,
    local_products: Sequence[Sequence[torch.Tensor]],
    *,
    signature: tuple[Any, ...],
    compression_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reuse static observable compression within one process."""

    key = (
        signature,
        str(coefficients.device),
        coefficients.dtype,
        None if compression_device is None else str(torch.device(compression_device)),
        tuple(tuple(matrix.dtype for matrix in product) for product in local_products),
    )
    cached = _PAULI_MPO_CORE_CACHE.get(key)
    if cached is not None:
        _PAULI_MPO_CORE_CACHE.move_to_end(key)
        return cached
    cores = _compress_pauli_sum_mpo(
        coefficients,
        local_products,
        compression_device=compression_device,
    )
    _PAULI_MPO_CORE_CACHE[key] = cores
    _PAULI_MPO_CORE_CACHE.move_to_end(key)
    while len(_PAULI_MPO_CORE_CACHE) > _PAULI_MPO_CORE_CACHE_LIMIT:
        _PAULI_MPO_CORE_CACHE.popitem(last=False)
    return cores


def _path(nodes: Sequence[TensorNetworkNode]) -> tuple[ContractionPathStep, ...]:
    return tuple(
        ContractionPathStep(
            node_index=index,
            name=node.name,
            labels=node.labels,
            shape=tuple(node.tensor.shape),
        )
        for index, node in enumerate(nodes)
    )


def build_tensor_network_expectation(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    *,
    z: Sequence[int] | None = None,
    x: Sequence[int] | None = None,
    y: Sequence[int] | None = None,
) -> TensorNetworkExpectationPlan:
    """Build a direct tensor-network expectation plan for a Pauli product."""

    ket_plan = ensure_local_tensor_network_plan(plan_or_circuit)
    x_set, y_set, z_set = set(x or ()), set(y or ()), set(z or ())
    if (x_set & y_set) or (x_set & z_set) or (y_set & z_set):
        raise ValueError("A wire can appear in only one of x, y, or z.")

    max_label = max(
        (label for node in ket_plan.nodes for label in node.labels), default=0
    )
    offset = max_label + 1
    ket_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=0, conjugate=False)
    bra_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=offset, conjugate=True)
    ket_outputs = ket_plan.output_labels
    bra_outputs = tuple(label + offset for label in ket_outputs)
    batch_label = ket_outputs[0]
    nodes: list[TensorNetworkNode] = list(ket_nodes) + list(bra_nodes)
    reference = (
        ket_plan.nodes[0].tensor
        if ket_plan.nodes
        else torch.empty((), dtype=torch.complex64)
    )
    nodes.append(
        TensorNetworkNode(
            tensor=_identity_size_like(reference, ket_plan.bsz),
            labels=(batch_label, bra_outputs[0]),
            name="batch_identity",
        )
    )
    observable_wires = tuple(sorted(x_set | y_set | z_set))
    for wire in range(ket_plan.n_wires):
        name = (
            "x"
            if wire in x_set
            else "y" if wire in y_set else "z" if wire in z_set else None
        )
        matrix = (
            _identity_matrix_like(reference)
            if name is None
            else GATE_MAT_DICT[name].to(device=reference.device, dtype=reference.dtype)
        )
        nodes.append(
            TensorNetworkNode(
                tensor=matrix,
                labels=(bra_outputs[wire + 1], ket_outputs[wire + 1]),
                name=f"obs_{wire}",
                metadata={"wire": wire},
            )
        )
    return TensorNetworkExpectationPlan(
        n_wires=ket_plan.n_wires,
        bsz=ket_plan.bsz,
        nodes=tuple(nodes),
        output_labels=(batch_label,),
        observable_wires=observable_wires,
        path=_path(nodes),
    )


def build_tensor_network_hamiltonian_expectation(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    hamiltonian: Any,
    *,
    compression_device: torch.device | str | None = None,
) -> TensorNetworkExpectationPlan:
    """Build one exact MPO tensor network for a sum of Pauli products."""

    ket_plan = ensure_local_tensor_network_plan(plan_or_circuit)
    terms = tuple(getattr(hamiltonian, "terms", ()))
    if not terms:
        raise ValueError("tensor-network Hamiltonian expectation requires terms")
    max_label = max(
        (label for node in ket_plan.nodes for label in node.labels), default=0
    )
    offset = max_label + 1
    ket_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=0, conjugate=False)
    bra_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=offset, conjugate=True)
    ket_outputs = ket_plan.output_labels
    bra_outputs = tuple(label + offset for label in ket_outputs)
    batch_label = ket_outputs[0]
    nodes: list[TensorNetworkNode] = list(ket_nodes) + list(bra_nodes)
    reference = ket_plan.nodes[0].tensor
    nodes.append(
        TensorNetworkNode(
            tensor=_identity_size_like(reference, ket_plan.bsz),
            labels=(batch_label, bra_outputs[0]),
            name="batch_identity",
        )
    )
    identity = _identity_matrix_like(reference)
    pauli_matrices = {
        name: GATE_MAT_DICT[name].to(reference) for name in ("x", "y", "z")
    }
    local_products = []
    observable_wires = set()
    for term in terms:
        by_wire = {int(wire): str(name).lower() for wire, name in term.ops}
        observable_wires.update(by_wire)
        local_products.append(
            tuple(
                pauli_matrices.get(by_wire.get(wire), identity)
                for wire in range(ket_plan.n_wires)
            )
        )
    coefficients = tuple(
        torch.as_tensor(
            term.coefficient, dtype=reference.dtype, device=reference.device
        )
        for term in terms
    )
    if ket_plan.n_wires == 1:
        matrix = torch.stack(
            tuple(
                coefficient * product[0]
                for coefficient, product in zip(coefficients, local_products)
            )
        ).sum(dim=0)
        nodes.append(
            TensorNetworkNode(
                tensor=matrix,
                labels=(bra_outputs[1], ket_outputs[1]),
                name="hamiltonian_mpo_0",
            )
        )
    else:
        next_label = max((*bra_outputs, *ket_outputs)) + 1
        bond_labels = tuple(range(next_label, next_label + ket_plan.n_wires - 1))
        cores = _cached_pauli_sum_mpo(
            torch.stack(coefficients),
            local_products,
            compression_device=compression_device,
            signature=(
                ket_plan.n_wires,
                tuple(
                    (
                        complex(term.coefficient),
                        tuple(
                            (int(wire), str(name).lower()) for wire, name in term.ops
                        ),
                    )
                    for term in terms
                ),
            ),
        )
        nodes.append(
            TensorNetworkNode(
                tensor=cores[0].squeeze(0),
                labels=(bond_labels[0], bra_outputs[1], ket_outputs[1]),
                name="hamiltonian_mpo_0",
            )
        )
        for wire in range(1, ket_plan.n_wires - 1):
            nodes.append(
                TensorNetworkNode(
                    tensor=cores[wire],
                    labels=(
                        bond_labels[wire - 1],
                        bond_labels[wire],
                        bra_outputs[wire + 1],
                        ket_outputs[wire + 1],
                    ),
                    name=f"hamiltonian_mpo_{wire}",
                )
            )
        nodes.append(
            TensorNetworkNode(
                tensor=cores[-1].squeeze(1),
                labels=(bond_labels[-1], bra_outputs[-1], ket_outputs[-1]),
                name=f"hamiltonian_mpo_{ket_plan.n_wires - 1}",
            )
        )
    return TensorNetworkExpectationPlan(
        n_wires=ket_plan.n_wires,
        bsz=ket_plan.bsz,
        nodes=tuple(nodes),
        output_labels=(batch_label,),
        observable_wires=tuple(sorted(observable_wires)),
        path=_path(nodes),
    )


def build_tensor_network_hamiltonian_expectations(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    hamiltonians: Sequence[Any],
) -> TensorNetworkExpectationPlan:
    """Build one exact block-MPO network for several Hamiltonian expectations."""

    observables = tuple(hamiltonians)
    if not observables:
        raise ValueError("tensor-network Hamiltonian batch requires observables")
    ket_plan = ensure_local_tensor_network_plan(plan_or_circuit)
    term_keys: list[tuple[tuple[int, str], ...]] = []
    term_index: dict[tuple[tuple[int, str], ...], int] = {}
    for observable in observables:
        for term in tuple(getattr(observable, "terms", ())):
            key = tuple((int(wire), str(name).lower()) for wire, name in term.ops)
            if key not in term_index:
                term_index[key] = len(term_keys)
                term_keys.append(key)
    if not term_keys:
        raise ValueError("tensor-network Hamiltonian batch requires terms")

    max_label = max(
        (label for node in ket_plan.nodes for label in node.labels), default=0
    )
    offset = max_label + 1
    ket_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=0, conjugate=False)
    bra_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=offset, conjugate=True)
    ket_outputs = ket_plan.output_labels
    bra_outputs = tuple(label + offset for label in ket_outputs)
    batch_label = ket_outputs[0]
    nodes: list[TensorNetworkNode] = list(ket_nodes) + list(bra_nodes)
    reference = ket_plan.nodes[0].tensor
    nodes.append(
        TensorNetworkNode(
            tensor=_identity_size_like(reference, ket_plan.bsz),
            labels=(batch_label, bra_outputs[0]),
            name="batch_identity",
        )
    )
    coefficients = torch.zeros(
        len(observables), len(term_keys), dtype=reference.dtype, device=reference.device
    )
    for observable_index, observable in enumerate(observables):
        for term in tuple(getattr(observable, "terms", ())):
            key = tuple((int(wire), str(name).lower()) for wire, name in term.ops)
            coefficients[observable_index, term_index[key]] += torch.as_tensor(
                term.coefficient, dtype=reference.dtype, device=reference.device
            )

    identity = _identity_matrix_like(reference)
    matrices = {name: GATE_MAT_DICT[name].to(reference) for name in ("x", "y", "z")}
    local_products = tuple(
        tuple(
            matrices.get(dict(key).get(wire), identity)
            for wire in range(ket_plan.n_wires)
        )
        for key in term_keys
    )
    observable_wires = tuple(sorted({wire for key in term_keys for wire, _ in key}))
    observable_label = max((*bra_outputs, *ket_outputs)) + 1
    if ket_plan.n_wires == 1:
        local = torch.stack(tuple(product[0] for product in local_products))
        matrix = torch.einsum("ot,tij->oij", coefficients, local)
        nodes.append(
            TensorNetworkNode(
                tensor=matrix,
                labels=(observable_label, bra_outputs[1], ket_outputs[1]),
                name="hamiltonian_block_mpo_0",
            )
        )
    else:
        bond_labels = tuple(
            range(observable_label + 1, observable_label + ket_plan.n_wires)
        )
        local = torch.stack(tuple(product[0] for product in local_products))
        first = coefficients[:, :, None, None] * local[None, :, :, :]
        nodes.append(
            TensorNetworkNode(
                tensor=first,
                labels=(
                    observable_label,
                    bond_labels[0],
                    bra_outputs[1],
                    ket_outputs[1],
                ),
                name="hamiltonian_block_mpo_0",
            )
        )
        diagonal = torch.eye(
            len(term_keys), dtype=reference.dtype, device=reference.device
        )
        for wire in range(1, ket_plan.n_wires - 1):
            local = torch.stack(tuple(product[wire] for product in local_products))
            core = diagonal[:, :, None, None] * local[:, None, :, :]
            nodes.append(
                TensorNetworkNode(
                    tensor=core,
                    labels=(
                        bond_labels[wire - 1],
                        bond_labels[wire],
                        bra_outputs[wire + 1],
                        ket_outputs[wire + 1],
                    ),
                    name=f"hamiltonian_block_mpo_{wire}",
                )
            )
        last = torch.stack(tuple(product[-1] for product in local_products))
        nodes.append(
            TensorNetworkNode(
                tensor=last,
                labels=(bond_labels[-1], bra_outputs[-1], ket_outputs[-1]),
                name=f"hamiltonian_block_mpo_{ket_plan.n_wires - 1}",
            )
        )
    return TensorNetworkExpectationPlan(
        n_wires=ket_plan.n_wires,
        bsz=ket_plan.bsz,
        nodes=tuple(nodes),
        output_labels=(batch_label, observable_label),
        observable_wires=observable_wires,
        path=_path(nodes),
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


def _expectation_batch_projection(
    plan: TensorNetworkContractionPlan,
    observables: Sequence[Mapping[str, Sequence[int]]],
) -> tuple[tuple[TensorNetworkNode, ...], tuple[int, ...]]:
    """Build one bra-operator-ket network with a shared observable axis."""

    if not observables:
        raise ValueError("observables must contain at least one Pauli product")
    normalized = []
    for observable in observables:
        unknown = set(observable) - {"x", "y", "z"}
        if unknown:
            raise ValueError(f"unsupported Pauli axes: {sorted(unknown)}")
        axes = {
            axis: set(int(wire) for wire in observable.get(axis, ()))
            for axis in ("x", "y", "z")
        }
        if (
            (axes["x"] & axes["y"])
            or (axes["x"] & axes["z"])
            or (axes["y"] & axes["z"])
        ):
            raise ValueError("a wire can appear in only one Pauli axis")
        if any(
            wire < 0 or wire >= plan.n_wires
            for wires in axes.values()
            for wire in wires
        ):
            raise ValueError("observable wire is outside the circuit")
        normalized.append(axes)

    max_label = max((label for node in plan.nodes for label in node.labels), default=0)
    offset = max_label + 1
    ket_nodes = _clone_nodes_with_offset(plan.nodes, offset=0, conjugate=False)
    bra_nodes = _clone_nodes_with_offset(plan.nodes, offset=offset, conjugate=True)
    ket_outputs = plan.output_labels
    bra_outputs = tuple(label + offset for label in plan.output_labels)
    target_label = max(bra_outputs) + 1
    reference = plan.nodes[0].tensor
    nodes = list(ket_nodes) + list(bra_nodes)
    nodes.append(
        TensorNetworkNode(
            tensor=_identity_size_like(reference, plan.bsz),
            labels=(ket_outputs[0], bra_outputs[0]),
            name="batch_identity",
        )
    )
    for wire in range(plan.n_wires):
        operators = []
        for axes in normalized:
            name = (
                "x"
                if wire in axes["x"]
                else "y" if wire in axes["y"] else "z" if wire in axes["z"] else None
            )
            operators.append(
                _identity_matrix_like(reference)
                if name is None
                else GATE_MAT_DICT[name].to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
            )
        nodes.append(
            TensorNetworkNode(
                tensor=torch.stack(operators),
                labels=(target_label, bra_outputs[wire + 1], ket_outputs[wire + 1]),
                name=f"observable_batch_{wire}",
                metadata={"wire": wire, "targets": len(normalized)},
            )
        )
    return tuple(nodes), (ket_outputs[0], target_label)


def tensor_network_expectations(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    observables: Sequence[Mapping[str, Sequence[int]]],
    *,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    contraction_strategy: str = "quality_multistart",
    max_slices: int | None = 4096,
    max_recomputation_factor: float | None = 64.0,
) -> torch.Tensor:
    """Contract several Pauli products through one shared bra-ket network."""

    plan = ensure_local_tensor_network_plan(plan_or_circuit)
    nodes, output_labels = _expectation_batch_projection(plan, observables)
    if (
        max_intermediate_size is not None
        or max_intermediate_bytes is not None
        or sliced_labels is not None
    ):
        slicing = _build_slicing_plan(
            nodes,
            output_labels,
            max_intermediate_size=max_intermediate_size,
            max_intermediate_bytes=max_intermediate_bytes,
            sliced_labels=sliced_labels,
            contraction_strategy=contraction_strategy,
        )
        slicing.validate_economics(
            max_slices=max_slices,
            max_recomputation_factor=max_recomputation_factor,
        )
        result, _ = _contract_nodes_sliced(
            nodes,
            output_labels,
            max_intermediate_size=max_intermediate_size,
            max_intermediate_bytes=max_intermediate_bytes,
            sliced_labels=slicing.sliced_labels,
            contraction_strategy=contraction_strategy,
        )
    else:
        result, _ = _contract_nodes_greedy(
            nodes,
            output_labels,
            objective="quality",
        )
    return torch.real(result.reshape(plan.bsz, len(observables)))
