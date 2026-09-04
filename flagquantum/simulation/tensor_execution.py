"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch

from ..circuit import Circuit
from ..ops.matrices import GATE_MAT_DICT

_PAULI_MPO_CORE_CACHE: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, ...]] = (
    OrderedDict()
)
_PAULI_MPO_CORE_CACHE_LIMIT = 32


from .tensor_contraction import (  # noqa: E402
    _build_slicing_plan,
    _clone_nodes_with_offset,
    _contract_nodes_greedy,
    _contract_nodes_sliced,
)
from .tensor_local import (  # noqa: E402
    build_local_tensor_network,
    run_local_tensor_network,
)
from .tensor_models import (  # noqa: E402
    ContractionPathStep,
    TensorNetworkContractionPlan,
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


def _compress_pauli_sum_mpo(
    coefficients: torch.Tensor,
    local_products: Sequence[Sequence[torch.Tensor]],
    *,
    relative_tolerance: float = 1e-14,
) -> tuple[torch.Tensor, ...]:
    """Exactly compress a direct-sum Pauli MPO without dense materialization."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if coefficients.is_cuda and world_size > local_world_size:
        # This is a one-time compression of a static observable, not a TN
        # contraction kernel.  Direct-sum Pauli MPOs have highly degenerate
        # spectra for which CUDA's default Jacobi SVD can enter a very slow
        # fallback.  LAPACK is robust and gives every distributed rank the
        # same compressed topology; move only the small resulting cores back.
        device = coefficients.device
        cpu_cores = _compress_pauli_sum_mpo(
            coefficients.to("cpu"),
            tuple(
                tuple(matrix.to("cpu") for matrix in product)
                for product in local_products
            ),
            relative_tolerance=relative_tolerance,
        )
        return tuple(core.to(device) for core in cpu_cores)

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
    return tuple(core.reshape(core.shape[0], core.shape[1], 2, 2) for core in cores)


def _cached_pauli_sum_mpo(
    coefficients: torch.Tensor,
    local_products: Sequence[Sequence[torch.Tensor]],
    *,
    signature: tuple[Any, ...],
) -> tuple[torch.Tensor, ...]:
    """Reuse static observable compression within a rank.

    Circuit parameters live outside the Hamiltonian MPO, so these cores are
    immutable across initial, optimizer, and final-energy plans.
    """

    key = (
        signature,
        str(coefficients.device),
        coefficients.dtype,
        tuple(tuple(matrix.dtype for matrix in product) for product in local_products),
    )
    cached = _PAULI_MPO_CORE_CACHE.get(key)
    if cached is not None:
        _PAULI_MPO_CORE_CACHE.move_to_end(key)
        return cached
    cores = _compress_pauli_sum_mpo(coefficients, local_products)
    _PAULI_MPO_CORE_CACHE[key] = cores
    _PAULI_MPO_CORE_CACHE.move_to_end(key)
    while len(_PAULI_MPO_CORE_CACHE) > _PAULI_MPO_CORE_CACHE_LIMIT:
        _PAULI_MPO_CORE_CACHE.popitem(last=False)
    return cores


def build_tensor_network(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> TensorNetworkContractionPlan:
    """Build a general tensor-network contraction plan from a circuit or IR."""

    return build_local_tensor_network(
        circuit_or_ir, bsz=bsz, device=device, dtype=dtype
    )


def _ensure_tensor_network_plan(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
) -> TensorNetworkContractionPlan:
    if isinstance(plan_or_circuit, TensorNetworkContractionPlan):
        return plan_or_circuit
    if isinstance(plan_or_circuit, Circuit):
        return build_tensor_network(
            plan_or_circuit,
            bsz=plan_or_circuit.bsz,
            device=plan_or_circuit.device,
            dtype=plan_or_circuit.dtype,
        )
    return build_tensor_network(plan_or_circuit)


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
        ket_plan = _ensure_tensor_network_plan(plan_or_circuit)

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


def build_tensor_network_hamiltonian_expectation(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    hamiltonian: Any,
) -> TensorNetworkExpectationPlan:
    """Build one exact MPO tensor network for a sum of Pauli products."""

    ket_plan = (
        plan_or_circuit
        if isinstance(plan_or_circuit, TensorNetworkContractionPlan)
        else _ensure_tensor_network_plan(plan_or_circuit)
    )
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
    bra_batch_label = bra_outputs[0]
    nodes: list[TensorNetworkNode] = list(ket_nodes) + list(bra_nodes)
    reference = ket_plan.nodes[0].tensor
    nodes.append(
        TensorNetworkNode(
            tensor=_identity_size_like(reference, ket_plan.bsz),
            labels=(batch_label, bra_batch_label),
            name="batch_identity",
        )
    )
    identity = _identity_matrix_like(reference)
    pauli_matrices = {
        "x": GATE_MAT_DICT["x"].to(reference),
        "y": GATE_MAT_DICT["y"].to(reference),
        "z": GATE_MAT_DICT["z"].to(reference),
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
        first = cores[0].squeeze(0)
        nodes.append(
            TensorNetworkNode(
                tensor=first,
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
        last = cores[-1].squeeze(1)
        nodes.append(
            TensorNetworkNode(
                tensor=last,
                labels=(bond_labels[-1], bra_outputs[-1], ket_outputs[-1]),
                name=f"hamiltonian_mpo_{ket_plan.n_wires - 1}",
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
        observable_wires=tuple(sorted(observable_wires)),
        path=path,
    )


def build_tensor_network_hamiltonian_expectations(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    hamiltonians: Sequence[Any],
) -> TensorNetworkExpectationPlan:
    """Build one exact block-MPO network for several Hamiltonian expectations."""

    observables = tuple(hamiltonians)
    if not observables:
        raise ValueError("tensor-network Hamiltonian batch requires observables")
    ket_plan = (
        plan_or_circuit
        if isinstance(plan_or_circuit, TensorNetworkContractionPlan)
        else _ensure_tensor_network_plan(plan_or_circuit)
    )
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
        output_labels=(batch_label, observable_label),
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


def _normalize_bitstring(
    bitstring: int | str | Sequence[int],
    n_wires: int,
) -> tuple[int, ...]:
    if isinstance(bitstring, int):
        if bitstring < 0 or bitstring >= 2**n_wires:
            raise ValueError("integer bitstring is outside the circuit state space")
        bits = tuple(int(bit) for bit in f"{bitstring:0{n_wires}b}")
    elif isinstance(bitstring, str):
        if len(bitstring) != n_wires or set(bitstring) - {"0", "1"}:
            raise ValueError("bitstring must contain exactly n_wires binary digits")
        bits = tuple(int(bit) for bit in bitstring)
    else:
        bits = tuple(int(bit) for bit in bitstring)
        if len(bits) != n_wires or any(bit not in {0, 1} for bit in bits):
            raise ValueError("bitstring must contain exactly n_wires binary values")
    return bits


def _amplitude_projection(
    plan: TensorNetworkContractionPlan,
    bitstring: int | str | Sequence[int],
) -> tuple[tuple[TensorNetworkNode, ...], tuple[int, ...]]:
    """Attach computational-basis projectors to a ket contraction plan."""

    bits = _normalize_bitstring(bitstring, plan.n_wires)
    reference = plan.nodes[0].tensor
    nodes = list(plan.nodes)
    for wire, (label, bit) in enumerate(zip(plan.output_labels[1:], bits)):
        projector = torch.zeros(2, dtype=reference.dtype, device=reference.device)
        projector[bit] = 1
        nodes.append(
            TensorNetworkNode(
                tensor=projector,
                labels=(label,),
                name=f"amplitude_projector_{wire}_{bit}",
                metadata={"wire": wire, "bit": bit},
            )
        )
    return tuple(nodes), (plan.output_labels[0],)


def _amplitude_batch_projection(
    plan: TensorNetworkContractionPlan,
    bitstrings: Sequence[int | str | Sequence[int]],
) -> tuple[tuple[TensorNetworkNode, ...], tuple[int, ...]]:
    """Attach a shared target axis for a batch of computational basis states."""

    bits = tuple(
        _normalize_bitstring(bitstring, plan.n_wires) for bitstring in bitstrings
    )
    if not bits:
        raise ValueError("bitstrings must contain at least one target")
    reference = plan.nodes[0].tensor
    target_label = (
        max((label for node in plan.nodes for label in node.labels), default=0) + 1
    )
    nodes = list(plan.nodes)
    for wire, label in enumerate(plan.output_labels[1:]):
        projector = torch.zeros(
            (len(bits), 2),
            dtype=reference.dtype,
            device=reference.device,
        )
        for target, target_bits in enumerate(bits):
            projector[target, target_bits[wire]] = 1
        nodes.append(
            TensorNetworkNode(
                tensor=projector,
                labels=(target_label, label),
                name=f"amplitude_batch_projector_{wire}",
                metadata={"wire": wire, "targets": len(bits)},
            )
        )
    return tuple(nodes), (plan.output_labels[0], target_label)


def tensor_network_amplitude(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    bitstring: int | str | Sequence[int],
    *,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    contraction_strategy: str = "quality_multistart",
    max_slices: int | None = 4096,
    max_recomputation_factor: float | None = 64.0,
) -> torch.Tensor:
    """Contract one computational-basis amplitude without materializing the state."""

    plan = _ensure_tensor_network_plan(plan_or_circuit)
    nodes, output_labels = _amplitude_projection(plan, bitstring)
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
    return result.reshape(plan.bsz)


def tensor_network_amplitudes(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    bitstrings: Sequence[int | str | Sequence[int]],
    *,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    contraction_strategy: str = "quality_multistart",
    max_slices: int | None = 4096,
    max_recomputation_factor: float | None = 64.0,
) -> torch.Tensor:
    """Contract a small amplitude batch through one shared tensor network."""

    plan = _ensure_tensor_network_plan(plan_or_circuit)
    nodes, output_labels = _amplitude_batch_projection(plan, bitstrings)
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
    return result.reshape(plan.bsz, len(bitstrings))


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
            if wire in axes["x"]:
                name = "x"
            elif wire in axes["y"]:
                name = "y"
            elif wire in axes["z"]:
                name = "z"
            else:
                name = None
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

    plan = _ensure_tensor_network_plan(plan_or_circuit)
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
    return run_local_tensor_network(
        circuit_or_ir,
        bsz=bsz,
        device=device,
        dtype=dtype,
        contraction_strategy=contraction_strategy,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=sliced_labels,
        dense_observable_wires=dense_observable_wires,
    )
