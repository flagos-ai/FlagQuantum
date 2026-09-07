"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..circuit import Circuit
from .tensor_contraction import (  # noqa: E402
    _build_slicing_plan,
    _contract_nodes_greedy,
    _contract_nodes_sliced,
)
from .tensor_local import (  # noqa: E402
    build_local_tensor_network,
    ensure_local_tensor_network_plan,
    run_local_tensor_network,
)
from .tensor_network.models import (  # noqa: E402
    TensorNetworkContractionPlan,
    TensorNetworkExpectationPlan,
    TensorNetworkNode,
)
from .tensor_observables import (  # noqa: E402
    build_tensor_network_expectation as _build_tensor_network_expectation,
)
from .tensor_observables import (
    build_tensor_network_hamiltonian_expectation as _build_tensor_network_hamiltonian_expectation,
)
from .tensor_observables import (
    build_tensor_network_hamiltonian_expectations as _build_tensor_network_hamiltonian_expectations,
)
from .tensor_observables import (
    tensor_network_expectation_ps as _tensor_network_expectation_ps,
)
from .tensor_observables import (
    tensor_network_expectations as _tensor_network_expectations,
)
from .tensor_network.state import (  # noqa: E402
    TensorNetworkState,
)


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
    return ensure_local_tensor_network_plan(plan_or_circuit)


def build_tensor_network_expectation(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    *,
    z: Sequence[int] | None = None,
    x: Sequence[int] | None = None,
    y: Sequence[int] | None = None,
) -> TensorNetworkExpectationPlan:
    """Build a direct tensor-network expectation plan for a Pauli product."""

    return _build_tensor_network_expectation(plan_or_circuit, x=x, y=y, z=z)


def build_tensor_network_hamiltonian_expectation(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    hamiltonian: Any,
) -> TensorNetworkExpectationPlan:
    """Build one exact MPO tensor network for a sum of Pauli products."""

    return _build_tensor_network_hamiltonian_expectation(plan_or_circuit, hamiltonian)


def build_tensor_network_hamiltonian_expectations(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    hamiltonians: Sequence[Any],
) -> TensorNetworkExpectationPlan:
    """Build one exact block-MPO network for several Hamiltonian expectations."""

    return _build_tensor_network_hamiltonian_expectations(plan_or_circuit, hamiltonians)


def tensor_network_expectation_ps(
    plan_or_circuit: TensorNetworkContractionPlan | Any,
    *,
    z: Sequence[int] | None = None,
    x: Sequence[int] | None = None,
    y: Sequence[int] | None = None,
) -> torch.Tensor:
    """Directly contract a Pauli-product expectation from a tensor network."""

    return _tensor_network_expectation_ps(plan_or_circuit, x=x, y=y, z=z)


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

    return _tensor_network_expectations(
        plan_or_circuit,
        observables,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        sliced_labels=sliced_labels,
        contraction_strategy=contraction_strategy,
        max_slices=max_slices,
        max_recomputation_factor=max_recomputation_factor,
    )


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
