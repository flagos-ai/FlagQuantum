"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

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


@dataclass(frozen=True)
class TensorNode:
    tensor: torch.Tensor
    axes: tuple[str, ...]
    name: str = ""


@dataclass(frozen=True)
class EinsumProgram:
    equation: str
    tensors: tuple[torch.Tensor, ...]

    def run(self) -> torch.Tensor:
        return torch.einsum(self.equation, *self.tensors)


class ContractionGraph:
    def __init__(self, nodes: Sequence[TensorNode] = ()) -> None:
        self.nodes = list(nodes)

    def add(
        self, tensor: torch.Tensor, axes: Sequence[str], name: str = ""
    ) -> TensorNode:
        node = TensorNode(tensor=tensor, axes=tuple(axes), name=name)
        self.nodes.append(node)
        return node

    def to_einsum(self, output_axes: Sequence[str]) -> EinsumProgram:
        inputs = ["".join(node.axes) for node in self.nodes]
        equation = ",".join(inputs) + "->" + "".join(output_axes)
        return EinsumProgram(equation, tuple(node.tensor for node in self.nodes))


@dataclass(frozen=True)
class TensorNetworkNode:
    """Integer-labeled tensor network node used by the PyTorch sublist einsum."""

    tensor: torch.Tensor
    labels: tuple[int, ...]
    name: str = ""
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ContractionPathStep:
    """A lightweight contraction path record for inspection and future tuning."""

    node_index: int
    name: str
    labels: tuple[int, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class PairContractionStep:
    """A concrete pairwise tensor contraction step."""

    step: int
    left: str
    right: str
    left_labels: tuple[int, ...]
    right_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    output_shape: tuple[int, ...]
    estimated_cost: int
    intermediate_size: int


@dataclass(frozen=True)
class TensorNetworkContractionProfile:
    """Cost profile for a planned tensor-network contraction path."""

    strategy: str
    estimated_cost: int
    peak_size: int
    n_steps: int
    output_size: int
    n_slices: int = 1
    sliced_labels: tuple[int, ...] = ()
    total_intermediate_size: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "estimated_cost": self.estimated_cost,
            "peak_size": self.peak_size,
            "n_steps": self.n_steps,
            "output_size": self.output_size,
            "n_slices": self.n_slices,
            "sliced_labels": self.sliced_labels,
            "total_intermediate_size": self.total_intermediate_size,
        }


@dataclass(frozen=True)
class TensorNetworkSlicingPlan:
    """Internal-edge slicing plan for bounded-memory contraction."""

    sliced_labels: tuple[int, ...]
    slice_shape: tuple[int, ...]
    n_slices: int
    per_slice_cost: int
    total_estimated_cost: int
    peak_size: int
    target_peak_size: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "sliced_labels": self.sliced_labels,
            "slice_shape": self.slice_shape,
            "n_slices": self.n_slices,
            "per_slice_cost": self.per_slice_cost,
            "total_estimated_cost": self.total_estimated_cost,
            "peak_size": self.peak_size,
            "target_peak_size": self.target_peak_size,
        }


@dataclass(frozen=True)
class TensorNetworkContractionPlan:
    """Tensor network contraction plan generated from a circuit IR."""

    n_wires: int
    bsz: int
    nodes: tuple[TensorNetworkNode, ...]
    output_labels: tuple[int, ...]
    path: tuple[ContractionPathStep, ...]
    program_cache: dict[tuple[Any, ...], Any] | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len({label for node in self.nodes for label in node.labels})

    def greedy_path(self) -> tuple[PairContractionStep, ...]:
        """Return the native greedy pairwise contraction path."""

        _, steps = _contract_nodes_greedy(self.nodes, self.output_labels, dry_run=True)
        return steps

    def memory_greedy_path(self) -> tuple[PairContractionStep, ...]:
        """Return a memory-first greedy path that minimizes peak intermediates."""

        _, steps = _contract_nodes_greedy(
            self.nodes, self.output_labels, dry_run=True, objective="memory"
        )
        return steps

    def beam_path(self, *, beam_width: int = 8) -> tuple[PairContractionStep, ...]:
        """Return a beam-search contraction path."""

        _, steps = _contract_nodes_beam(
            self.nodes, self.output_labels, dry_run=True, beam_width=beam_width
        )
        return steps

    def optimal_path(self, *, max_nodes: int = 7) -> tuple[PairContractionStep, ...]:
        """Return an exact small-network path, falling back to beam for larger networks."""

        _, steps = _contract_nodes_optimal(
            self.nodes, self.output_labels, dry_run=True, max_nodes=max_nodes
        )
        return steps

    def contraction_profile(
        self,
        strategy: str = "greedy",
        *,
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
        beam_width: int = 8,
    ) -> TensorNetworkContractionProfile:
        """Return a detailed native contraction profile for planning and benchmarking."""

        return _contraction_profile(
            self.nodes,
            self.output_labels,
            strategy=strategy,
            max_intermediate_size=max_intermediate_size,
            sliced_labels=sliced_labels,
            beam_width=beam_width,
        )

    def contraction_cost(self, strategy: str = "greedy") -> dict[str, int]:
        """Estimate contraction cost and peak intermediate tensor size."""

        profile = self.contraction_profile(strategy)
        out = {
            "estimated_cost": profile.estimated_cost,
            "peak_size": profile.peak_size,
        }
        if profile.n_slices != 1:
            out["n_slices"] = profile.n_slices
        return out

    def slicing_plan(
        self,
        *,
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
    ) -> TensorNetworkSlicingPlan:
        """Plan internal-edge slicing for bounded-memory contraction."""

        return _build_slicing_plan(
            self.nodes,
            self.output_labels,
            max_intermediate_size=max_intermediate_size,
            sliced_labels=sliced_labels,
        )

    def contract(
        self,
        *,
        strategy: str = "greedy",
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
        beam_width: int = 8,
    ) -> torch.Tensor:
        if strategy in {"greedy", "memory_greedy"}:
            objective = "memory" if strategy == "memory_greedy" else "balanced"
            result, _ = _contract_nodes_greedy(
                self.nodes, self.output_labels, objective=objective
            )
            return result.reshape(self.bsz, 2**self.n_wires)
        if strategy == "beam":
            result, _ = _contract_nodes_beam(
                self.nodes, self.output_labels, beam_width=beam_width
            )
            return result.reshape(self.bsz, 2**self.n_wires)
        if strategy == "optimal":
            result, _ = _contract_nodes_optimal(self.nodes, self.output_labels)
            return result.reshape(self.bsz, 2**self.n_wires)
        if strategy in {"sliced", "auto_sliced", "beam_sliced"}:
            result, _ = _contract_nodes_sliced(
                self.nodes,
                self.output_labels,
                max_intermediate_size=max_intermediate_size,
                sliced_labels=sliced_labels,
                contraction_strategy="beam" if strategy == "beam_sliced" else "greedy",
                beam_width=beam_width,
            )
            return result.reshape(self.bsz, 2**self.n_wires)
        if strategy != "einsum":
            raise ValueError(
                "strategy must be 'greedy', 'memory_greedy', 'beam', 'optimal', 'einsum', 'sliced', 'auto_sliced', or 'beam_sliced'."
            )
        operands: list[Any] = []
        for node in self.nodes:
            operands.extend([node.tensor, list(node.labels)])
        operands.append(list(self.output_labels))
        result = torch.einsum(*operands)
        return result.reshape(self.bsz, 2**self.n_wires)

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "tensor_network",
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_wires": self.n_wires,
            "path_length": len(self.path),
            "greedy_cost": self.contraction_cost("greedy")["estimated_cost"],
            "greedy_peak_size": self.contraction_cost("greedy")["peak_size"],
            "memory_greedy_cost": self.contraction_cost("memory_greedy")[
                "estimated_cost"
            ],
            "memory_greedy_peak_size": self.contraction_cost("memory_greedy")[
                "peak_size"
            ],
            "beam_cost": self.contraction_profile("beam").estimated_cost,
            "beam_peak_size": self.contraction_profile("beam").peak_size,
        }


@dataclass(frozen=True)
class CompiledTNNode:
    labels: tuple[int, ...]
    name: str
    metadata: Mapping[str, Any] | None


@dataclass(frozen=True)
class CompiledTNContractionBucket:
    equation: str
    batched_equation: str
    operations: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class CompiledTNContractionStage:
    buckets: tuple[CompiledTNContractionBucket, ...]


@dataclass(frozen=True)
class CompiledTNStagePlan:
    stages: tuple[CompiledTNContractionStage, ...]
    output_labels: tuple[int, ...]


@dataclass(frozen=True)
class CompiledTNProgram:
    """Immutable TN topology with dynamic tensor binding slots."""

    n_wires: int
    bsz: int
    nodes: tuple[CompiledTNNode, ...]
    output_labels: tuple[int, ...]
    path: tuple[ContractionPathStep, ...]

    @classmethod
    def compile(cls, plan: TensorNetworkContractionPlan) -> "CompiledTNProgram":
        return cls(
            n_wires=plan.n_wires,
            bsz=plan.bsz,
            nodes=tuple(
                CompiledTNNode(node.labels, node.name, node.metadata)
                for node in plan.nodes
            ),
            output_labels=plan.output_labels,
            path=plan.path,
        )

    def bind(
        self,
        tensors: Sequence[torch.Tensor],
        *,
        program_cache: dict[tuple[Any, ...], Any] | None = None,
    ) -> TensorNetworkContractionPlan:
        if len(tensors) != len(self.nodes):
            raise RuntimeError("compiled TN tensor slot count changed")
        return TensorNetworkContractionPlan(
            n_wires=self.n_wires,
            bsz=self.bsz,
            nodes=tuple(
                TensorNetworkNode(tensor, node.labels, node.name, node.metadata)
                for tensor, node in zip(tensors, self.nodes)
            ),
            output_labels=self.output_labels,
            path=self.path,
            program_cache=program_cache,
        )


@dataclass(frozen=True)
class CompiledTNObservableProgram:
    """Shared bra-ket program for a batch of single-wire Z observables."""

    n_wires: int
    wires: tuple[int, ...]

    def bind(
        self, ket_plan: TensorNetworkContractionPlan
    ) -> TensorNetworkExpectationPlan:
        if ket_plan.n_wires != self.n_wires:
            raise RuntimeError("compiled TN observable qubit count changed")
        max_label = max(
            (label for node in ket_plan.nodes for label in node.labels), default=0
        )
        offset = max_label + 1
        observable_label = 2 * offset
        ket_nodes = _clone_nodes_with_offset(ket_plan.nodes, offset=0, conjugate=False)
        bra_nodes = _clone_nodes_with_offset(
            ket_plan.nodes, offset=offset, conjugate=True
        )
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
        key = (
            self.n_wires,
            self.wires,
            str(reference.device),
            reference.dtype,
        )
        matrices = _Z_OBSERVABLE_NODE_CACHE.get(key)
        if matrices is None:
            identity = _identity_matrix_like(reference)
            z_matrix = GATE_MAT_DICT["z"].to(
                device=reference.device, dtype=reference.dtype
            )
            built = []
            for wire in range(self.n_wires):
                batch = identity.expand(len(self.wires), 2, 2).clone()
                for observable, target in enumerate(self.wires):
                    if wire == target:
                        batch[observable] = z_matrix
                built.append(batch)
            matrices = tuple(built)
            _Z_OBSERVABLE_NODE_CACHE[key] = matrices
        for wire, matrix in enumerate(matrices):
            nodes.append(
                TensorNetworkNode(
                    tensor=matrix,
                    labels=(
                        observable_label,
                        bra_outputs[wire + 1],
                        ket_outputs[wire + 1],
                    ),
                    name=f"z_batch_{wire}",
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
            n_wires=self.n_wires,
            bsz=ket_plan.bsz,
            nodes=tuple(nodes),
            output_labels=(batch_label, observable_label),
            observable_wires=self.wires,
            path=path,
        )


@dataclass(frozen=True)
class TensorNetworkExpectationPlan:
    """Direct bra-operator-ket tensor-network expectation plan."""

    n_wires: int
    bsz: int
    nodes: tuple[TensorNetworkNode, ...]
    output_labels: tuple[int, ...]
    observable_wires: tuple[int, ...]
    path: tuple[ContractionPathStep, ...]

    def greedy_path(self) -> tuple[PairContractionStep, ...]:
        """Return the native greedy pairwise contraction path."""

        _, steps = _contract_nodes_greedy(self.nodes, self.output_labels, dry_run=True)
        return steps

    def memory_greedy_path(self) -> tuple[PairContractionStep, ...]:
        """Return a memory-first greedy path for direct expectation contraction."""

        _, steps = _contract_nodes_greedy(
            self.nodes, self.output_labels, dry_run=True, objective="memory"
        )
        return steps

    def beam_path(self, *, beam_width: int = 8) -> tuple[PairContractionStep, ...]:
        """Return a beam-search path for direct expectation contraction."""

        _, steps = _contract_nodes_beam(
            self.nodes, self.output_labels, dry_run=True, beam_width=beam_width
        )
        return steps

    def optimal_path(self, *, max_nodes: int = 7) -> tuple[PairContractionStep, ...]:
        """Return an exact small-network expectation path, falling back to beam."""

        _, steps = _contract_nodes_optimal(
            self.nodes, self.output_labels, dry_run=True, max_nodes=max_nodes
        )
        return steps

    def contraction_profile(
        self,
        strategy: str = "greedy",
        *,
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
        beam_width: int = 8,
    ) -> TensorNetworkContractionProfile:
        """Return a detailed native expectation contraction profile."""

        return _contraction_profile(
            self.nodes,
            self.output_labels,
            strategy=strategy,
            max_intermediate_size=max_intermediate_size,
            sliced_labels=sliced_labels,
            beam_width=beam_width,
        )

    def contraction_cost(self, strategy: str = "greedy") -> dict[str, int]:
        """Estimate expectation-network contraction cost."""

        profile = self.contraction_profile(strategy)
        out = {
            "estimated_cost": profile.estimated_cost,
            "peak_size": profile.peak_size,
        }
        if profile.n_slices != 1:
            out["n_slices"] = profile.n_slices
        return out

    def slicing_plan(
        self,
        *,
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
    ) -> TensorNetworkSlicingPlan:
        """Plan internal-edge slicing for direct expectation contraction."""

        return _build_slicing_plan(
            self.nodes,
            self.output_labels,
            max_intermediate_size=max_intermediate_size,
            sliced_labels=sliced_labels,
        )

    def contract(
        self,
        *,
        strategy: str = "greedy",
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
        beam_width: int = 8,
    ) -> torch.Tensor:
        if strategy in {"greedy", "memory_greedy"}:
            objective = "memory" if strategy == "memory_greedy" else "balanced"
            result, _ = _contract_nodes_greedy(
                self.nodes, self.output_labels, objective=objective
            )
            return result
        if strategy == "beam":
            result, _ = _contract_nodes_beam(
                self.nodes, self.output_labels, beam_width=beam_width
            )
            return result
        if strategy == "optimal":
            result, _ = _contract_nodes_optimal(self.nodes, self.output_labels)
            return result
        if strategy in {"sliced", "auto_sliced", "beam_sliced"}:
            result, _ = _contract_nodes_sliced(
                self.nodes,
                self.output_labels,
                max_intermediate_size=max_intermediate_size,
                sliced_labels=sliced_labels,
                contraction_strategy="beam" if strategy == "beam_sliced" else "greedy",
                beam_width=beam_width,
            )
            return result
        if strategy != "einsum":
            raise ValueError(
                "strategy must be 'greedy', 'memory_greedy', 'beam', 'optimal', 'einsum', 'sliced', 'auto_sliced', or 'beam_sliced'."
            )
        operands: list[Any] = []
        for node in self.nodes:
            operands.extend([node.tensor, list(node.labels)])
        operands.append(list(self.output_labels))
        return torch.einsum(*operands)

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "tensor_network",
            "contraction": "expectation",
            "n_nodes": len(self.nodes),
            "n_edges": len({label for node in self.nodes for label in node.labels}),
            "n_wires": self.n_wires,
            "observable_wires": self.observable_wires,
            "path_length": len(self.path),
            "greedy_cost": self.contraction_cost("greedy")["estimated_cost"],
            "greedy_peak_size": self.contraction_cost("greedy")["peak_size"],
            "memory_greedy_cost": self.contraction_cost("memory_greedy")[
                "estimated_cost"
            ],
            "memory_greedy_peak_size": self.contraction_cost("memory_greedy")[
                "peak_size"
            ],
            "beam_cost": self.contraction_profile("beam").estimated_cost,
            "beam_peak_size": self.contraction_profile("beam").peak_size,
        }


def _contraction_call(name: str, *args: Any, **kwargs: Any) -> Any:
    from . import tensor_contraction

    return getattr(tensor_contraction, name)(*args, **kwargs)


def _build_slicing_plan(*args: Any, **kwargs: Any) -> Any:
    return _contraction_call("_build_slicing_plan", *args, **kwargs)


def _clone_nodes_with_offset(*args: Any, **kwargs: Any) -> Any:
    return _contraction_call("_clone_nodes_with_offset", *args, **kwargs)


def _contraction_profile(*args: Any, **kwargs: Any) -> Any:
    return _contraction_call("_contraction_profile", *args, **kwargs)


def _contract_nodes_beam(*args: Any, **kwargs: Any) -> Any:
    return _contraction_call("_contract_nodes_beam", *args, **kwargs)


def _contract_nodes_greedy(*args: Any, **kwargs: Any) -> Any:
    return _contraction_call("_contract_nodes_greedy", *args, **kwargs)


def _contract_nodes_optimal(*args: Any, **kwargs: Any) -> Any:
    return _contraction_call("_contract_nodes_optimal", *args, **kwargs)


def _contract_nodes_sliced(*args: Any, **kwargs: Any) -> Any:
    return _contraction_call("_contract_nodes_sliced", *args, **kwargs)


def _identity_matrix_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.eye(2, dtype=reference.dtype, device=reference.device)


def _identity_size_like(reference: torch.Tensor, size: int) -> torch.Tensor:
    return torch.eye(int(size), dtype=reference.dtype, device=reference.device)
