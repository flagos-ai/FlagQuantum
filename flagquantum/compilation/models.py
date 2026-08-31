"""Typed analysis and execution-plan models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from ..core.contracts import RuntimePlanContract
from ..core.ir import Instruction
from .candidates import RuntimeCandidate

if TYPE_CHECKING:
    from .noise.planning import NoisyExecutionPlan
    from .performance_calibration import CalibratedPlanCost


@dataclass(frozen=True)
class LayerPlan:
    index: int
    instructions: tuple[Instruction, ...]
    wires: tuple[int, ...]


@dataclass(frozen=True)
class CircuitAnalysis:
    n_wires: int
    n_instructions: int
    depth: int
    gate_counts: dict[str, int]
    wire_usage: tuple[int, ...]
    max_gate_width: int
    two_qubit_gates: int
    multi_qubit_gates: int
    channel_count: int = 0
    has_noise: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    analysis: CircuitAnalysis
    layers: tuple[LayerPlan, ...]
    state_bytes: int
    recommended_mode: str
    world_size: int
    shardable_wires: tuple[int, ...]
    state_mode: str = "statevector"
    user_tier: str = "single_device"
    usability_contract: str = "single_api_fast_path"
    runtime_config: Mapping[str, Any] | None = None
    routing_plan: Mapping[str, Any] | None = None
    noisy_execution_plan: NoisyExecutionPlan | None = None
    _contract_payload_json: str | None = None

    @property
    def identity(self) -> str:
        return str(self.to_dict()["identity"])

    @property
    def schema_version(self) -> str:
        return str(self.to_dict()["version"])

    @property
    def program_fingerprint(self) -> str:
        return str(self.to_dict()["fingerprints"]["program"])

    @property
    def options_fingerprint(self) -> str:
        return str(self.to_dict()["fingerprints"]["options"])

    @property
    def environment_fingerprint(self) -> str:
        return str(self.to_dict()["fingerprints"]["environment"])

    @property
    def compiler_fingerprint(self) -> str:
        return str(self.to_dict()["fingerprints"]["compiler"])

    @property
    def mode(self) -> str:
        return str(self.to_dict()["decision"]["mode"])

    @property
    def backend(self) -> str:
        return str(self.to_dict()["decision"]["backend"])

    @property
    def device(self) -> str:
        return str(self.to_dict()["decision"]["device"])

    @property
    def target(self) -> str:
        return str(self.to_dict()["decision"]["target"])

    @property
    def batch_size(self) -> int:
        return int(self.to_dict()["decision"]["batch_size"])

    @property
    def precision(self) -> str:
        return str(self.to_dict()["decision"]["precision"])

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def to_dict(self) -> dict[str, object]:
        from .execution_plan_contract import plan_to_dict

        return plan_to_dict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        from .execution_plan_contract import plan_to_json

        return plan_to_json(self, indent=indent)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionPlan":
        from .execution_plan_contract import plan_from_dict

        return plan_from_dict(payload)

    @classmethod
    def from_json(cls, text: str) -> "ExecutionPlan":
        from .execution_plan_contract import plan_from_json

        return plan_from_json(text)

    def to_contract(self) -> RuntimePlanContract:
        from .contract_adapter import execution_plan_contract

        return execution_plan_contract(self)

    def calibrated_cost(self, artifact: Mapping[str, Any]) -> CalibratedPlanCost:
        from .performance_calibration import calibrate_plan_cost

        return calibrate_plan_cost(self, artifact)

    def summary(self) -> dict[str, Any]:
        is_distributed = self.world_size > 1
        distribution_semantics = (
            "sharded_across_ranks"
            if self.recommended_mode == "distributed_statevector" and is_distributed
            else (
                "single_device_fast_path"
                if not is_distributed
                else "requires_runtime_summary"
            )
        )
        sharding_available = distribution_semantics == "sharded_across_ranks"
        blockers: tuple[str, ...] = ()
        if is_distributed and not sharding_available:
            blockers = ("runtime_summary_required_for_scalability_claim",)
        summary = {
            "state_mode": self.state_mode,
            "recommended_mode": self.recommended_mode,
            "claim_evidence_type": "plan_preflight",
            "user_tier": self.user_tier,
            "usability_contract": self.usability_contract,
            "distribution_semantics": distribution_semantics,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sharding_plan_available": sharding_available,
            "scalability_blockers": blockers,
            "world_size": self.world_size,
            "state_bytes": self.state_bytes,
            "depth": self.analysis.depth,
            "n_wires": self.analysis.n_wires,
            "n_instructions": self.analysis.n_instructions,
            "two_qubit_gates": self.analysis.two_qubit_gates,
            "multi_qubit_gates": self.analysis.multi_qubit_gates,
            "shardable_wires": self.shardable_wires,
            "runtime_config": dict(self.runtime_config or {}),
            "routing_plan": dict(self.routing_plan or {}),
            "noisy_execution_plan": (
                None
                if self.noisy_execution_plan is None
                else self.noisy_execution_plan.summary()
            ),
        }
        if self._contract_payload_json is not None:
            summary.update(
                {
                    "identity": self.identity,
                    "schema_version": self.schema_version,
                    "program_fingerprint": self.program_fingerprint,
                    "options_fingerprint": self.options_fingerprint,
                    "environment_fingerprint": self.environment_fingerprint,
                    "compiler_fingerprint": self.compiler_fingerprint,
                    "mode": self.mode,
                    "backend": self.backend,
                    "device": self.device,
                    "target": self.target,
                    "batch_size": self.batch_size,
                    "precision": self.precision,
                    "is_distributed": self.is_distributed,
                }
            )
        return summary


@dataclass(frozen=True)
class RuntimeSelectionPlan:
    analysis: CircuitAnalysis
    recommended_mode: str
    recommended_candidate: RuntimeCandidate
    candidates: tuple[RuntimeCandidate, ...]
    objective: str
    user_tier: str
    usability_contract: str
    world_size: int = 1
    local_world_size: int = 1
    node_count: int = 1

    def summary(self) -> dict[str, Any]:
        return {
            "planner": "runtime_selection",
            "objective": self.objective,
            "recommended_mode": self.recommended_mode,
            "recommended_candidate": self.recommended_candidate.summary(),
            "user_tier": self.user_tier,
            "usability_contract": self.usability_contract,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "n_wires": self.analysis.n_wires,
            "n_instructions": self.analysis.n_instructions,
            "depth": self.analysis.depth,
            "two_qubit_gates": self.analysis.two_qubit_gates,
            "multi_qubit_gates": self.analysis.multi_qubit_gates,
            "has_noise": self.analysis.has_noise,
            "memory_plan": dict(self.recommended_candidate.memory_plan or {}),
            "communication_plan": dict(
                self.recommended_candidate.communication_plan or {}
            ),
            "gradient_plan": dict(self.recommended_candidate.gradient_plan or {}),
            "deployment_plan": dict(self.recommended_candidate.deployment_plan or {}),
            "candidates": tuple(candidate.summary() for candidate in self.candidates),
        }


__all__ = [
    "CircuitAnalysis",
    "ExecutionPlan",
    "LayerPlan",
    "RuntimeSelectionPlan",
]
