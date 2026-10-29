"""Private CS06-A checks over the existing sealed Runtime execution plan."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ..compiler import ResourceEstimate, estimate_resources
from ..compiler._construction.constraints import CircuitLimits
from ..compiler.basis_conversion import require_basis
from ..compiler.coupling import CouplingMap
from ..compiler.layout import Layout, _routing_swap_phases, final_layout
from ..core.ir import CircuitIR
from ..core.operator_schema import get_operator_schema
from ..core.runtime_config import RuntimeConfig
from ..runtime.dynamic._conditions import instruction_is_dynamic
from ..runtime.execution_plan import ExecutionPlan
from ..runtime.execution_plan_contract import (
    canonical_json,
    plan_decision,
    plan_execution_program,
    plan_program,
)
from ..runtime.result import ExecutionResult


class ExecutionContractError(ValueError):
    """A private execution check failed, with an actionable category."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _check_limits(limits: CircuitLimits, resources: ResourceEstimate) -> None:
    violations = limits.violations(resources)
    if violations:
        raise ExecutionContractError(
            "execution.constraints", "; ".join(d.message for d in violations)
        )


_UNBOUNDED_LIMITS = CircuitLimits()


def _program(plan: ExecutionPlan) -> CircuitIR:
    if not isinstance(plan, ExecutionPlan):
        raise TypeError("inspection requires an existing ExecutionPlan from fq.plan")
    decision = plan_decision(plan)  # Validates the existing serialized contract.
    if (
        plan.analysis is None
        or plan.mode != "statevector"
        or plan.state_mode != plan.mode
        or plan.backend not in {"torch", "pytorch"}
        or plan.device != "cpu"
        or type(plan.world_size) is not int
        or plan.world_size != 1
        or decision["world_size"] != 1
        or plan.batch_size != 1
        or plan.precision not in {"complex64", "complex128"}
        or plan.target not in {"auto", "state"}
        or decision["allow_approximate"]
        or decision["allow_backend_fallback"]
        or plan.noisy_execution_plan is not None
        or plan.to_dict()["extensions"]
    ):
        raise ExecutionContractError(
            "execution.profile",
            "require one exact local CPU torch statevector without noise or fallback",
        )
    if plan.state_bytes != decision["state_bytes"]:
        raise ExecutionContractError(
            "execution.controls", "state allocation differs from the sealed decision"
        )
    if not isinstance(plan.runtime_config, Mapping):
        raise ExecutionContractError(
            "execution.controls", "missing execution RuntimeConfig"
        )
    try:
        config = RuntimeConfig.from_manifest(plan.runtime_config)
    except (TypeError, ValueError) as error:
        raise ExecutionContractError("execution.controls", str(error)) from error
    if (
        config.backend not in {"torch", "pytorch"}
        or config.device != plan.device
        or config.complex_dtype != plan.precision
    ):
        raise ExecutionContractError(
            "execution.controls",
            "RuntimeConfig differs from the sealed backend, device or precision",
        )
    for index, layer in enumerate(plan.layers):
        qubits = tuple(sorted({q for gate in layer.instructions for q in gate.wires}))
        if (
            type(layer.index) is not int
            or layer.index != index
            or layer.wires != qubits
        ):
            raise ExecutionContractError(
                "execution.layers",
                "layer order or qubit summary differs from the canonical program",
            )
    recorded = plan_program(plan)
    actual = plan_execution_program(plan)
    if recorded.content_hash != actual.content_hash:
        raise ExecutionContractError(
            "execution.layers",
            "actual instruction layers differ from the sealed program",
        )
    if plan.analysis.n_qubits != actual.n_wires or plan.analysis.n_instructions != len(
        actual.instructions
    ):
        raise ExecutionContractError(
            "execution.controls", "analysis dimensions differ from the executable"
        )
    if actual.measurements or actual.observables:
        raise ExecutionContractError(
            "execution.profile",
            "this contract checks a full state, without output projections",
        )
    for gate in actual.instructions:
        schema = get_operator_schema(gate.name)
        if (
            schema is None
            or schema.semantic_kind != "unitary"
            or gate.matrix is not None
            or instruction_is_dynamic(gate)
        ):
            raise ExecutionContractError(
                "execution.profile",
                "require registered static unitaries with explicit gate semantics",
            )
    return actual


def _controls(plan: ExecutionPlan, program: CircuitIR) -> str:
    return canonical_json(
        {
            "runtime_config": plan.runtime_config,
            # The existing plan serializer retains routing in the canonical
            # program, but does not restore the optional diagnostic field.
            "routing_plan": (
                plan.routing_plan
                if plan.routing_plan is not None
                else program.metadata.get("routing", {})
            ),
            "state_mode": plan.state_mode,
            "world_size": plan.world_size,
            "state_bytes": plan.state_bytes,
        }
    )


def _placement(value: object, n: int) -> Layout:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != n
        or any(type(q) is not int for q in value)
    ):
        raise ExecutionContractError(
            "execution.layout",
            "routing must describe every logical qubit with plain integer slots",
        )
    return Layout(value)


def _layout(
    program: CircuitIR, plan: ExecutionPlan, coupling: CouplingMap | None
) -> Layout:
    n = program.n_wires
    if coupling is not None:
        if not isinstance(coupling, CouplingMap) or coupling.n_qubits != n:
            raise ExecutionContractError(
                "execution.topology",
                "provide an undirected CouplingMap with exactly the executable width",
            )
        edges = set(coupling.edges)
        for gate in program.instructions:
            if len(gate.wires) > 2 or (
                len(gate.wires) == 2 and tuple(sorted(gate.wires)) not in edges
            ):
                raise ExecutionContractError(
                    "execution.topology",
                    f"{gate.name}{gate.wires} has no admitted two-qubit coupling",
                )
    routing = program.metadata.get("routing")
    if not routing:
        if plan.routing_plan or any(
            g.metadata.get("routing_phase") is not None for g in program.instructions
        ):
            raise ExecutionContractError(
                "execution.layout",
                "Runtime routing controls have no corresponding program witness",
            )
        return Layout(range(n))
    if not isinstance(routing, dict) or coupling is None:
        raise ExecutionContractError(
            "execution.layout",
            "routed admission requires matching program evidence and an explicit topology",
        )
    if (
        routing.get("schema") != "flagquantum_routing_plan_v1"
        or routing.get("strategy") != "persistent_layout"
        or routing.get("direction_semantics") != "logical_wire_order_preserved"
        or routing.get("coupling_n_wires") != n
        or type(routing.get("mapping_restored")) is not bool
    ):
        raise ExecutionContractError(
            "execution.layout",
            "only intact v1 undirected persistent-layout routes are admitted",
        )
    if plan.routing_plan is not None and canonical_json(
        plan.routing_plan
    ) != canonical_json(routing):
        raise ExecutionContractError(
            "execution.layout",
            "Runtime routing controls differ from the executable routing witness",
        )
    if canonical_json(routing.get("coupling_edges")) != canonical_json(coupling.edges):
        raise ExecutionContractError(
            "execution.topology", "requested coupling differs from the routing witness"
        )
    try:
        initial = _placement(routing.get("initial_logical_to_physical"), n)
        before_restore = _placement(routing.get("pre_restore_logical_to_physical"), n)
        final = final_layout(program)
        marked = _routing_swap_phases(program)
    except ValueError as error:
        raise ExecutionContractError("execution.layout", str(error)) from error
    if initial != Layout(range(n)) or final.physical_slot_count != n:
        raise ExecutionContractError(
            "execution.layout",
            "initial placement and allocated physical slots are outside this contract",
        )
    for index, phase, _ in marked:
        source = program.instructions[index].metadata.get("source_instruction_index")
        if (
            type(source) is not int
            or source < 0
            or phase not in {"forward", "orientation", "final_restore"}
        ):
            raise ExecutionContractError(
                "execution.layout", "routed SWAP provenance is unsupported"
            )
    count = routing.get(
        "post_optimization_inserted_swap_count", routing.get("inserted_swap_count")
    )
    if type(count) is not int or count != len(marked):
        raise ExecutionContractError(
            "execution.layout", "routed SWAP count lacks complete instruction evidence"
        )
    if (
        initial.apply_swaps(tuple(pair for _, _, pair in marked)) != final
        or initial.apply_swaps(
            tuple(pair for _, phase, pair in marked if phase != "final_restore")
        )
        != before_restore
        or (routing["mapping_restored"] and final != initial)
        or (not routing["mapping_restored"] and final != before_restore)
    ):
        raise ExecutionContractError(
            "execution.layout",
            "replayed SWAPs disagree with the reported logical layout",
        )
    return final


@dataclass(frozen=True)
class CheckedExecution:
    """Private view of a checked plan; no execution or source-authentication role."""

    plan: ExecutionPlan
    limits: CircuitLimits
    basis: tuple[str, ...] | None
    coupling: CouplingMap | None
    registers: Mapping[str, tuple[int, ...]]
    _identity: str
    _program_hash: str
    _controls_json: str

    def _check(self, plan: ExecutionPlan) -> CircuitIR:
        program = _program(plan)
        if (
            plan.identity != self._identity
            or program.content_hash != self._program_hash
            or _controls(plan, program) != self._controls_json
        ):
            raise ExecutionContractError(
                "execution.identity",
                "program, configuration or layout changed; inspect a new plan",
            )
        _check_basis(program, self.basis)
        _layout(program, plan, self.coupling)
        return program

    @property
    def resources(self) -> ResourceEstimate:
        resources = estimate_resources(self._check(self.plan))
        _check_limits(self.limits, resources)
        return resources

    def verify_result(self, result: ExecutionResult) -> ResourceEstimate:
        """Verify association and actual executable costs without planning or running."""
        self._check(self.plan)
        if not isinstance(result.plan, ExecutionPlan):
            raise ExecutionContractError(
                "execution.identity", "result has no matching executable plan"
            )
        program = self._check(result.plan)
        state = result.state
        expected_dtype = (
            torch.complex128
            if result.plan.precision == "complex128"
            else torch.complex64
        )
        if (
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != expected_dtype
            or state.ndim not in {1, 2}
            or state.shape[-1] != 1 << program.n_wires
            or state.numel() != 1 << program.n_wires
        ):
            raise ExecutionContractError(
                "execution.result",
                "require one materialized CPU statevector matching the plan dimensions and precision",
            )
        resources = estimate_resources(program)
        _check_limits(self.limits, resources)
        return resources


def _check_basis(program: CircuitIR, gates: tuple[str, ...] | None) -> None:
    if gates is not None:
        outside = sorted({gate.name for gate in program.instructions} - set(gates))
        if outside:
            raise ExecutionContractError(
                "execution.basis",
                f"actual executable contains {outside} outside the requested basis; no conversion or substitution was performed",
            )


def inspect_execution(
    plan: ExecutionPlan,
    *,
    limits: CircuitLimits = _UNBOUNDED_LIMITS,
    basis: Iterable[str] | None = None,
    coupling: CouplingMap | None = None,
    registers: Mapping[str, Sequence[int]] | None = None,
) -> CheckedExecution:
    """Check a sealed plan before execution; perform no planning or conversion.

    Resource ceilings cover its entire executable. Logical named ports are caller
    declarations checked/projected through a replayed Compiler layout; source
    semantic ownership and workspace cleanup remain Compiler obligations.
    """
    if not isinstance(limits, CircuitLimits):
        raise TypeError("limits must be CircuitLimits")
    program = _program(plan)
    gates = None if basis is None else require_basis(basis)
    _check_basis(program, gates)
    layout = _layout(program, plan, coupling)
    mapped: dict[str, tuple[int, ...]] = {}
    seen: set[int] = set()
    if registers is not None and not isinstance(registers, Mapping):
        raise TypeError("registers must map logical names to qubit sequences")
    for name, ports in (registers or {}).items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(ports, Sequence)
            or not ports
        ):
            raise ExecutionContractError(
                "execution.registers",
                "provide nonempty names and nonempty logical qubit sequences",
            )
        for q in ports:
            if type(q) is not int or not 0 <= q < layout.n_qubits or q in seen:
                raise ExecutionContractError(
                    "execution.registers",
                    "logical named ports must be bounded plain integers and disjoint",
                )
            seen.add(q)
        mapped[name] = tuple(layout.physical_of(q) for q in ports)
    _check_limits(limits, estimate_resources(program))
    return CheckedExecution(
        plan,
        limits,
        gates,
        coupling,
        MappingProxyType(mapped),
        plan.identity,
        program.content_hash,
        _controls(plan, program),
    )
