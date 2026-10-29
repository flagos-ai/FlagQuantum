"""Finite, deterministic selection over the two existing arithmetic schedules."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter
from types import MappingProxyType
from typing import Literal

from ...core.classical_control import is_classical_control_instruction
from ...core.ir import CircuitIR
from ...core.operator_schema import get_operator_schema
from ...core.parameters import parameter_names_in_value
from .._hybrid.model import HybridProgram
from .._hybrid.verifier import Diagnostic, HybridVerificationError
from ..resource_estimation import ResourceEstimate, estimate_resources
from .constraints import CircuitLimits
from .lowering import ArithmeticCandidate, Schedule, schedule_arithmetic
from .profile import verify_construction
from .target import Target, TargetError, TargetProduct, compile_target


@dataclass(frozen=True)
class Constraints:
    limits: CircuitLimits = field(default_factory=CircuitLimits)
    max_workspace: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.limits, CircuitLimits):
            raise TypeError("limits must be the shared CircuitLimits")
        if self.max_workspace is not None and (
            type(self.max_workspace) is not int or self.max_workspace < 0
        ):
            raise ValueError("max_workspace requires a nonnegative plain integer")

    def violations(
        self, resources: ResourceEstimate, workspace: int
    ) -> tuple[Diagnostic, ...]:
        extra: tuple[Diagnostic, ...] = ()
        if self.max_workspace is not None and workspace > self.max_workspace:
            extra = (
                Diagnostic(
                    "synthesis.constraints.max_workspace",
                    f"max_workspace={self.max_workspace}, actual={workspace}",
                ),
            )
        return (*self.limits.violations(resources), *extra)


@dataclass(frozen=True)
class ConstructionBudget:
    qubits: int = 256
    operations: int = 100_000
    candidate_checks: int = 2

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("qubits", self.qubits, 256),
            ("operations", self.operations, 100_000),
            ("candidate_checks", self.candidate_checks, 2),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(
                    f"{name} construction budget must be a plain integer in [1,{maximum}]"
                )


@dataclass(frozen=True)
class CandidateRow:
    strategy: Schedule
    source_identities: tuple[str, ...]
    arithmetic: ArithmeticCandidate | None
    target: TargetProduct | None
    resources: ResourceEstimate | None
    stage_resources: tuple[tuple[str, ResourceEstimate], ...]
    diagnostics: tuple[Diagnostic, ...]
    construction_seconds: float

    @property
    def feasible(self) -> bool:
        return self.target is not None and not self.diagnostics


@dataclass(frozen=True)
class Recovery:
    strategy: Schedule
    constraints: Constraints
    message: str = "Feasible for this checked candidate; not a global lower bound."


@dataclass(frozen=True)
class Selection:
    rows: tuple[CandidateRow, ...]
    selected: CandidateRow | None
    registers: Mapping[str, tuple[int, ...]]
    constraints: Constraints
    target: Target
    objective: Literal["width", "depth"]
    complete: bool
    recoveries: tuple[Recovery, ...]
    diagnostics: tuple[Diagnostic, ...]
    ranking_version: int = 1


def _rank(row: CandidateRow, objective: str) -> tuple[int, ...]:
    assert row.resources is not None and row.arithmetic is not None
    r = row.resources
    costs = (
        (r.n_qubits, r.depth, r.n_operations)
        if objective == "width"
        else (r.depth, r.n_operations, r.n_qubits)
    )
    return (*costs, row.arithmetic.peak_workspace, 0 if row.strategy == "retain" else 1)


def _recover(row: CandidateRow, constraints: Constraints) -> Recovery:
    assert row.resources is not None and row.arithmetic is not None
    r, limits = row.resources, constraints.limits

    def relaxed(value: int | None, actual: int) -> int | None:
        return None if value is None else max(value, actual)

    amended = Constraints(
        CircuitLimits(
            relaxed(limits.max_qubits, r.n_qubits),
            relaxed(limits.max_operations, r.n_operations),
            relaxed(limits.max_depth, r.depth),
        ),
        relaxed(constraints.max_workspace, row.arithmetic.peak_workspace),
    )
    assert not amended.violations(r, row.arithmetic.peak_workspace)
    return Recovery(row.strategy, amended)


_DEFAULT_CONSTRAINTS = Constraints()
_DEFAULT_BUDGET = ConstructionBudget()
_DEFAULT_TARGET = Target()
_STRATEGIES: tuple[Schedule, ...] = ("retain", "recompute")


def select_arithmetic(
    functions: Sequence[HybridProgram],
    *,
    entry: str,
    input_names: Sequence[str] | None = None,
    preparation: CircuitIR | None = None,
    constraints: Constraints = _DEFAULT_CONSTRAINTS,
    budget: ConstructionBudget = _DEFAULT_BUDGET,
    target: Target = _DEFAULT_TARGET,
    objective: Literal["width", "depth"] = "width",
) -> Selection:
    """Check both schedules; return no selected program after incomplete search.

    Input names are explicit presentation labels, never SSA namespace scopes.
    Preparation is an already explicit static input/output block. It may prepare
    a nonzero XOR target but never touches fresh workspace. No state is executed.
    """
    if (
        not isinstance(constraints, Constraints)
        or not isinstance(budget, ConstructionBudget)
        or not isinstance(target, Target)
    ):
        raise TypeError("use Constraints, ConstructionBudget and Target records")
    if objective not in {"width", "depth"}:
        raise ValueError("objective must be width or depth")
    graph = verify_construction(functions, entry=entry)
    block = graph.body.blocks[0]
    if isinstance(input_names, (str, bytes)):
        raise TypeError("input_names must be a sequence of logical parameter names")
    names = (
        tuple(f"arg{i}" for i in range(len(block.arguments)))
        if input_names is None
        else tuple(input_names)
    )
    if (
        len(names) != len(block.arguments)
        or any(
            not isinstance(name, str) or not name or name == "result" for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "input_names must uniquely cover all arguments and cannot reserve result"
        )
    mapping: dict[str, tuple[int, ...]] = {}
    bindings = {}
    position = 0
    for name, value in zip(names, block.arguments, strict=True):
        width = value.type.parameters[0]
        assert isinstance(width, int)
        ports = tuple(range(position, position + width))
        mapping[name] = ports
        bindings[value.id] = ports
        position += width
    width = block.operations[-1].operands[0].type.parameters[0]
    assert isinstance(width, int)
    mapping["result"] = tuple(range(position, position + width))
    position += width
    if preparation is not None and not isinstance(preparation, CircuitIR):
        raise TypeError("preparation must be an explicit CircuitIR")
    prepared = () if preparation is None else preparation.instructions
    if preparation is not None:
        if (
            preparation.n_wires > position
            or preparation.shape
            not in {(1 << preparation.n_wires,), (1, 1 << preparation.n_wires)}
            or preparation.measurements
            or preparation.observables
        ):
            raise ValueError(
                "preparation must cover input/output only without observations"
            )
        for gate in prepared:
            if any(parameter_names_in_value(value) for value in gate.params.values()):
                raise ValueError(
                    "bind symbolic preparation parameters before selection"
                )
            schema = get_operator_schema(gate.name)
            if (
                schema is None
                or schema.semantic_kind != "unitary"
                or gate.matrix is not None
                or gate.metadata.get("is_dynamic")
                or is_classical_control_instruction(gate)
            ):
                raise ValueError(
                    "preparation requires explicit registered static unitaries"
                )
    identities = tuple(
        f.semantic_identity for f in sorted(functions, key=lambda f: f.name)
    )
    rows: list[CandidateRow] = []
    complete = True
    for index, strategy in enumerate(_STRATEGIES):
        start = perf_counter()
        arithmetic = None
        product = None
        resources = None
        stages: tuple[tuple[str, ResourceEstimate], ...] = ()
        diagnostics: tuple[Diagnostic, ...] = ()
        if index >= budget.candidate_checks or position > budget.qubits:
            diagnostics = (
                Diagnostic(
                    "selection.budget",
                    "candidate checks or input/output construction capacity exhausted",
                ),
            )
            complete = False
        else:
            try:
                arithmetic = schedule_arithmetic(
                    functions,
                    bindings,
                    mapping["result"],
                    entry=entry,
                    n_qubits=budget.qubits,
                    workspace=tuple(range(position, budget.qubits)),
                    workspace_state="fresh_zero",
                    max_operations=budget.operations,
                    scheduling=strategy,
                )
                # Contiguous fresh pool: trim unused trailing capacity, not costs.
                circuit = replace(
                    arithmetic.circuit,
                    n_wires=position + arithmetic.peak_workspace,
                    shape=(),
                    dtype=target.config.complex_dtype,
                )
                arithmetic = replace(arithmetic, circuit=circuit)
                construction_seconds = perf_counter() - start
                source = replace(
                    circuit, instructions=(*prepared, *circuit.instructions)
                )
                product = compile_target(source, target, budget.operations)
                product.validate()
                resources = estimate_resources(product.executable)
                stages = tuple(
                    (name, estimate_resources(program))
                    for name, program in product.stages
                )
                diagnostics = constraints.violations(
                    resources, arithmetic.peak_workspace
                )
            except HybridVerificationError as error:
                diagnostics = error.diagnostics
                complete = False
            except TargetError as error:
                diagnostics = (Diagnostic(error.code, str(error)),)
                if error.code == "target.budget":
                    complete = False
        elapsed = (
            construction_seconds if arithmetic is not None else perf_counter() - start
        )
        rows.append(
            CandidateRow(
                strategy,
                identities,
                arithmetic,
                product,
                resources,
                stages,
                diagnostics,
                elapsed,
            )
        )
    feasible = [row for row in rows if row.feasible]
    selected = (
        min(feasible, key=lambda row: _rank(row, objective))
        if complete and feasible
        else None
    )
    recoveries = (
        tuple(
            _recover(row, constraints)
            for row in rows
            if row.target is not None and row.diagnostics
        )
        if complete and selected is None
        else ()
    )
    diagnostics = (
        ()
        if selected is not None
        else (
            Diagnostic(
                (
                    (
                        "selection.infeasible"
                        if any(row.target is not None for row in rows)
                        else "selection.unsupported_target"
                    )
                    if complete
                    else "selection.budget"
                ),
                (
                    "no feasible checked candidate"
                    if complete
                    else "incomplete evaluation; no best-so-far program is authorized"
                ),
            ),
        )
    )
    if selected is not None:
        assert selected.resources is not None
        for other in rows:
            if other.resources is None:
                continue
            if 0 < other.resources.n_qubits - selected.resources.n_qubits <= 1 and (
                selected.resources.depth >= 2 * max(1, other.resources.depth)
                or selected.resources.n_operations
                >= 2 * max(1, other.resources.n_operations)
            ):
                diagnostics = (
                    *diagnostics,
                    Diagnostic(
                        "selection.tradeoff",
                        f"{selected.strategy} saves at most one qubit but doubles depth/operations versus {other.strategy}; inspect both rows",
                    ),
                )
    return Selection(
        tuple(rows),
        selected,
        MappingProxyType(mapping),
        constraints,
        target,
        objective,
        complete,
        recoveries,
        diagnostics,
    )
