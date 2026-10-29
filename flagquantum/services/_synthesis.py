"""Private constrained synthesis workflow; no stable public namespace."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, TypedDict

from ..compiler import ResourceEstimate, estimate_resources
from ..compiler._construction.constraints import CircuitLimits
from ..compiler._construction.frontend_python import (
    Definition,
    _QuantumFunction,
    capture,
)
from ..compiler._construction.frontend_python import qfunc as qfunc
from ..compiler._construction.profile import uint_type
from ..compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    Selection,
    select_arithmetic,
)
from ..compiler._construction.target import Target
from ..compiler._hybrid.model import SourceLocation
from ..compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from ..compiler.basis_conversion import require_basis
from ..compiler.coupling import CouplingMap
from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import get_operator_schema
from ..core.parameters import parameter_names_in_value
from ..runtime.dynamic._conditions import instruction_is_dynamic
from ..runtime.execution_plan import ExecutionPlan
from ..runtime.execution_plan_contract import plan_execution_program
from ..runtime.options import ExecutionOptions
from ..runtime.result import ExecutionResult
from ._synthesis_execution import CheckedExecution, inspect_execution
from ._synthesis_readout import Distribution
from ._synthesis_readout import probabilities as _probabilities

__all__ = (
    "ConstructionBudget",
    "InputState",
    "Program",
    "SynthesisError",
    "UInt",
    "qfunc",
    "synthesize",
)


class _Report(TypedDict):
    input_widths: Mapping[str, int]
    result_width: int
    input_state: str
    constants: Mapping[str, int]
    max_qubits: int | None
    workspace_qubits: int
    candidate: str
    function: str
    source_locations: tuple[SourceLocation, ...]
    scope: str
    resources: ResourceEstimate
    source_resources: ResourceEstimate
    arithmetic_resources: ResourceEstimate


@dataclass(frozen=True)
class UInt:
    """Unsigned basis domain; width is structural and is not differentiable."""

    width: int

    def __post_init__(self) -> None:
        if type(self.width) is not int:
            raise TypeError("UInt width must be a plain Python integer")
        uint_type(self.width)
        if self.width > 256:
            raise ValueError("UInt exceeds the 256-bit construction safety cap")


@dataclass(frozen=True)
class InputState:
    """Preparation description, not an uploaded unknown quantum state."""

    kind: Literal["basis", "uniform"]
    values: Mapping[str, int]

    @classmethod
    def basis(cls, **values: int) -> InputState:
        return cls("basis", MappingProxyType(dict(values)))

    @classmethod
    def uniform(cls) -> InputState:
        return cls("uniform", MappingProxyType({}))


class SynthesisError(HybridVerificationError):
    """A failed finite search, including checked rows and executable repairs."""

    def __init__(self, selection: Selection) -> None:
        self.selection = selection
        self.repairs = tuple(
            MappingProxyType(
                {
                    "max_qubits": repair.constraints.limits.max_qubits,
                    "max_operations": repair.constraints.limits.max_operations,
                    "max_depth": repair.constraints.limits.max_depth,
                    "max_workspace": repair.constraints.max_workspace,
                }
            )
            for repair in selection.recoveries
        )
        super().__init__(
            (
                *selection.diagnostics,
                *(d for row in selection.rows for d in row.diagnostics),
                *(
                    Diagnostic(
                        "synthesis.recovery",
                        f"retry one checked candidate with {dict(repair)}; not a global lower bound",
                    )
                    for repair in self.repairs
                ),
            )
        )


@dataclass(frozen=True)
class Program:
    """One synthesis product; run execution, then read declared logical names."""

    circuit: CircuitIR = field(repr=False)
    arithmetic: CircuitIR = field(repr=False)
    report: Mapping[str, object]
    registers: Mapping[str, tuple[int, ...]]
    selection: Selection = field(repr=False)
    _checked: CheckedExecution = field(repr=False)

    def _validate(self) -> None:
        row = self.selection.selected
        if row is None or row.target is None:
            raise ValueError("synthesis has no selected target")
        try:
            row.target.validate()
        except ValueError as error:
            raise ValueError(
                "the synthesis circuit changed; synthesize again before reading or execution"
            ) from error
        if self.circuit.content_hash != row.target.submitted.content_hash:
            raise ValueError("the synthesis circuit changed; synthesize again")
        if dict(self.registers) != dict(self.selection.registers):
            raise ValueError("logical registers changed; synthesize again")
        physical = {
            name: tuple(row.target.layout.physical_of(q) for q in ports)
            for name, ports in self.registers.items()
        }
        if (
            physical != dict(self._checked.registers)
            or self._checked.coupling != row.target.coupling
        ):
            raise ValueError(
                "execution layout or named ports differ from the selected target"
            )
        actual = plan_execution_program(self._checked.plan)
        expected = replace(
            row.target.executable,
            metadata={
                key: value
                for key, value in row.target.executable.metadata.items()
                if key != "runtime_config"
            },
        )
        if actual.content_hash != expected.content_hash:
            raise ValueError(
                "execution identity differs from the selected Compiler product"
            )
        if (
            self._checked.limits != self.selection.constraints.limits
            or self._checked.basis != self.selection.target.basis
        ):
            raise ValueError("execution constraints or basis changed; synthesize again")
        if self.selection.constraints.violations(
            self._checked.resources,
            row.arithmetic.peak_workspace if row.arithmetic is not None else 0,
        ):
            raise ValueError(
                "selected candidate no longer satisfies synthesis constraints"
            )

    @property
    def execution(self) -> ExecutionPlan:
        """Existing checked plan for fq.run; accessing it performs no planning."""
        self._validate()
        return self._checked.plan

    def probabilities(
        self,
        result: ExecutionResult,
        *,
        registers: Sequence[str] = ("result",),
        max_bits: int = 16,
        max_keys: int = 65536,
        max_amplitudes: int = 1 << 22,
        threshold: float = 1e-14,
    ) -> Distribution:
        """Read a bounded exact named marginal without execution or replanning."""
        self._validate()
        return _probabilities(
            result,
            checked=self._checked,
            registers=registers,
            max_bits=max_bits,
            max_keys=max_keys,
            max_amplitudes=max_amplitudes,
            threshold=threshold,
        )


def _prepare(
    inputs: Mapping[str, UInt],
    mapping: Mapping[str, tuple[int, ...]],
    input_state: InputState | None,
    preparation: CircuitIR | None,
    preparation_map: Mapping[str, Sequence[int]] | None,
) -> tuple[tuple[Instruction, ...], str]:
    """Validate explicit preparation before emitting the arithmetic candidate."""
    input_count = sum(spec.width for spec in inputs.values())
    prepared: list[Instruction] = []
    state_name = "basis_default_zero"
    if preparation is not None:
        if (
            not isinstance(preparation, CircuitIR)
            or preparation.n_wires != input_count
            or preparation.shape not in {(1 << input_count,), (1, 1 << input_count)}
            or preparation.measurements
            or preparation.observables
        ):
            raise ValueError(
                "preparation must be single-batch unitary input-only CircuitIR without measurements or observables"
            )
        if preparation_map is None or set(preparation_map) != set(inputs):
            raise ValueError("preparation_map must cover every named quantum input")
        remapping: dict[int, int] = {}
        for name, logical in mapping.items():
            if name == "result":
                continue
            physical = tuple(preparation_map[name])
            if len(physical) != len(logical):
                raise ValueError(f"preparation map width differs for {name!r}")
            for source, destination in zip(physical, logical, strict=True):
                if (
                    isinstance(source, bool)
                    or not isinstance(source, int)
                    or not 0 <= source < input_count
                    or source in remapping
                ):
                    raise ValueError(
                        "preparation maps must be disjoint and cover the input qubits"
                    )
                remapping[source] = destination
        for instruction in preparation.instructions:
            schema = get_operator_schema(instruction.name)
            if (
                schema is None
                or schema.semantic_kind != "unitary"
                or instruction.matrix is not None
                or instruction_is_dynamic(instruction)
            ):
                raise ValueError(
                    "use registered static unitary preparation gates; opaque matrices, measurements and reset are outside this trial"
                )
            if any(parameter_names_in_value(v) for v in instruction.params.values()):
                raise ValueError(
                    "bind symbolic preparation angles before synthesis; compose the arithmetic block for differentiable runtime bindings"
                )
            prepared.append(
                replace(
                    instruction, wires=tuple(remapping[q] for q in instruction.wires)
                )
            )
        state_name = "explicit_preparation"
    else:
        state = input_state or InputState.basis(**dict.fromkeys(inputs, 0))
        if not isinstance(state, InputState) or state.kind not in {"basis", "uniform"}:
            raise TypeError(
                "input_state must be InputState.basis(...) or InputState.uniform()"
            )
        if state.kind == "basis":
            if set(state.values) != set(inputs):
                raise ValueError("basis state must bind every declared input by name")
            for name, number in state.values.items():
                if (
                    isinstance(number, bool)
                    or not isinstance(number, int)
                    or not 0 <= number < 1 << inputs[name].width
                ):
                    raise ValueError(
                        f"basis value for {name!r} must be an integer in [0, {(1 << inputs[name].width) - 1}]"
                    )
                prepared.extend(
                    Instruction("x", (q,))
                    for bit, q in enumerate(reversed(mapping[name]))
                    if number & (1 << bit)
                )
        else:
            if state.values:
                raise ValueError(
                    "uniform preparation covers all input domains and accepts no basis values"
                )
            prepared.extend(Instruction("h", (q,)) for q in range(input_count))
        if input_state is not None:
            state_name = state.kind
    return tuple(prepared), state_name


_DEFAULT_BUDGET = ConstructionBudget()
_LINE_BASIS = ("x", "h", "s", "sdg", "t", "tdg", "ry", "rz", "cx", "u3", "swap")


def synthesize(
    function: Definition,
    *,
    inputs: Mapping[str, UInt],
    input_state: InputState | None = None,
    constants: Mapping[str, int] | None = None,
    functions: Sequence[Definition] = (),
    max_qubits: int | None = 256,
    max_operations: int | None = 100_000,
    max_depth: int | None = None,
    max_workspace: int | None = None,
    construction_budget: ConstructionBudget = _DEFAULT_BUDGET,
    objective: Literal["width", "depth"] = "width",
    basis: Iterable[str] | None = None,
    topology: Literal["line"] | None = None,
    preparation: CircuitIR | None = None,
    preparation_map: Mapping[str, Sequence[int]] | None = None,
) -> Program:
    """Capture high-level arithmetic, select and check one exact CPU program.

    Final ceilings are distinct from construction_budget. Synthesis plans once
    through the existing Runtime API but never executes. Run program.execution
    through fq.run and read actual probabilities by declared names.
    """
    constraints = Constraints(
        CircuitLimits(max_qubits, max_operations, max_depth), max_workspace
    )
    if not isinstance(construction_budget, ConstructionBudget):
        raise TypeError("construction_budget must be ConstructionBudget")
    if topology not in {None, "line"}:
        raise ValueError("the admitted topology profiles are logical (None) and line")
    if not isinstance(inputs, Mapping) or (
        constants is not None and not isinstance(constants, Mapping)
    ):
        raise TypeError(
            "inputs and constants must be mappings keyed by parameter names"
        )
    if any(not isinstance(spec, UInt) for spec in inputs.values()):
        raise TypeError(
            "each input must declare UInt(width); tensor and signed domains are unsupported"
        )
    if preparation is not None and input_state is not None:
        raise ValueError(
            "choose named input_state or existing preparation IR, not both"
        )
    if preparation is None and preparation_map is not None:
        raise ValueError("preparation_map requires an existing preparation IR")
    graph = capture(
        function,
        {name: spec.width for name, spec in inputs.items()},
        constants=constants,
        functions=functions,
    )
    if isinstance(function, _QuantumFunction):
        parameters = tuple(
            arg.arg
            for arg in (
                *function.source.node.args.posonlyargs,
                *function.source.node.args.args,
            )
        )
    else:
        parameters = tuple(inspect.signature(function).parameters)
    names = tuple(name for name in parameters if name in inputs)
    mapping: dict[str, tuple[int, ...]] = {}
    position = 0
    for name in names:
        mapping[name] = tuple(range(position, position + inputs[name].width))
        position += inputs[name].width
    prepared, state_name = _prepare(
        inputs, mapping, input_state, preparation, preparation_map
    )
    selected_basis = (
        (_LINE_BASIS if topology == "line" else None)
        if basis is None
        else require_basis(basis)
    )
    target = Target(
        basis=selected_basis,
        couplings=(
            tuple(
                CouplingMap.line(n)
                for n in range(position, construction_budget.qubits + 1)
            )
            if topology == "line"
            else ()
        ),
        optimize=topology is None,
        restore_layout=topology is None,
    )
    selection = select_arithmetic(
        (graph,),
        entry=graph.name,
        input_names=names,
        preparation=CircuitIR(position, prepared, dtype="complex128"),
        constraints=constraints,
        budget=construction_budget,
        target=target,
        objective=objective,
    )
    row = selection.selected
    if row is None or row.target is None or row.arithmetic is None:
        raise SynthesisError(selection)
    from .._api import plan

    options = ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    execution = plan(row.target.submitted, options=options)
    checked = inspect_execution(
        execution,
        limits=constraints.limits,
        basis=target.basis,
        coupling=row.target.coupling,
        registers=selection.registers,
    )
    resources = checked.resources
    block = graph.body.blocks[0]
    report: _Report = {
        "input_widths": MappingProxyType({name: inputs[name].width for name in names}),
        "result_width": len(selection.registers["result"]),
        "input_state": state_name,
        "constants": MappingProxyType(dict(constants or {})),
        "max_qubits": max_qubits,
        "workspace_qubits": row.arithmetic.peak_workspace,
        "candidate": row.strategy,
        "function": graph.name,
        "source_locations": tuple(
            dict.fromkeys(
                op.location for op in block.operations if op.location is not None
            )
        ),
        "scope": "actual_cpu_preparation_arithmetic_cleanup",
        "resources": resources,
        "source_resources": dict(row.stage_resources)["source"],
        "arithmetic_resources": estimate_resources(row.arithmetic.circuit),
    }
    program = Program(
        row.target.submitted,
        row.arithmetic.circuit,
        MappingProxyType(report),
        selection.registers,
        selection,
        checked,
    )
    program._validate()
    return program
