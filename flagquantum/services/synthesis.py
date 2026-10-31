"""Constrained synthesis workflow composing Compiler and checked Runtime APIs."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from os import PathLike
from types import MappingProxyType
from typing import Literal, overload

from ..compiler import estimate_resources
from ..compiler._construction.constraints import CircuitLimits
from ..compiler._construction.frontend_python import (
    Definition,
    _QuantumFunction,
    capture,
)
from ..compiler._construction.frontend_python import qfunc as qfunc
from ..compiler._construction.lowering import ArithmeticCandidate
from ..compiler._construction.profile import uint_type
from ..compiler._construction.register_profile import register_ports
from ..compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    Selection,
    select_arithmetic,
)
from ..compiler._construction.target import Target
from ..compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from ..compiler.basis_conversion import require_basis
from ..compiler.coupling import CouplingMap
from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import get_operator_schema
from ..core.parameters import parameter_names_in_value
from ..errors import PlanningError
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


def _target(
    position: int,
    budget: ConstructionBudget,
    basis: tuple[str, ...] | None,
    topology: Literal["line"] | None,
) -> Target:
    return Target(
        basis=basis,
        couplings=(
            tuple(CouplingMap.line(n) for n in range(position, budget.qubits + 1))
            if topology == "line"
            else ()
        ),
        optimize=topology is None,
        restore_layout=topology is None,
    )


@overload
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
    source_format: None = None,
    source_name: None = None,
    source_version: None = None,
    memory_limit_bytes: int | None = None,
    output_mode: Literal["value"] = "value",
) -> Program: ...


@overload
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
    basis: None = None,
    topology: None = None,
    preparation: CircuitIR | None = None,
    preparation_map: Mapping[str, Sequence[int]] | None = None,
    source_format: None = None,
    source_name: None = None,
    source_version: None = None,
    memory_limit_bytes: int | None = None,
    output_mode: Literal["phase"],
) -> Program: ...


@overload
def synthesize(
    function: str,
    *,
    inputs: None = None,
    input_state: None = None,
    constants: None = None,
    functions: tuple[()] = (),
    max_qubits: int | None = 256,
    max_operations: int | None = 100_000,
    max_depth: int | None = None,
    max_workspace: int | None = None,
    construction_budget: ConstructionBudget = _DEFAULT_BUDGET,
    objective: Literal["width", "depth"] = "width",
    basis: Iterable[str] | None = None,
    topology: Literal["line"] | None = None,
    preparation: None = None,
    preparation_map: None = None,
    source_format: Literal["qmod"],
    source_name: str | None = None,
    source_version: str | None = None,
    memory_limit_bytes: int | None = None,
    output_mode: Literal["value"] = "value",
) -> Program: ...


@overload
def synthesize(
    function: PathLike[str],
    *,
    inputs: None = None,
    input_state: None = None,
    constants: None = None,
    functions: tuple[()] = (),
    max_qubits: int | None = 256,
    max_operations: int | None = 100_000,
    max_depth: int | None = None,
    max_workspace: int | None = None,
    construction_budget: ConstructionBudget = _DEFAULT_BUDGET,
    objective: Literal["width", "depth"] = "width",
    basis: Iterable[str] | None = None,
    topology: Literal["line"] | None = None,
    preparation: None = None,
    preparation_map: None = None,
    source_format: Literal["qmod"] | None = None,
    source_name: None = None,
    source_version: str | None = None,
    memory_limit_bytes: int | None = None,
    output_mode: Literal["value"] = "value",
) -> Program: ...


def synthesize(
    function: Definition | str | PathLike[str],
    *,
    inputs: Mapping[str, UInt] | None = None,
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
    source_format: Literal["qmod"] | None = None,
    source_name: str | None = None,
    source_version: str | None = None,
    memory_limit_bytes: int | None = None,
    output_mode: Literal["value", "phase"] = "value",
) -> Program:
    """Synthesize high-level unsigned arithmetic for exact local CPU execution.

    Final ceilings are distinct from construction_budget. Synthesis plans once
    but never executes. Pass the resulting Program to fq.run, then read actual
    probabilities by declared names. The initial source-retaining Python profile
    and target support are documented in flagquantum.synth/README.md.

    PathLike[str] accepts local .qmod files; strings require source_format="qmod"
    and are always source text. Native source declares its own preparation and
    output names. Its default planned-state buffer ceiling is 512 MiB; explicit
    memory_limit_bytes applies to both source profiles. This is not a process
    peak-memory limit. Python retains its existing default memory behavior.

    output_mode="value" preserves the reversible XOR result. "phase" accepts
    only a one-bit Python return on logical CPU targets (basis/topology=None):
    it multiplies each input amplitude by (-1)**predicate and cleans workspace.
    Preparation stays outside that oracle. No result register is exported in
    phase mode; read inputs with explicit register names. Uniform probabilities
    alone cannot verify phase. Phase reports omit arithmetic_resources, and
    result_width=0 denotes the absence of an exported result register.

    Returns:
        A factory-created Program with checked execution, circuit and report.

    Raises:
        TypeError: output_mode is not a string, or input declarations are invalid.
        ValueError: The mode, phase source/target, or one-bit return is unsupported.
        InteropConversionError: Native input violates the bounded source profile.
        PlanningError: The planned state buffer exceeds the effective memory cap.
        SynthesisError: No fully checked candidate satisfies the request, or
            construction/search is incomplete. Checked repair arguments, when
            available, are attached to the error.

    Examples:
        Define the function in a source-retaining Python file::

            import flagquantum as fq
            from flagquantum import synth

            def add(x: int, y: int) -> int:
                return x + y

            program = synth.synthesize(
                add, inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
                input_state=synth.InputState.basis(x=1, y=1),
            )
            result = fq.run(program)
            probabilities = program.probabilities(result)
    """
    if not isinstance(output_mode, str):
        raise TypeError("output_mode must be a string: value or phase")
    if output_mode not in {"value", "phase"}:
        raise ValueError("output_mode must be value or phase")
    if output_mode == "phase":
        if isinstance(function, (PathLike, str)):
            raise ValueError(
                "phase output supports Python definitions only; native source requires value mode"
            )
        if basis is not None or topology is not None:
            raise ValueError(
                "phase output requires logical CPU targets: basis=None and topology=None"
            )
    constraints = Constraints(
        CircuitLimits(max_qubits, max_operations, max_depth), max_workspace
    )
    if not isinstance(construction_budget, ConstructionBudget):
        raise TypeError("construction_budget must be ConstructionBudget")
    if topology not in {None, "line"}:
        raise ValueError("the admitted topology profiles are logical (None) and line")
    if objective not in {"width", "depth"}:
        raise ValueError("objective must be width or depth")
    ExecutionOptions(memory_limit_bytes=memory_limit_bytes)
    if source_format is not None and not isinstance(source_format, str):
        raise TypeError("source_format must be a string or None")
    if source_format not in {None, "qmod"}:
        raise ValueError("the admitted source format is qmod")
    if not isinstance(functions, Sequence) or isinstance(functions, (str, bytes)):
        raise TypeError("functions must be a sequence of Python definitions")
    selected_basis = (
        (_LINE_BASIS if topology == "line" else None)
        if basis is None
        else require_basis(basis)
    )
    is_file = isinstance(function, PathLike)
    if isinstance(function, (PathLike, str)):
        if not is_file and source_format != "qmod":
            raise ValueError("text requires source_format='qmod'; use Path for files")
        if (
            any(
                value is not None
                for value in (
                    inputs,
                    input_state,
                    constants,
                    preparation,
                    preparation_map,
                )
            )
            or functions
        ):
            raise ValueError(
                "native source owns preparation and inputs; omit Python inputs/state/constants/functions/preparation"
            )
        from ..ecosystem._qmod.source import load_qmod
        from ._synthesis_registers import synthesize_registers

        imported = load_qmod(
            function, source_name=source_name, source_version=source_version
        )
        _, _, position = register_ports(imported.program)
        effective_memory = (
            512 * 1024**2 if memory_limit_bytes is None else memory_limit_bytes
        )
        return synthesize_registers(
            imported.program,
            imported.functions,
            constraints=constraints,
            budget=construction_budget,
            target=_target(position, construction_budget, selected_basis, topology),
            objective=objective,
            memory_limit_bytes=effective_memory,
            memory_limit_origin=(
                "profile_default" if memory_limit_bytes is None else "explicit"
            ),
            source_format="qmod",
            source_name=imported.source_name,
            source_version=imported.source_version,
        )
    if any(value is not None for value in (source_format, source_name, source_version)):
        raise ValueError("source_format/name/version apply only to native source")
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
    target = _target(position, construction_budget, selected_basis, topology)
    selection = select_arithmetic(
        (graph,),
        entry=graph.name,
        input_names=names,
        preparation=CircuitIR(position, prepared, dtype="complex128"),
        constraints=constraints,
        budget=construction_budget,
        target=target,
        objective=objective,
        output_mode=output_mode,
    )
    row = selection.selected
    if row is None or row.target is None or row.arithmetic is None:
        raise SynthesisError(selection)
    block = graph.body.blocks[0]
    report: dict[str, object] = {
        "input_widths": MappingProxyType({name: inputs[name].width for name in names}),
        "result_width": len(selection.registers.get("result", ())),
        "input_state": state_name,
        "constants": MappingProxyType(dict(constants or {})),
        "max_qubits": max_qubits,
        "function": graph.name,
        "source_locations": tuple(
            dict.fromkeys(
                op.location for op in block.operations if op.location is not None
            )
        ),
        "scope": (
            "actual_cpu_preparation_predicate_phase_cleanup"
            if output_mode == "phase"
            else "actual_cpu_preparation_arithmetic_cleanup"
        ),
        "source_format": "python",
        "output_mode": output_mode,
    }
    return _complete(
        selection,
        report,
        memory_limit_bytes=memory_limit_bytes,
        memory_limit_origin=(
            "profile_default" if memory_limit_bytes is None else "explicit"
        ),
    )


def _complete(
    selection: Selection,
    report: Mapping[str, object],
    *,
    memory_limit_bytes: int | None = None,
    memory_limit_origin: Literal["profile_default", "explicit"] = "profile_default",
) -> Program:
    """Admit either construction producer through one checked execution authority."""
    row = selection.selected
    if row is None or row.target is None or row.arithmetic is None:
        raise SynthesisError(selection)
    from .._api import plan

    options = ExecutionOptions(
        memory_limit_bytes=memory_limit_bytes,
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
        limits=selection.constraints.limits,
        basis=selection.target.basis,
        coupling=row.target.coupling,
        registers=selection.registers,
    )
    if memory_limit_bytes is not None and execution.state_bytes > memory_limit_bytes:
        raise PlanningError(
            f"execution memory budget exceeded: state requires {execution.state_bytes} "
            f"bytes, limit is {memory_limit_bytes}; no allocation was performed"
        )
    resources = checked.resources
    complete_report = dict(report)
    complete_report.update(
        workspace_qubits=row.arithmetic.peak_workspace,
        candidate=row.strategy,
        resources=resources,
        source_resources=dict(row.stage_resources)["source"],
        register_widths=MappingProxyType(
            {name: len(ports) for name, ports in selection.registers.items()}
        ),
        memory_limit_bytes=memory_limit_bytes,
        memory_limit_origin=memory_limit_origin,
    )
    complete_report.setdefault("output_mode", "value")
    if (
        isinstance(row.arithmetic, ArithmeticCandidate)
        and complete_report["output_mode"] == "value"
    ):
        complete_report["arithmetic_resources"] = estimate_resources(
            row.arithmetic.circuit
        )
    program = Program(
        row.target.submitted,
        row.arithmetic.circuit,
        MappingProxyType(complete_report),
        selection.registers,
        selection,
        checked,
    )
    program._validate()
    return program
