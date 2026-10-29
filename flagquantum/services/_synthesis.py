"""Private CS-04 workflow trial; no public exports or compatibility guarantee."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, TypedDict

from ..compiler import ResourceEstimate, estimate_resources, schedule_layers
from ..compiler import compile as compile_program
from ..compiler._construction.frontend_python import Definition, capture
from ..compiler._construction.frontend_python import qfunc as qfunc
from ..compiler._construction.lowering import lower_arithmetic
from ..compiler._construction.profile import _positive_integer, uint_type
from ..compiler._hybrid.model import SourceLocation
from ..compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import get_operator_schema
from ..core.parameters import parameter_names_in_value
from ..runtime.dynamic._conditions import instruction_is_dynamic
from ..runtime.result import ExecutionResult
from ._synthesis_readout import Distribution
from ._synthesis_readout import read_distribution as _read_distribution

__all__ = ("InputState", "Program", "UInt", "qfunc", "synthesize")


class _Report(TypedDict):
    input_widths: Mapping[str, int]
    result_width: int
    input_state: str
    constants: Mapping[str, int]
    max_qubits: int
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


@dataclass(frozen=True)
class Program:
    """One trial product with existing IR, report and optional named reading."""

    circuit: CircuitIR
    arithmetic: CircuitIR
    report: Mapping[str, object]
    registers: Mapping[str, tuple[int, ...]]
    _source_hash: str
    _executable_hash: str

    def read_distribution(
        self,
        result: ExecutionResult,
        *,
        registers: Sequence[str] = ("result",),
        max_bits: int = 16,
        max_keys: int = 65536,
        max_amplitudes: int = 1 << 22,
        threshold: float = 1e-14,
    ) -> Distribution:
        """Read a bounded exact marginal without executing or replanning."""
        return _read_distribution(
            result,
            circuit=self.circuit,
            source_hash=self._source_hash,
            executable_hash=self._executable_hash,
            mapping=self.registers,
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
            or preparation.measurements
            or preparation.observables
        ):
            raise ValueError(
                "preparation must be unitary input-only CircuitIR without measurements or observables"
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


def synthesize(
    function: Definition,
    *,
    inputs: Mapping[str, UInt],
    input_state: InputState | None = None,
    constants: Mapping[str, int] | None = None,
    functions: Sequence[Definition] = (),
    max_qubits: int = 256,
    max_operations: int = 100_000,
    preparation: CircuitIR | None = None,
    preparation_map: Mapping[str, Sequence[int]] | None = None,
) -> Program:
    """Capture, infer and lower one conservative logical arithmetic candidate.

    Ordinary functions are the default trial entry. qfunc snapshots a definition
    earlier but uses the same arithmetic semantics. Named preparation and existing
    preparation IR are mutually exclusive. Target search is outside this trial.
    """
    _positive_integer(max_qubits, "max_qubits")
    _positive_integer(max_operations, "max_operations")
    if max_qubits > 256 or max_operations > 100_000:
        raise ValueError(
            "construction safety caps are 256 qubits and 100000 operations"
        )
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
    block = graph.body.blocks[0]
    mapping: dict[str, tuple[int, ...]] = {}
    position = 0
    bindings = {}
    for argument in block.arguments:
        name = argument.id.scope
        width = inputs[name].width
        mapping[name] = tuple(range(position, position + width))
        bindings[argument.id] = mapping[name]
        position += width
    result_width = block.operations[-1].operands[0].type.parameters[0]
    assert isinstance(result_width, int)
    mapping["result"] = tuple(range(position, position + result_width))
    position += result_width
    if position > max_qubits:
        raise ValueError(
            f"declared inputs and exact result need {position} qubits, exceeding max_qubits={max_qubits}"
        )
    prepared, state_name = _prepare(
        inputs, mapping, input_state, preparation, preparation_map
    )
    try:
        arithmetic = lower_arithmetic(
            (graph,),
            bindings,
            mapping["result"],
            entry=graph.name,
            n_qubits=max_qubits,
            workspace=tuple(range(position, max_qubits)),
            workspace_state="fresh_zero",
            max_operations=max_operations,
        )
    except HybridVerificationError as error:
        diagnostics = tuple(
            Diagnostic(
                d.code,
                d.message
                + (
                    "; increase max_qubits for this conservative candidate; smaller UInt domains are valid only if they fit your inputs"
                    if d.code == "construction.lowering.workspace"
                    else (
                        "; increase max_operations for this candidate"
                        if d.code == "construction.lowering.operations"
                        else ""
                    )
                ),
            )
            for d in error.diagnostics
        )
        raise HybridVerificationError(diagnostics) from error
    # Unused trailing zero allocations never enter the circuit; retain every
    # named input/output and every qubit actually touched by the candidate.
    total = max(
        position, max((max(i.wires) + 1 for i in arithmetic.instructions), default=0)
    )
    arithmetic = replace(arithmetic, n_wires=total, shape=())
    circuit = replace(
        arithmetic,
        instructions=(*prepared, *arithmetic.instructions),
        dtype="complex128",
    )
    compiled = compile_program(circuit)
    if max(len(circuit.instructions), len(compiled.instructions)) > max_operations:
        raise ValueError(
            "preparation plus arithmetic exceeds max_operations; increase the budget or reduce the supported computation"
        )
    canonical = replace(
        circuit,
        instructions=tuple(i for layer in schedule_layers(compiled) for i in layer),
    )
    report: _Report = {
        "input_widths": MappingProxyType(
            {name: spec.width for name, spec in inputs.items()}
        ),
        "result_width": result_width,
        "input_state": state_name,
        "constants": MappingProxyType(dict(constants or {})),
        "max_qubits": max_qubits,
        "workspace_qubits": total - position,
        "candidate": "conservative_integer",
        "function": graph.name,
        "source_locations": tuple(
            dict.fromkeys(
                op.location for op in block.operations if op.location is not None
            )
        ),
        "scope": "logical_default_compiled_preparation_and_arithmetic",
        "resources": estimate_resources(compiled),
        "source_resources": estimate_resources(circuit),
        "arithmetic_resources": estimate_resources(arithmetic),
    }
    return Program(
        circuit,
        arithmetic,
        MappingProxyType(report),
        MappingProxyType(mapping),
        circuit.content_hash,
        canonical.content_hash,
    )
