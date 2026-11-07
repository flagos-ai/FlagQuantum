"""One function adaptation context and its owned mutable state.

A context owns bindings, register views, scope and ordered effect emission.
Handlers share this context within a function; recursive helpers restore scoped
bindings before returning. Parser orchestration creates and discards contexts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType

from ...compiler._construction.register_profile import (
    register_type,
)
from ...compiler._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from .adapter_arithmetic import (
    bind_registers as _bind_registers,
)
from .adapter_arithmetic import (
    emit_indexed_x as _emit_indexed_x,
)
from .adapter_arithmetic import (
    increment as _increment,
)
from .adapter_arithmetic import (
    phase_factor as _phase_factor,
)
from .adapter_arithmetic import (
    real_phase as _real_phase,
)
from .adapter_calls import (
    borrowed_effect_callback as _borrowed_effect_callback,
)
from .adapter_calls import (
    call as _call,
)
from .adapter_calls import (
    call_bound_helper as _call_bound_helper,
)
from .adapter_controls import (
    coherent_helper_control as _coherent_helper_control,
)
from .adapter_controls import (
    controlled_rz_branches as _controlled_rz_branches,
)
from .adapter_controls import (
    scalar_control as _scalar_control,
)
from .adapter_controls import (
    within as _within,
)
from .adapter_evolution import (
    evolution_angle as _evolution_angle,
)
from .adapter_evolution import (
    exponentiate as _exponentiate,
)
from .adapter_evolution import (
    linear_pauli_rotations as _linear_pauli_rotations,
)
from .adapter_evolution import (
    pauli_evolution as _pauli_evolution,
)
from .adapter_evolution import (
    pauli_exponential as _pauli_exponential,
)
from .adapter_evolution import (
    pauli_terms as _pauli_terms,
)
from .adapter_evolution import (
    prepare_exponential_state as _prepare_exponential_state,
)
from .adapter_evolution import (
    prepare_state_vector as _prepare_state_vector,
)
from .adapter_evolution import (
    qdrift as _qdrift,
)
from .adapter_evolution import (
    qpe_flexible as _qpe_flexible,
)
from .adapter_evolution import (
    single_pauli_exponent as _single_pauli_exponent,
)
from .adapter_evolution import (
    suzuki_trotter as _suzuki_trotter,
)
from .adapter_evolution import (
    xor_update as _xor_update,
)
from .adapter_model import (
    MAX_EFFECT_OPERATIONS,
)
from .adapter_model import (
    SUPPORTED_SOURCE_VERSION as SUPPORTED_SOURCE_VERSION,
)
from .adapter_model import (
    QmodImport as QmodImport,
)
from .adapter_statements import (
    statement as _statement,
)
from .adapter_views import (
    angle as _angle,
)
from .adapter_views import (
    array_view as _array_view,
)
from .adapter_views import (
    classical_number as _classical_number,
)
from .adapter_views import (
    encode_data as _encode_data,
)
from .adapter_views import (
    gate_target as _gate_target,
)
from .adapter_views import (
    integer_array as _integer_array,
)
from .adapter_views import (
    real_array as _real_array,
)
from .adapter_views import (
    record_bit as _record_bit,
)
from .adapter_views import (
    record_field as _record_field,
)
from .adapter_views import (
    repeat as _repeat,
)
from .adapter_views import (
    static_integer as _static_integer,
)
from .cnf_kickback import lower_cnf_kickback
from .syntax import (
    Expression,
    Function,
    Parameter,
    Record,
    Token,
    reject,
)


class _Adapt:
    """Own the state shared by the bounded Qmod lowering handlers."""

    def __init__(
        self,
        syntax: Function,
        resolve: Callable[[Token], tuple[Function, HybridProgram]],
        definitions: Mapping[str, Function],
        constants: Mapping[str, Expression],
        record_types: Mapping[str, Record] = MappingProxyType({}),
    ) -> None:
        self.record_types = record_types
        self.syntax = syntax
        self.resolve = resolve
        self.definitions = definitions
        self.constants = constants
        self.classical_values: dict[str, float] = {}
        self.callback_bindings: dict[str, Function] = {}
        self.zero_callbacks: dict[str, tuple[Operation, ...]] = {}
        self.callback_arrays: dict[str, tuple[tuple[Operation, ...], ...]] = {}
        self.classical_arrays: dict[str, tuple[float, ...]] = {}
        self.classical_matrices: dict[str, tuple[tuple[float, ...], ...]] = {}
        self.classical_integers: dict[str, int] = {}
        self.integer_arrays: dict[str, tuple[int, ...]] = {}
        self.pauli_bindings: dict[str, Expression] = {}
        self.indexed_pauli: dict[str, tuple[tuple[int, int], ...]] = {}
        self.scope = syntax.name.text
        self.parameters: dict[str, Parameter] = {}
        for parameter in syntax.parameters:
            name = parameter.name.text
            if name in self.parameters:
                reject(parameter.name, "name", f"duplicate output {name!r}")
            self.parameters[name] = (
                replace(parameter, kind="qnum")
                if parameter.kind == "qnum_signed"
                else parameter
            )
        self.widths = {name: p.width for name, p in self.parameters.items()}
        self.inferred: dict[str, int] = {}
        self.signed: set[str] = {
            p.name.text
            for p in syntax.parameters
            if p.kind in {"qphase_signed", "qnum_signed"}
        }
        self.fractional_bits: dict[str, int] = {}
        self.borrowed_zero_phases: set[Value] = set()
        self.zero_entry: Callable[[Value], bool] | None = None
        self.certified_workspace: set[Value] = set()
        self.registers: dict[str, Value] = {}
        self.views: dict[str, tuple[tuple[Value, int], ...]] = {}
        self.suspended: set[str] = set()
        self.counter = len(self.parameters) + 1
        self.entry = Value(ValueId(self.scope, 0), QUANTUM_EFFECT)
        self.effect = self.entry
        self.operations: list[Operation] = []
        self.functions: list[HybridProgram] = []
        self.loop_indices: dict[str, int] = {}
        self.local_scope: list[tuple[str, Token]] | None = None
        self.dropped: set[str] = set()
        self.helper_stack: tuple[str, ...] = ()
        self.helper_expansions = 0
        self.loop_iterations = 0
        self.selector_guard: tuple[tuple[tuple[Value, int], ...], str, int] | None = (
            None
        )

    def register(
        self,
        token: Token,
        inferred: int | None = None,
        *,
        allow_qubit_array: bool = False,
        allow_fractional_output: bool = False,
        allocate_local: bool = False,
    ) -> Value:
        name = token.text
        if name in self.suspended:
            reject(token, "binding", "a bound source name is suspended")
        if name in self.views:
            reject(token, "view", "this operation requires a whole register")
        if name in self.dropped:
            reject(token, "drop", "a dropped register cannot be used again")
        if name in self.fractional_bits and not allow_fractional_output:
            reject(token, "precision", "a fractional result is a terminal output")
        if name not in self.parameters:
            reject(token, "name", f"unknown output {name!r}")
        if (
            self.parameters[name].kind in {"qphase", "qphase_signed"}
            and not allow_fractional_output
        ):
            reject(
                token, "precision", "fixed-point phase register requires qpe_flexible"
            )
        if self.parameters[name].kind == "qbit[]" and not allow_qubit_array:
            reject(
                token,
                "type",
                "qubit arrays support allocation and whole-array Hadamard",
            )
        if self.parameters[name].role in {"real", "integer"}:
            reject(token, "type", "classical parameter is not a quantum register")
        local = self.parameters[name].role == "local"
        hidden = local and (
            self.parameters[name].kind == "qbit[]"
            or (
                allocate_local
                and (
                    self.scope == "main"
                    or (self.helper_stack and self.parameters[name].kind == "qbit")
                )
                and self.local_scope is None
                and self.parameters[name].kind in {"qbit", "qnum", "qphase"}
            )
        )
        if (
            local
            and not hidden
            and self.local_scope is None
            and name not in self.registers
        ):
            reject(
                token,
                "scope",
                "local references are available only inside within/apply",
            )
        width = self.widths[name]
        if width is None and inferred is not None:
            self.widths[name] = width = inferred
            if not local:
                self.inferred[name] = inferred
        if width is None:
            reject(
                token,
                "width",
                f"{name!r} needs a declared width or explicit initialization size",
            )
        if inferred is not None and inferred > width:
            reject(
                token,
                "width",
                f"declared width {width} cannot fit the expression domain requiring {inferred} bits",
            )
        if name not in self.registers:
            if local:
                index = self.counter
                self.counter += 1
            else:
                index = tuple(self.parameters).index(name) + 1
            self.registers[name] = Value(
                ValueId(self.scope, index), register_type(width)
            )
            if local:
                if not hidden:
                    assert self.local_scope is not None
                    self.local_scope.append((name, token))
                self.append_op(
                    token,
                    Operation(
                        "qreg.hidden" if hidden else "qreg.local",
                        results=(self.registers[name],),
                        attributes={"name": f"{name}.within{index}"},
                        location=token.location,
                    ),
                )
        return self.registers[name]

    def append_op(self, token: Token, operation: Operation) -> None:
        if len(self.operations) >= MAX_EFFECT_OPERATIONS:
            reject(
                token,
                "budget.expansion",
                "native expansion exceeds 4096 effect operations",
            )
        self.operations.append(operation)

    def effect_op(
        self, name: str, token: Token, values: tuple[Value, ...], **attributes: object
    ) -> None:
        effect = Value(ValueId(self.scope, self.counter), QUANTUM_EFFECT)
        self.counter += 1
        self.append_op(
            token,
            Operation(
                name,
                (self.effect, *values),
                (effect,),
                attributes,
                location=token.location,
            ),
        )
        self.effect = effect

    @staticmethod
    def atom(node: Expression) -> Token:
        if node.children or node.token.text.isdecimal():
            reject(node.token, "signature", "this argument must be one output variable")
        return node.token

    # Bind the domain handlers to this adapter's shared lowering state.
    record_field = _record_field
    record_bit = _record_bit
    static_integer = _static_integer
    gate_target = _gate_target
    array_view = _array_view
    integer_array = _integer_array
    real_array = _real_array
    repeat = _repeat
    classical_number = _classical_number
    angle = _angle
    encode_data = _encode_data
    borrowed_effect_callback = _borrowed_effect_callback
    call_bound_helper = _call_bound_helper
    call = _call
    real_phase = _real_phase
    phase_factor = _phase_factor
    bind_registers = _bind_registers
    increment = _increment
    emit_indexed_x = _emit_indexed_x
    coherent_helper_control = _coherent_helper_control
    controlled_rz_branches = _controlled_rz_branches
    scalar_control = _scalar_control
    within = _within
    evolution_angle = _evolution_angle
    linear_pauli_rotations = _linear_pauli_rotations
    single_pauli_exponent = _single_pauli_exponent
    suzuki_trotter = _suzuki_trotter
    exponentiate = _exponentiate
    pauli_evolution = _pauli_evolution
    qdrift = _qdrift
    pauli_terms = _pauli_terms
    qpe_flexible = _qpe_flexible
    pauli_exponential = _pauli_exponential
    prepare_exponential_state = _prepare_exponential_state
    prepare_state_vector = _prepare_state_vector
    xor_update = _xor_update
    statement = _statement

    def adapt(self) -> HybridProgram:
        statements = self.syntax.statements
        for index, statement in enumerate(statements):
            if self.scope == "main" and index + 4 == len(statements):
                if lower_cnf_kickback(self, statements[index:]):
                    break
            self.statement(statement, str(index))
        if any(
            parameter.role == "output" and name in self.suspended
            for name, parameter in self.parameters.items()
        ):
            reject(self.syntax.name, "binding", "restore every output before return")
        if any(
            parameter.role == "local"
            and name in self.views
            and name not in self.suspended
            for name, parameter in self.parameters.items()
        ):
            reject(
                self.syntax.name,
                "binding",
                "merge or restore every local view before return",
            )
        output_parameters = tuple(
            p for p in self.syntax.parameters if p.role == "output"
        )
        borrowed = tuple(
            self.register(p.name) for p in self.syntax.parameters if p.role == "const"
        )
        for parameter in output_parameters:
            kind = self.parameters[parameter.name.text].kind
            if (
                kind == "qphase_signed"
                and parameter.name.text not in self.fractional_bits
            ):
                reject(
                    parameter.name,
                    "qpe",
                    "signed fixed-point output requires phase estimation or a checked writable bit view",
                )
            if kind == "qphase":
                width = self.widths[parameter.name.text]
                assert width is not None
                self.fractional_bits.setdefault(parameter.name.text, width)
            if parameter.record is not None:
                for field in self.record_types[parameter.record].leaves(
                    self.record_types
                ):
                    if field.kind == "qphase":
                        assert field.width is not None
                        self.fractional_bits[
                            f"{parameter.name.text}.{field.name.text}"
                        ] = field.width
        declared = tuple(
            self.register(p.name, allow_qubit_array=True, allow_fractional_output=True)
            for p in output_parameters
        )
        declarations = tuple(
            Operation(
                "qreg.declare",
                results=(value,),
                attributes={
                    "name": parameter.name.text,
                    **(
                        {
                            "record_type": parameter.record,
                            "fields": tuple(
                                (
                                    (
                                        field.name.text,
                                        field.kind,
                                        field.width,
                                        field.length,
                                    )
                                    if field.length is not None
                                    else (
                                        field.name.text,
                                        (
                                            "qnum"
                                            if field.kind in {"qnum_signed", "qphase"}
                                            else field.kind
                                        ),
                                        field.width,
                                    )
                                )
                                for field in self.record_types[parameter.record].leaves(
                                    self.record_types
                                )
                            ),
                        }
                        if parameter.record is not None
                        else {}
                    ),
                },
                location=parameter.name.location,
            )
            for parameter, value in zip(output_parameters, declared, strict=True)
        )
        program = HybridProgram(
            self.scope,
            Region(
                (
                    Block(
                        (self.entry, *borrowed),
                        (
                            *declarations,
                            *self.operations,
                            Operation(
                                "qreg.return",
                                (self.effect, *borrowed, *declared),
                                location=self.syntax.name.location,
                            ),
                        ),
                    ),
                )
            ),
            location=self.syntax.name.location,
        )
        return program
