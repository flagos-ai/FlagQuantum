"""Native syntax to the admitted Compiler models, without synthesis or execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType

from ...compiler._construction.register_profile import (
    register_type,
    verify_register_program,
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
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError
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
    _INTRINSICS,
    _STATIC_GATES,
    MAX_EFFECT_OPERATIONS,
    _bound_helper,
    _literal_angle,
    _pauli_support,
    _preparation_value,
    _sparse_width,
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
from .affine import snapshot_start
from .amplification import expand_amplification
from .array_kickback import expand_nested_array_kickbacks
from .basis_regions import emit_closed_region, try_closed_region
from .bindings import (
    is_binding_scope,
)
from .borrowed_phase_views import expand_borrowed_phase_views
from .clean_selector_normalization import normalize_clean_selectors, probe_source
from .cnf_kickback import lower_cnf_kickback
from .compact_quantum import candidate_tape, emit_compact_region
from .computed_rotation import (
    composed_rotation,
    computed_output,
    reinterpreted_selector,
)
from .consuming_preparation import fuse_consuming_preparations
from .controlled_evolution import EVOLUTION_CALLS
from .copied_selector import slice_output
from .cscope import QaoaWorkflow, VqeSettings, split_cscope
from .deferred_measurement import defer_private_measurements
from .fixed_payoff import payoff_helpers
from .fixed_regions import (
    FixedPreparation,
    build_fixed_preparation,
    emit_fixed_preparation,
)
from .integers import checked_integer, checked_integer_array
from .iterations import expand_iterations
from .matrices import validate_matrix_constant
from .oracles import expand_static_oracles
from .permutations import is_const_cx_permutation, verify_permutations
from .phase_kickback import expand_phase_kickbacks
from .predicates import expand_predicate_oracles
from .qpe import (
    is_modular_power,
    power_action,
)
from .quantized_numeric import emit_numeric_program, try_numeric_program
from .quantum_callbacks import (
    specialize_quantum_callbacks,
)
from .selector_rotations import (
    fold_skip_control_tables,
    is_selector_helper,
)
from .syntax import (
    Expression,
    Function,
    Parameter,
    Parser,
    Record,
    Token,
    reject,
    tokenize,
)
from .unary_selection_proof import prove_annotated_selections
from .unitary_effects import (
    is_reversible_scope,
)
from .zero_walk import prove_zero_walk


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
                                            if field.kind == "qnum_signed"
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


def check_source_version(source_version: str, token: Token) -> None:
    if source_version != SUPPORTED_SOURCE_VERSION:
        reject(
            token,
            "version",
            f"only the pinned source profile {SUPPORTED_SOURCE_VERSION} is admitted",
        )


def _qaoa_workflow(
    settings: VqeSettings, main: Function, constants: Mapping[str, Expression]
) -> QaoaWorkflow:
    token = Token("cscope", settings.location)
    inputs = [p for p in main.parameters if p.role == "real" and p.kind == "real[]"]
    outputs = [p for p in main.parameters if p.role == "output" and p.kind == "qbit[]"]
    if (
        len(main.parameters) != 2
        or len(inputs) != 1
        or len(outputs) != 1
        or inputs[0].width != len(settings.initial_point)
        or outputs[0].width is None
        or len(main.statements) != 2
    ):
        reject(
            token,
            "cscope",
            "VQE requires a sized QAOA parameter vector and bit-array output",
        )
    initial, ansatz = main.statements
    parameter, output = inputs[0].name.text, outputs[0].name.text
    width = outputs[0].width
    assert width is not None
    if (
        initial.token.text != "allocate"
        or not initial.arguments
        or initial.arguments[-1].children
        or initial.arguments[-1].token.text != output
        or ansatz.token.text != "qaoa_penalty"
        or len(ansatz.arguments) != 4
        or any(arg.children for arg in ansatz.arguments[1:])
        or tuple(arg.token.text for arg in ansatz.arguments[1:])
        != (parameter, settings.operator, output)
    ):
        reject(
            token,
            "cscope",
            "VQE requires fresh QAOA preparation with its declared Hamiltonian",
        )
    operator = constants.get(settings.operator)
    if operator is None or operator.token.text != "PauliTerm[]":
        reject(token, "cscope", "VQE requires a declared PauliTerm[] Hamiltonian")
    diagonal = []
    for term in operator.children:
        axes, coefficient = term.children
        support = _pauli_support(axes, width, sparse=False)
        if any(axis != "Z" for _, axis in support):
            reject(
                axes.token, "cscope", "exact CVaR requires a diagonal I/Z Hamiltonian"
            )
        negative = coefficient.token.text == "-"
        magnitude = coefficient.children[0] if negative else coefficient
        value = _literal_angle(magnitude, token)
        diagonal.append(
            (sum(1 << bit for bit, _ in support), -value if negative else value)
        )
    return QaoaWorkflow(settings, parameter, output, width, tuple(diagonal))


def parse_qmod(
    source: str,
    *,
    filename: str = "<qmod>",
    source_version: str,
    classical_inputs: Mapping[str, float | Sequence[float]] | None = None,
    machine_precision: int | None = None,
) -> QmodImport:
    """Adapt the pinned unsigned scalar native subset to verified owned records.

    The caller declares source_version; this does not authenticate source bytes.
    Originals are independently checked against their retained source manifest.
    No files, SDKs, synthesis, execution, eval or generated Python are involved.
    """
    source, settings = split_cscope(source, filename)
    tokens = tokenize(source, filename)
    check_source_version(source_version, tokens[0])
    definitions: dict[str, Function] = {}
    parser = Parser(tokens)
    token_names = {token.text for token in tokens}
    parser.closed_fixed_point = (
        machine_precision is not None
        and "amplitude_amplification" in token_names
        and bool({"SIGNED", "True"} & token_names)
    )
    parser.zero_walk_proof = {
        "qstruct",
        "lambda",
        "bool",
        "CCX",
        "CRZ",
        "SDG",
        "free",
        "apply_to_all",
    } <= token_names and sum(token.text == "qstruct" for token in tokens) >= 2
    functions_syntax, declarations = parser.parse()
    constants: dict[str, Expression] = {}
    constant_integers: dict[str, int] = {}
    for name_token, operator in declarations:
        name = name_token.text
        if name in constants or name in _INTRINSICS or name == "main":
            reject(name_token, "name", "duplicate or reserved top-level constant")
        if name in parser.integer_constants:
            constant_integers[name] = parser.integer_values[name]
            constants[name] = operator
            continue
        if name in parser.matrix_constants:
            validate_matrix_constant(name_token, operator)
            constants[name] = operator
            continue
        if operator.token.text == "PauliTerm[]":
            if not operator.children:
                reject(operator.token, "pauli", "PauliTerm[] constant cannot be empty")
            width = len(operator.children[0].children[0].children)
            if not 1 <= width <= 256:
                reject(operator.token, "pauli", "Pauli word width must be in 1..256")
            for term in operator.children:
                axes, coefficient = term.children
                _pauli_support(axes, width, sparse=False)
                if coefficient.token.text == "-":
                    coefficient = coefficient.children[0]
                _literal_angle(coefficient, name_token)
            constants[name] = operator
            continue
        if operator.token.text != "SparsePauliOp":
            if operator.token.text == "[":
                for item in operator.children:
                    _preparation_value(item)
            else:
                _preparation_value(operator)
            constants[name] = operator
            continue
        terms, width_arg = operator.children
        width = _sparse_width(width_arg)
        if not terms.children:
            reject(
                terms.token,
                "pauli",
                "SparsePauliOp constant requires a nonempty Pauli sum",
            )
        for term in terms.children:
            axes, coefficient = term.children
            _pauli_support(axes, width, sparse=True)
            if coefficient.token.text == "-":
                coefficient = coefficient.children[0]
            _literal_angle(coefficient, name_token)
        constants[name] = operator
    for syntax in functions_syntax:
        name = syntax.name.text
        if name in definitions or name in _INTRINSICS:
            reject(syntax.name, "name", "duplicate function or intrinsic redefinition")
        if name in constants:
            reject(syntax.name, "name", "function shadows a top-level constant")
        for parameter in syntax.parameters:
            if parameter.name.text in constants and parameter.role != "operator":
                reject(parameter.name, "name", "parameter shadows a top-level constant")
        definitions[name] = syntax
    if "main" not in definitions:
        reject(tokens[0], "entry", "source requires one qfunc main")
    compact_tape = candidate_tape(definitions, parser.records)
    if compact_tape is not None:
        definitions = {"main": replace(definitions["main"], statements=())}
    if parser.zero_walk_proof:
        if constants:
            reject(
                tokens[0], "zero_walk", "closed zero-walk proof needs local static data"
            )
        definitions = dict(prove_zero_walk(definitions, parser.records).definitions)
    fixed_program: FixedPreparation | None = None
    if parser.closed_fixed_point:
        assert machine_precision is not None
        if not 1 <= machine_precision <= 8:
            reject(tokens[0], "precision", "machine precision must be in 1..8")
        root = definitions["main"]
        declared = {
            p.name.text: p for p in root.parameters if p.role in {"real", "integer"}
        }
        supplied = classical_inputs or {}
        if (
            not isinstance(supplied, Mapping)
            or set(declared) != set(supplied)
            or any(p.kind != "int" for p in declared.values())
        ):
            reject(
                root.name, "binding", f"classical inputs must match {sorted(declared)}"
            )
        integer_values = {
            name: checked_integer(supplied[name], port.name)
            for name, port in declared.items()
        }
        fixed_program = build_fixed_preparation(
            definitions, parser.records, machine_precision, integer_values
        )
        definitions = {"main": replace(root, statements=())}
    numeric_program = try_numeric_program(definitions)
    if numeric_program is not None:
        definitions = {"main": replace(definitions["main"], statements=())}
    closed_region = None
    if {"skip_control", "qft_const_adder_phase"} <= token_names:
        closed_region = try_closed_region(definitions, parser.records)
        if closed_region is not None:
            definitions = {"main": replace(definitions["main"], statements=())}
    definitions, deferred_measurements = defer_private_measurements(definitions)
    definitions = fold_skip_control_tables(definitions)
    definitions = expand_amplification(definitions, constants)
    definitions = expand_iterations(definitions)
    definitions = specialize_quantum_callbacks(definitions, constants)
    definitions = expand_nested_array_kickbacks(definitions)
    definitions = fuse_consuming_preparations(definitions)
    definitions = prove_annotated_selections(definitions)

    def compile_probe(
        function: Function, proof_definitions: Mapping[str, Function]
    ) -> HybridProgram:
        def resolve_probe(token: Token) -> tuple[Function, HybridProgram]:
            reject(
                token, "ownership", "selector proof requires statically bound helpers"
            )

        probe = _Adapt(
            probe_source(function),
            resolve_probe,
            proof_definitions,
            constants,
            parser.records,
        )
        probe.classical_integers.update(constant_integers)
        return probe.adapt()

    definitions = normalize_clean_selectors(definitions, compile_probe)
    definitions = expand_borrowed_phase_views(definitions, constants)
    definitions = expand_phase_kickbacks(definitions)
    verify_permutations(definitions)
    fixed_helpers = payoff_helpers(definitions)
    definitions = expand_predicate_oracles(
        definitions, constants, parser.records, inferred_helpers=fixed_helpers
    )
    definitions = expand_static_oracles(definitions, constants, parser.records)
    main = definitions["main"]

    def fractional_record(record: str) -> bool:
        return any(
            field.fractional_bits
            or (field.record is not None and fractional_record(field.record))
            for field in parser.records[record].fields
        )

    if any(
        port.record is not None and fractional_record(port.record)
        for port in main.parameters
    ):
        reject(
            main.name,
            "precision",
            "fractional record fields require checked fixed-point lowering",
        )
    if any(port.fractional_bits for port in main.parameters):
        reject(
            main.name,
            "precision",
            "fixed-point outputs require a checked synthesis path",
        )
    workflow = (
        _qaoa_workflow(settings, main, constants) if settings is not None else None
    )
    parameters = {
        p.name.text: p for p in main.parameters if p.role in {"real", "integer"}
    }
    if classical_inputs is None and workflow is not None:
        classical_inputs = {workflow.parameter: workflow.settings.initial_point}
    inputs = {} if classical_inputs is None else classical_inputs
    if not isinstance(inputs, Mapping) or set(inputs) != set(parameters):
        reject(
            main.name, "binding", f"classical inputs must match {sorted(parameters)}"
        )
    bindings: dict[str, float] = {}
    arrays: dict[str, tuple[float, ...]] = {}
    integer_bindings: dict[str, int] = {}
    integer_arrays: dict[str, tuple[int, ...]] = {}

    def checked_real(value: object, name: str) -> float:
        if (
            type(value) not in {float, int}
            or not isinstance(value, (float, int))
            or abs(value) > 2**53
            or not math.isfinite(value)
        ):
            reject(
                main.name,
                "binding",
                f"classical input {name!r} must be a finite real in ±2**53",
            )
        return float(value)

    for name, value in inputs.items():
        if parameters[name].role == "integer":
            if parameters[name].kind == "int":
                integer_bindings[name] = checked_integer(value, parameters[name].name)
            else:
                integer_arrays[name] = checked_integer_array(
                    value, parameters[name].name, parameters[name].width
                )
            continue
        array_width = parameters[name].width
        if array_width is None:
            bindings[name] = checked_real(value, name)
        else:
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != array_width
            ):
                reject(
                    main.name,
                    "binding",
                    f"classical input {name!r} requires exactly {array_width} real elements",
                )
            arrays[name] = tuple(checked_real(item, name) for item in value)

    for name, syntax in definitions.items():
        if name == "main" or not _bound_helper(syntax) or name in fixed_helpers:
            continue
        if any(port.kind == "SparsePauliOp" for port in syntax.parameters):
            power_action(syntax)
            continue
        if slice_output(syntax) is not None:
            continue
        if len({port.name.text for port in syntax.parameters}) != len(
            syntax.parameters
        ):
            reject(syntax.name, "name", "duplicate helper parameter")
        if any(port.fractional_bits for port in syntax.parameters):
            if (
                computed_output(syntax) is None
                and reinterpreted_selector(syntax, definitions) is None
            ):
                reject(
                    syntax.name,
                    "precision",
                    "fixed-point helpers require a checked computed rotation",
                )
            continue
        fixed_locals = [
            position
            for position, statement in enumerate(syntax.statements)
            if statement.declaration is not None
            and statement.declaration.fractional_bits
        ]
        if fixed_locals and (
            fixed_locals != [len(syntax.statements) - 3]
            or composed_rotation(syntax.statements[-3:], definitions) is None
        ):
            reject(
                syntax.name,
                "precision",
                "fixed-point locals require a checked computed rotation",
            )
        if any(
            port.role not in {"real", "integer", "inout", "callable"}
            and port.kind not in {"PauliTerm[]", "IndexedPauli[]"}
            and not (
                port.role == "output"
                and port.kind in {"qbit", "qbit[]", "qphase", "qnum"}
            )
            and not (
                port.role == "const"
                and (
                    port.kind == "qphase"
                    or (
                        port.kind in {"qbit", "qnum", "qnum_signed"}
                        and port.record is None
                    )
                    or (
                        port.record is not None
                        and len(syntax.statements) == 1
                        and syntax.statements[0].token.text == "$projected_rz"
                    )
                    or (
                        port.kind == "qbit[]"
                        and port.record is None
                        and not syntax.permutation
                        and any(p.role == "inout" for p in syntax.parameters)
                    )
                    or is_const_cx_permutation(syntax)
                    or is_selector_helper(syntax)
                    or (
                        port.kind == "qbit[]"
                        and port.record is None
                        and any(p.kind == "IndexedPauli[]" for p in syntax.parameters)
                    )
                )
            )
            for port in syntax.parameters
        ):
            reject(
                syntax.name,
                "role",
                "bound helpers require real or writable qubit array ports",
            )
        local_bits = {
            statement.declaration.name.text
            for statement in syntax.statements
            if statement.declaration is not None
            and statement.declaration.kind == "qbit"
        }
        output_names = {
            port.name.text for port in syntax.parameters if port.role == "output"
        }
        snapshot = snapshot_start(syntax.statements)
        for ordinal, statement in enumerate(syntax.statements):
            if snapshot is not None and ordinal in {snapshot, snapshot + 1}:
                continue
            if statement.token.text in {"allocate", "free"} and (
                not statement.arguments
                or statement.arguments[-1].children
                or statement.arguments[-1].token.text
                not in (
                    local_bits | output_names
                    if statement.token.text == "allocate"
                    else local_bits
                )
            ):
                reject(
                    statement.token,
                    "scope",
                    "bound allocation and free require a local qbit",
                )
            declaration = statement.declaration
            if declaration is not None and declaration.kind in {
                "qbit",
                "qbit[]",
                "qnum",
            }:
                continue
            controlled_qft = (
                statement.control_body is not None
                and len(statement.control_body) == 1
                and statement.control_body[0].token.text == "qft"
            )
            hadamard_within = (
                statement.within is not None
                and len(statement.within) == 1
                and statement.within[0].token.text == "H"
            )
            called = definitions.get(statement.token.text)
            bound_call = (
                called is not None
                and called.name.text != "main"
                and _bound_helper(called)
            ) or any(
                port.role == "callable" and port.name.text == statement.token.text
                for port in syntax.parameters
            )
            copy_output = (
                statement.token.text == "="
                and not any(arg.children for arg in statement.arguments)
                and statement.arguments[0].token.text in output_names
                and any(
                    port.name.text == statement.arguments[1].token.text
                    and port.kind == "qnum"
                    and port.role in {"const", "inout"}
                    for port in syntax.parameters
                )
            )
            if statement.declaration is not None or (
                statement.token.text
                not in _STATIC_GATES
                | EVOLUTION_CALLS
                | {
                    "swap_test",
                    "qsvt_lcu",
                    "grover_diffuser",
                    "multiswap",
                    "unitary",
                    "+=",
                    "^=",
                    "phase",
                    "$projected_rz",
                    "$kickback_phase",
                    "inplace_prepare_state",
                    "inplace_prepare_amplitudes",
                    "prepare_amplitudes",
                    "select_rotation",
                    "linear_pauli_rotations",
                    "allocate",
                    "free",
                    "->",
                }
                and not copy_output
                and not controlled_qft
                and not is_modular_power(statement)
                and not hadamard_within
                and not is_binding_scope(statement)
                and not is_reversible_scope(statement)
                and statement.repeat_index is None
                and statement.static_if_body is None
                and statement.map_callback is None
                and statement.control_body is None
                and statement.inverse_body is None
                and statement.callback_index is None
                and not bound_call
            ):
                reject(
                    statement.token,
                    "scope",
                    "bound helper bodies admit only gates, bounded unitary and Pauli evolution, and scoped QFT",
                )

    compiled: dict[str, tuple[_Adapt, HybridProgram]] = {}
    active: set[str] = set()

    def resolve(token: Token) -> tuple[Function, HybridProgram]:
        name = token.text
        if name == "main" or name not in definitions:
            reject(token, "call", f"unsupported native helper call {name!r}")
        adapter, program = build(definitions[name])
        return adapter.syntax, program

    def build(syntax: Function) -> tuple[_Adapt, HybridProgram]:
        name = syntax.name.text
        if name in active:
            reject(
                syntax.name, "recursion", "quantum helper call graph must be acyclic"
            )
        if name not in compiled:
            active.add(name)
            adapter = _Adapt(syntax, resolve, definitions, constants, parser.records)
            adapter.classical_integers.update(constant_integers)
            if name == "main":
                adapter.classical_values.update(bindings)
                adapter.classical_arrays.update(arrays)
                adapter.classical_integers.update(integer_bindings)
                adapter.integer_arrays.update(integer_arrays)
                if numeric_program is not None:
                    emit_numeric_program(adapter, numeric_program)
                if closed_region is not None:
                    emit_closed_region(adapter, closed_region)
                if fixed_program is not None:
                    emit_fixed_preparation(adapter, fixed_program)
                if compact_tape is not None:
                    emit_compact_region(adapter, compact_tape)
            program = adapter.adapt()
            compiled[name] = adapter, program
            active.remove(name)
        return compiled[name]

    # Bound helpers are checked structurally above and specialized at call sites.
    for syntax in definitions.values():
        if syntax is main or not _bound_helper(syntax):
            build(syntax)
    adapter, program = compiled["main"]
    records: dict[str, HybridProgram] = {}
    for name, (producer, model) in compiled.items():
        for function in producer.functions:
            records[function.name] = function
        if name != "main":
            records[name] = model
    functions = tuple(records.values())
    try:
        verify_register_program(program, functions)
    except HybridVerificationError as error:
        loc = program.location
        assert loc is not None
        diagnostics = tuple(
            (
                item
                if item.message.startswith(f"{loc.filename}:")
                else Diagnostic(
                    item.code, f"{loc.filename}:{loc.line}:{loc.column}: {item.message}"
                )
            )
            for item in error.diagnostics
        )
        raise HybridVerificationError(diagnostics) from error
    return QmodImport(
        program,
        functions,
        source_version,
        MappingProxyType(adapter.inferred),
        filename,
        MappingProxyType(adapter.fractional_bits),
        tuple(
            name
            for p in main.parameters
            for name in (
                (p.name.text,)
                if p.name.text in adapter.signed
                else (
                    tuple(
                        f"{p.name.text}.{field.name.text}"
                        for field in parser.records[p.record].leaves(parser.records)
                        if field.kind == "qnum_signed"
                    )
                    if p.record is not None
                    else ()
                )
            )
        ),
        workflow,
        deferred_measurements,
        exact_mps=parser.zero_walk_proof,
    )
