"""Native syntax to the admitted Compiler models, without synthesis or execution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType

from ...compiler._construction.register_profile import (
    verify_register_program,
)
from ...compiler._hybrid.model import (
    HybridProgram,
)
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from .adapter_model import (
    _INTRINSICS,
    _STATIC_GATES,
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
from .affine import snapshot_start
from .amplification import expand_amplification
from .array_kickback import expand_nested_array_kickbacks
from .basis_regions import emit_closed_region, try_closed_region
from .bindings import (
    is_binding_scope,
)
from .borrowed_phase_views import expand_borrowed_phase_views
from .clean_selector_normalization import normalize_clean_selectors, probe_source
from .compact_quantum import candidate_tape, emit_compact_region
from .computed_rotation import (
    composed_rotation,
    computed_output,
    reinterpreted_selector,
)
from .consuming_preparation import fuse_consuming_preparations
from .context import _Adapt
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
    Parser,
    Token,
    reject,
    tokenize,
)
from .unary_selection_proof import prove_annotated_selections
from .unitary_effects import (
    is_reversible_scope,
)
from .zero_walk import prove_zero_walk


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
    """Adapt the supported pinned Qmod source to Compiler-owned records.

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
                    "exact_amplitude_amplification",
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
                        if field.kind in {"qnum_signed", "qphase"}
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
