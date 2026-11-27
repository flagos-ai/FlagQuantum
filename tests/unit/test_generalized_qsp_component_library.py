"""Serialized GQSP preserves source-selected children and complete complex channels."""

import json
from collections import Counter
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import cirq as cirq
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_arguments import ComponentReference
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_matrix_factories import matrix_factories
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.formats.legacy_scalar_factories import scalar_phase_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.ecosystem._generalized_qsp_component_library import (
    generalized_qsp_library_factories,
)
from flagquantum.ecosystem.cirq._register_graph import _leaf_program
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_legacy_arithmetic_factories import _factories, _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _gqsp_library(source: Any, depth: int) -> Any:
    if isinstance(source.U, import_module("qualtran").CompositeBloq) and depth:
        raw = import_module("qualtran.serialization.bloq").bloqs_to_proto(
            source, source.U, max_depth=depth
        )
        return import_module(
            "flagquantum.compiler.formats.legacy_component_library"
        ).deserialize_component_library(raw.SerializeToString())
    return _library(source, depth)


def _native_views(
    source: Any,
    library: Any,
    positive_pair: tuple[ProgramRecord, ProgramRecord] | None = None,
) -> tuple[
    dict[str, Any],
    FunctionRecord,
    tuple[ProgramRecord, ProgramRecord],
    tuple[ProgramRecord, ProgramRecord],
]:
    factories = {
        **_factories(),
        **rotation_factories(),
        **matrix_factories(),
        **scalar_phase_factories(),
    }
    reference = next(
        arg.value for arg in library.entries[0].declaration.arguments if arg.name == "U"
    )
    assert isinstance(reference, ComponentReference)
    forward = resolve_component_program(
        library,
        reference.identifier,
        factories,
        entry_name=f"legacy_component_{reference.identifier}",
    )
    inverse = resolve_component_program(
        _library(
            source.U.adjoint(),
            2 if isinstance(source.U, import_module("qualtran").CompositeBloq) else 0,
        ),
        0,
        factories,
        entry_name="bare_inverse",
    )
    if positive_pair is not None:
        return factories, forward.functions[0], (forward, inverse), positive_pair
    pair = []
    for name, child in (
        ("positive_forward", source.U),
        ("positive_inverse", source.U.adjoint()),
    ):
        view = child.controlled()
        registers = _library(view, 0).entries[0].declaration.registers
        assert registers is not None
        record = _leaf_program(view, name, registers)
        pair.append(
            record if isinstance(record, ProgramRecord) else ProgramRecord((record,))
        )
    return factories, forward.functions[0], (forward, inverse), (pair[0], pair[1])


def _resolve(
    source: Any,
    depth: int,
    monkeypatch: pytest.MonkeyPatch,
    positive_pair: tuple[ProgramRecord, ProgramRecord] | None = None,
) -> ProgramRecord:
    library = _gqsp_library(source, depth)
    factories, child, ordinary, positive = _native_views(source, library, positive_pair)

    def child_owner(function: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
        assert function == child
        return ordinary

    def positive_owner(function: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
        assert function == child
        return positive

    owner_module = import_module("flagquantum.compiler.formats.legacy_gqsp_factories")
    actual_su2 = owner_module._su2_factory
    observed = []

    def observe(name: Any, registers: Any, arguments: Any) -> ProgramRecord:
        observed.append(dict(arguments))
        result: ProgramRecord = actual_su2(name, registers, arguments)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(owner_module, "_su2_factory", observe)
        patch.setitem(__import__("sys").modules, "qualtran", None)
        program = resolve_component_program(
            library,
            0,
            {
                **factories,
                **generalized_qsp_library_factories(
                    child_owner=child_owner, positive_child_owner=positive_owner
                ),
            },
        )
    assert verify_program_record(program) == ()
    assert len(observed) == len(source.signal_rotations)
    for actual, rotation in zip(observed, source.signal_rotations, strict=True):
        assert actual == {
            "theta": rotation.theta,
            "phi": rotation.phi,
            "lambd": rotation.lambd,
            "eps": rotation.eps,
        }
    return program


def _assert_source_call_operands(source: Any, program: ProgramRecord) -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    from flagquantum.compiler.registers.component_graph import (
        _elements,
        iter_component_calls,
    )
    from flagquantum.core.ir.program.registers import RegisterSide

    graph = source.decompose_bloq()
    _, source_inputs = import_module("qualtran").BloqBuilder.from_signature(
        graph.signature
    )
    native_values = {}
    position = 0
    arguments = iter(program.functions[0].blocks[0].arguments)
    for register in program.functions[0].registers or ():
        if register.side & RegisterSide.LEFT:
            for _ in register.all_idxs():
                size = int(register.bitsize)
                native_values[next(arguments)] = tuple(
                    ("entry", bit) for bit in range(position, position + size)
                )
                position += size
    source_values = {}
    position = 0
    for collection in source_inputs.values():
        for soq in np.asarray(collection).flat:
            size = int(soq.reg.bitsize)
            source_values[soq] = tuple(
                ("entry", bit) for bit in range(position, position + size)
            )
            position += size
    actual = []
    for index, inputs, outputs in iter_component_calls(program):
        tokens = tuple(
            token
            for value in _elements(inputs.values())
            for token in native_values[value.value]
        )
        actual.append((index, tokens))
        offset = 0
        for value in _elements(outputs):
            size = int(value.register.bitsize)
            native_values[value.value] = tuple(
                (str(index), bit) for bit in range(offset, offset + size)
            )
            offset += size
    expected = []
    for visit_index, (instance, inputs, outputs) in enumerate(graph.iter_bloqsoqs()):
        tokens = tuple(
            token
            for register in instance.bloq.signature.lefts()
            for soq in np.asarray(inputs[register.name]).flat
            for token in source_values[soq]
        )
        expected.append((visit_index, tokens))
        offset = 0
        for collection in outputs:
            for soq in np.asarray(collection).flat:
                size = int(soq.reg.bitsize)
                source_values[soq] = tuple(
                    (str(visit_index), bit) for bit in range(offset, offset + size)
                )
                offset += size
    assert actual == expected


def _check(
    source: Any,
    program: ProgramRecord,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reference_operator: torch.Tensor | None = None,
) -> Counter[str]:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    _assert_source_call_operands(source, program)
    width = int(source.U.signature.n_qubits())
    logical = width + 2
    # A coherent spectator separates the signal from reversed U operands.
    targets = tuple(reversed(range(2, logical)))
    words = []
    offset = 0
    for register in source.U.signature:
        for _ in register.all_idxs():
            size = int(register.bitsize)
            words.append(targets[offset : offset + size])
            offset += size
    assert offset == width
    ports = ((0,), *words)
    clean = tuple(range(logical, logical + 2))
    bound = bind_register_program(program, ports, clean=clean, borrowed=(1,))
    assert bound.outputs == ports
    # Use the original unitary protocol. Circuit.has_unitary has a distinct
    # original CompositeBloq decomposition failure for shaped U children.
    reference = (
        cirq.unitary(source)
        if reference_operator is None
        else reference_operator.detach().cpu().numpy()
    )
    source_gate = cirq.MatrixGate(reference).on(
        cirq.LineQubit(0), *(cirq.LineQubit(q) for q in targets)
    )
    expected = torch.tensor(
        cirq.Circuit(source_gate).unitary(qubit_order=cirq.LineQubit.range(logical)),
        dtype=torch.complex128,
    )
    counts = Counter(op.name for op in bound.circuit.instructions)
    measured = counts["measure"]
    stride = 1 << (bound.circuit.n_wires - logical)
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        ir = CircuitIR.from_json(bound.circuit.to_json())
        for column in range(1 << logical):
            initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            initial[column * stride] = 1
            result = torch.zeros_like(initial)
            result[::stride] = expected[:, column] / 2 ** (measured / 2)
            for history in product((0, 1), repeat=measured):
                torch.testing.assert_close(
                    forced_branch(initial, ir.instructions, history),
                    result,
                    atol=1e-11,
                    rtol=0,
                )
        coefficients = torch.arange(1, (1 << logical) + 1).double()
        amplitudes = torch.complex(coefficients, 2 - 3 * coefficients)
        amplitudes /= amplitudes.norm()
        initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
        initial[::stride] = amplitudes
        appended = Circuit(ir.n_wires, inputs=initial, dtype=initial.dtype).append(
            FieldOperation(ir.n_wires, ir.instructions)
        )
        assert (
            json.loads(appended.to_ir().to_json())["instructions"]
            == json.loads(ir.to_json())["instructions"]
        )
        target = torch.zeros_like(initial)
        target[::stride] = expected @ amplitudes
        if measured:
            from flagquantum.dynamic import DynamicCircuit
            from flagquantum.runtime.dynamic.execution import run_dynamic

            dynamic = DynamicCircuit(ir.n_wires, inputs=initial, dtype=initial.dtype)
            dynamic.append(FieldOperation(ir.n_wires, ir.instructions))
            dynamic_result = run_dynamic(
                dynamic, shots=3, seed=17, strategy="trajectory"
            )
            torch.testing.assert_close(
                dynamic_result.final_states.reshape(3, -1),
                target.expand(3, -1),
                atol=1e-11,
                rtol=0,
            )
        else:
            torch.testing.assert_close(
                appended.state().flatten(), target, atol=1e-11, rtol=0
            )
    # Phase precision is retained by the actual SU2 constructor dependency.
    rotations = [
        f
        for f in program.functions
        if f.name.startswith("main_signal_rotation_") and f.name.count("_") == 3
    ]
    assert len(rotations) == len(source.P)
    return counts


@pytest.mark.parametrize(
    "name", ("_gqsp", "_gqsp_with_negative_power", "_gqsp_with_large_negative_power")
)
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_all_original_factories_keep_typed_calls_exact_phases_and_full_complex_execution(
    name: str, depth: int, locked_source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = getattr(import_module("qualtran.bloqs.qsp.generalized_qsp"), name)()
    program = _resolve(source, depth, monkeypatch)
    root_calls = [op.callee or "" for op in program.functions[0].blocks[0].operations]
    roles = [
        (
            "rotation"
            if "signal_rotation" in name
            else (
                "positive_inverse"
                if "positive_inverse" in name
                else (
                    "negative_forward" if "negative_forward" in name else "bare_inverse"
                )
            )
        )
        for name in root_calls
    ]
    upstream = import_module("qualtran")
    su2_type = import_module("qualtran.bloqs.basic_gates.su2_rotation").SU2RotationGate
    inverse = source.U.adjoint()
    positive = inverse.controlled()
    negative = source.U.controlled(upstream.CtrlSpec(cvs=0))
    expected_roles = []
    for instance, _, _ in source.decompose_bloq().iter_bloqsoqs():
        if isinstance(instance.bloq, su2_type):
            expected_roles.append("rotation")
        elif instance.bloq == positive:
            expected_roles.append("positive_inverse")
        elif instance.bloq == negative:
            expected_roles.append("negative_forward")
        else:
            assert instance.bloq == inverse
            expected_roles.append("bare_inverse")
    assert roles == expected_roles
    _check(source, program, monkeypatch)


@pytest.mark.parametrize("kind", ("matrix1", "matrix2", "zpow", "cnot"))
@pytest.mark.parametrize("power", (0, 1, 5))
def test_matrix_and_actual_measured_z_power_children_keep_all_histories_and_private_work(
    kind: str, power: int, locked_source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_module = import_module("qualtran.bloqs.qsp.generalized_qsp")
    if kind == "zpow":
        child = import_module("qualtran.bloqs.basic_gates.rotation").ZPowGate(0.29)
    elif kind == "cnot":
        child = import_module("qualtran.bloqs.basic_gates").CNOT()
    else:
        child = import_module(
            "qualtran.bloqs.for_testing.matrix_gate"
        ).MatrixGate.random(int(kind[-1]), random_state=42)
    source = source_module.GeneralizedQSP.from_qsp_polynomial(
        child, (0.5, 0, 0.5), negative_power=power, precision=2e-8
    )
    counts = _check(source, _resolve(source, 0, monkeypatch), monkeypatch)
    assert counts["measure"] == (2 if kind == "zpow" else 0)


@pytest.mark.parametrize("width,shape", ((2, ()), (1, (1, 2))))
@pytest.mark.parametrize("power", (0, 1, 5))
def test_shaped_stored_unitary_reuses_exact_helper_and_measured_owners(
    width: int,
    shape: tuple[int, ...],
    power: int,
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.formats.legacy_su2_factories import _su2_factory
    from flagquantum.compiler.formats.legacy_word_factories import _rename_program
    from flagquantum.compiler.registers.phase.generalized_qsp_component import (
        generalized_qsp_component_program,
    )
    from tests.unit.test_generalized_qsp_shaped_unitary import shaped_unitary

    programs, child = shaped_unitary(width, shape)
    source = import_module(
        "qualtran.bloqs.qsp.generalized_qsp"
    ).GeneralizedQSP.from_qsp_polynomial(child, (0.5, 0, 0.5), negative_power=power)
    rotations = tuple(
        _su2_factory(
            f"signal_rotation_{i}",
            None,
            {
                "theta": float(r.theta),
                "phi": float(r.phi),
                "lambd": float(r.lambd),
                "eps": r.eps,
            },
        )
        for i, r in enumerate(source.signal_rotations)
    )
    native = generalized_qsp_component_program(
        programs[None, False],
        programs[None, True],
        programs[1, True],
        programs[0, False],
        rotations,
        negative_power=power,
    )
    direct = _rename_program(native, "main")
    counts = _check(source, direct, monkeypatch)
    assert counts["measure"] == 2
    library_program = _resolve(
        source, 2, monkeypatch, (programs[1, False], programs[1, True])
    )
    _check(source, library_program, monkeypatch)
    from tests.unit.test_generalized_qsp_entangling_unitary import assert_costs

    assert_costs(source, 3, power, 1, system_width=2)


def test_phase_owner_replacement_retains_complete_program_and_bound_ir(
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.formats.legacy_gqsp_factories import (
        generalized_qsp_factories,
    )

    source = import_module(
        "qualtran.bloqs.qsp.generalized_qsp"
    )._gqsp_with_large_negative_power()
    library = _library(source, 0)
    factories, child, ordinary, positive = _native_views(source, library)
    theta, phi, lambd = source._qsp_phases
    frozen = tuple(map(float, theta)), tuple(map(float, phi)), float(lambd)

    def own(function: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
        assert function == child
        return ordinary

    def controlled(function: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
        assert function == child
        return positive

    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        native = resolve_component_program(
            library,
            0,
            {
                **factories,
                **generalized_qsp_library_factories(
                    child_owner=own, positive_child_owner=controlled
                ),
            },
        )
        replaced = resolve_component_program(
            library,
            0,
            {
                **factories,
                **generalized_qsp_factories(
                    phase_factors=lambda _p, _q: frozen,
                    child_owner=own,
                    positive_child_owner=controlled,
                ),
            },
        )
    assert native == replaced
    assert (
        bind_register_program(native, ((0,), (1,)), clean=(2,)).circuit.to_json()
        == bind_register_program(replaced, ((0,), (1,)), clean=(2,)).circuit.to_json()
    )


def test_invalid_or_missing_phase_and_exact_child_ownership_fails_explicitly(
    locked_source: Any,
) -> None:
    from dataclasses import replace

    from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
        fixed_gate_factories,
    )
    from flagquantum.compiler.formats.legacy_gqsp_factories import (
        generalized_qsp_factories,
    )
    from flagquantum.errors import SerializationError

    fixed = fixed_gate_factories()
    u = fixed["qualtran.bloqs.basic_gates.x_basis.XGate"]("u", None, {})
    cx = fixed["qualtran.bloqs.basic_gates.cnot.CNOT"]("controlled", None, {})
    assert isinstance(u, FunctionRecord) and isinstance(cx, FunctionRecord)
    ordinary = ProgramRecord((u,))
    positive = ProgramRecord((cx,))
    factory = next(
        iter(
            generalized_qsp_library_factories(
                child_owner=lambda _: (ordinary, ordinary),
                positive_child_owner=lambda _: (positive, positive),
            ).values()
        )
    )
    arguments = {"U": u, "P": (0.5, 0.5), "Q": (0.5, -0.5)}
    for invalid in (
        {},
        {**arguments, "U": 3},
        {**arguments, "P": (0.5,)},
        {**arguments, "P": "missing"},
        {**arguments, "negative_power": "not-an-integer"},
        {**arguments, "negative_power": float("inf")},
        {**arguments, "other": 1},
    ):
        with pytest.raises(SerializationError):
            factory("invalid", None, invalid)
    with pytest.raises(ValueError, match="same degree"):
        factory("invalid", None, {**arguments, "Q": (0.5, 0, -0.5)})
    with pytest.raises(SerializationError, match="signature"):
        factory("invalid", (), arguments)
    wrong_phases = next(
        iter(
            generalized_qsp_factories(
                phase_factors=lambda _p, _q: ((1.0,), (1.0,), 0.0),
                child_owner=lambda _: (ordinary, ordinary),
                positive_child_owner=lambda _: (positive, positive),
            ).values()
        )
    )
    with pytest.raises(SerializationError, match="phase lengths"):
        wrong_phases("invalid", None, arguments)
    missing = ProgramRecord((replace(u, blocks=()),))
    incomplete = next(
        iter(
            generalized_qsp_library_factories(
                child_owner=lambda _: (missing, missing),
                positive_child_owner=lambda _: (positive, positive),
            ).values()
        )
    )
    with pytest.raises(ValueError, match="complete logical"):
        incomplete("invalid", None, arguments)


@pytest.mark.parametrize("power", (0, 1, 5))
def test_scalar_unitary_uses_existing_carrier_and_retains_original_sdk_refusal(
    power: int,
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_module = import_module("qualtran.bloqs.qsp.generalized_qsp")
    scalar = import_module("qualtran.bloqs.basic_gates.global_phase").GlobalPhase(
        exponent=0.25
    )
    source = source_module.GeneralizedQSP.from_qsp_polynomial(
        scalar, (0.5, 0, 0.5), negative_power=power
    )
    program = _resolve(source, 0, monkeypatch)
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    try:
        import numpy as np
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    with pytest.raises(AssertionError, match="All registers"):
        cirq.unitary(source)
    reference = source.decompose_bloq().tensor_contract()
    if power > 2:
        # Disconnected scalar nodes are retained but renumbered by the source.
        assert (
            sum(
                type(instance.bloq).__name__ == "GlobalPhase"
                for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
            )
            == power - 2
        )
        literal = tuple(
            cirq.flatten_op_tree(
                source.decompose_from_registers(
                    context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                    signal=np.array([cirq.LineQubit(0)]),
                )
            )
        )
        operations: list[cirq.Operation] = []
        for operation in literal:
            assert isinstance(operation, cirq.Operation)
            operations.append(operation)
        assert (
            sum(
                type(getattr(op.gate, "bloq", op.gate)).__name__ == "GlobalPhase"
                for op in operations
            )
            == power - 2
        )
        assert (
            sum(
                op.callee is not None and "bare_inverse" in op.callee
                for op in program.functions[0].blocks[0].operations
            )
            == power - 2
        )
    counts = _check(
        source,
        program,
        monkeypatch,
        reference_operator=torch.tensor(reference, dtype=torch.complex128),
    )
    assert counts["measure"] == 0


def test_symbolic_inverse_power_remains_a_logical_declaration_without_owner_execution(
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    module = import_module("qualtran.bloqs.qsp.generalized_qsp")
    source = module._gqsp()
    source = import_module("attrs").evolve(
        source, negative_power=sympy.Symbol("k", integer=True, nonnegative=True)
    )
    library = _library(source, 0)

    def unused(_: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
        raise AssertionError("A declaration must not execute a child owner")

    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        program = resolve_component_program(
            library,
            0,
            {
                **_factories(),
                **rotation_factories(),
                **generalized_qsp_library_factories(
                    child_owner=unused, positive_child_owner=unused
                ),
            },
        )
    assert verify_program_record(program) == ()
    assert program.functions[0].blocks == ()
    assert program.functions[0].registers == library.entries[0].declaration.registers
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(program, ((0,), (1,)))


@pytest.mark.parametrize("kind", ("zpow", "cnot"))
def test_actual_specialized_control_view_keeps_literal_primitives_and_feedback(
    kind: str,
    locked_source: Any,
) -> None:
    from flagquantum.compiler.registers.phase.generalized_qsp_component import (
        gqsp_control_view,
    )

    child = (
        import_module("qualtran.bloqs.basic_gates.rotation").ZPowGate(0.29)
        if kind == "zpow"
        else import_module("qualtran.bloqs.basic_gates").CNOT()
    )
    source = child.controlled()
    targets = _library(child, 0).entries[0].declaration.registers
    ports = _library(source, 0).entries[0].declaration.registers
    assert targets is not None and ports is not None
    module = _leaf_program(source, "positive", ports)
    module = module if isinstance(module, ProgramRecord) else ProgramRecord((module,))
    adapted = gqsp_control_view(module, targets)
    placement = ((2,), (0,)) if kind == "zpow" else ((2,), (0,), (3,))
    original = bind_register_program(module, placement, clean=(4, 5), entry="positive")
    normalized = bind_register_program(
        adapted, placement, clean=(4, 5), entry=adapted.functions[0].name
    )
    assert original.circuit.to_json() == normalized.circuit.to_json()
    assert original.events == normalized.events


@pytest.mark.parametrize("power", (-2, -1, 1.9, 5.9))
def test_source_integer_coercion_of_laurent_execution_power_is_preserved(
    power: int | float,
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = import_module(
        "qualtran.bloqs.qsp.generalized_qsp"
    )._gqsp_with_negative_power()
    source = import_module("attrs").evolve(source, negative_power=power)
    _check(source, _resolve(source, 0, monkeypatch), monkeypatch)
