"""Source phase gates, exact encoder pairs and bounded lookup form one clean channel."""

import json
from collections import Counter
from collections.abc import Mapping
from importlib import import_module
from itertools import product
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._complex_phase_library import (
    complex_phase_library_factories,
)
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_hamiltonian_gqsp_component_library import _capture
from tests.unit.test_legacy_arithmetic_factories import _factories, _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _source_operations(source: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    position = 0
    quregs = {}
    for register in source.signature:
        size = int(register.total_bits())
        quregs[register.name] = np.array(
            cirq.LineQubit.range(position, position + size)
        ).reshape((*register.shape, int(register.bitsize)))
        position += size
    context = cirq.DecompositionContext(
        cirq.SimpleQubitManager(prefix="phase_reference")
    )
    return (
        tuple(
            cirq.flatten_op_tree(
                source.decompose_from_registers(context=context, **quregs)
            )
        ),
        quregs,
    )


def _reference_operator(source: Any, *, encoder_tensors: bool = False) -> torch.Tensor:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    operations, ports = _source_operations(source)
    public = tuple(q for r in source.signature for q in ports[r.name].flat)
    literal = list(operations)
    if encoder_tensors:
        for index, child in ((0, source.encoder), (-1, source.encoder.adjoint())):
            matrix = child.decompose_bloq().tensor_contract()
            literal[index] = cirq.MatrixGate(matrix).on(*literal[index].qubits)
    circuit = cirq.Circuit(literal)
    hidden = tuple(sorted(circuit.all_qubits() - set(public)))
    full = circuit.unitary(qubit_order=(*public, *hidden))
    stride = 1 << len(hidden)
    columns = full[:, ::stride]
    support = np.zeros(len(full), dtype=bool)
    support[::stride] = True
    np.testing.assert_allclose(columns[~support], 0, atol=1e-11, rtol=0)
    return torch.tensor(columns[::stride], dtype=torch.complex128)


def _resolve(source: Any, depth: int, monkeypatch: pytest.MonkeyPatch) -> ProgramRecord:
    library = _library(source, depth)
    encoder = _capture(source.encoder, "owned_encoder")
    inverse = _capture(source.encoder.adjoint(), "owned_inverse")
    source_operations, _ = _source_operations(source)
    expected_gates = [op.gate for op in source_operations[2:-2]]
    module = import_module("flagquantum.ecosystem.cirq._complex_phase_library")
    actual_gate_program = module._gate_program
    seen = []

    def observe(gate: Any, signature: Any, name: str) -> FunctionRecord | ProgramRecord:
        seen.append(gate)
        result: FunctionRecord | ProgramRecord = actual_gate_program(
            gate, signature, name
        )
        return result

    def owner(child: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
        assert child.registers == encoder.functions[0].registers
        return encoder, inverse

    def encoder_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        assert registers == encoder.functions[0].registers
        return _rename_program(encoder, name)

    key = type(source.encoder).__module__ + "." + type(source.encoder).__qualname__
    with monkeypatch.context() as patch:
        patch.setattr(module, "_gate_program", observe)
        patch.setitem(__import__("sys").modules, "qualtran", None)
        program = resolve_component_program(
            library,
            0,
            {
                **_factories(),
                key: encoder_factory,
                **complex_phase_library_factories(
                    encoder_owner=owner,
                    target_names=lambda _: tuple(
                        r.name for r in source.encoder.target_registers
                    ),
                ),
            },
        )
    assert seen == expected_gates
    assert verify_program_record(program) == ()
    return program


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

    width = int(source.signature.n_qubits())
    precision = source.arctan_bitsize
    placement = tuple(reversed(range(1, width + 1)))
    ports_list = []
    offset = 0
    for register in source.signature:
        for _ in register.all_idxs():
            size = int(register.bitsize)
            ports_list.append(placement[offset : offset + size])
            offset += size
    ports = tuple(ports_list)
    work_count = (
        sum(int(r.total_bits()) for r in source.encoder.target_registers)
        + 1
        + precision
        + max(width - 1, 1)
    )
    clean = tuple(range(width + 1, width + 1 + work_count))
    bound = bind_register_program(program, ports, clean=clean, borrowed=(0,))
    assert bound.outputs == ports
    reference = (
        _reference_operator(source)
        if reference_operator is None
        else reference_operator
    ).numpy()
    expected = torch.tensor(
        cirq.Circuit(
            cirq.MatrixGate(reference).on(*(cirq.LineQubit(q) for q in placement))
        ).unitary(qubit_order=cirq.LineQubit.range(width + 1)),
        dtype=torch.complex128,
    )
    counts = Counter(op.name for op in bound.circuit.instructions)
    measurements = counts["measure"]
    stride = 1 << work_count
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        ir = CircuitIR.from_json(bound.circuit.to_json())
        for column in range(1 << (width + 1)):
            state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            state[column * stride] = 1
            target = torch.zeros_like(state)
            target[::stride] = expected[:, column] / 2 ** (measurements / 2)
            for history in product((0, 1), repeat=measurements):
                torch.testing.assert_close(
                    forced_branch(state, ir.instructions, history),
                    target,
                    atol=1e-11,
                    rtol=0,
                )
        raw = torch.arange(1, (1 << (width + 1)) + 1).double()
        amplitudes = torch.complex(raw, 2 - 3 * raw)
        amplitudes /= amplitudes.norm()
        state = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
        state[::stride] = amplitudes
        target = torch.zeros_like(state)
        target[::stride] = expected @ amplitudes
        circuit = DynamicCircuit(ir.n_wires, inputs=state, dtype=state.dtype)
        circuit.append(FieldOperation(ir.n_wires, ir.instructions))
        assert (
            json.loads(circuit.to_ir().to_json())["instructions"]
            == json.loads(ir.to_json())["instructions"]
        )
        result = run_dynamic(circuit, shots=3, seed=19, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(3, -1), target.expand(3, -1), atol=1e-11, rtol=0
        )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live
    return counts


@pytest.mark.parametrize("width,precision", [(1, 0), (1, 1), (1, 3), (2, 1)])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_actual_constructor_source_phase_gates_and_every_history(
    width: int, precision: int, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
    from qualtran.bloqs.mean_estimation.complex_phase_oracle_test import ExampleSelect

    source = ComplexPhaseOracle(ExampleSelect(width), precision)
    _check(source, _resolve(source, depth, monkeypatch), monkeypatch)


def _private_source_encoder(
    *, shaped: bool = False, controlled: bool = False, reverse_targets: bool = False
) -> Any:
    import attrs

    q = import_module("qualtran")
    gates = import_module("qualtran.bloqs.basic_gates")
    mcmt = import_module("qualtran.bloqs.mcmt")
    from qualtran.bloqs.multiplexers.select_base import SelectOracle

    def selections(self: Any) -> Any:
        return (q.Register("selection", q.QAny(2)),)

    def targets(self: Any) -> Any:
        return (
            (q.Register("target", q.QBit(), shape=(2,)),)
            if self.shaped
            else (q.Register("first", q.QBit()), q.Register("second", q.QBit()))
        )

    def build(self: Any, bb: Any, **soqs: Any) -> Any:
        controls = bb.split(soqs["selection"])
        target = np.asarray(
            soqs["target"] if self.shaped else [soqs["first"], soqs["second"]],
            dtype=object,
        ).copy()
        conjunction = (
            np.array([soqs["control"], controls[0]], dtype=object)
            if self.controlled
            else controls
        )
        conjunction, work = bb.add(mcmt.And(), ctrl=conjunction)
        work = bb.add(gates.SGate(), q=work)
        work, target[0] = bb.add(gates.CNOT(), ctrl=work, target=target[0])
        conjunction = bb.add(mcmt.And().adjoint(), ctrl=conjunction, target=work)
        if self.controlled:
            control, controls[0] = conjunction
            pair, target[1] = bb.add(
                gates.Toffoli(),
                ctrl=np.array([control, controls[1]], dtype=object),
                target=target[1],
            )
            control, controls[1] = pair
        else:
            controls = conjunction
            controls[0], target[1] = bb.add(
                gates.CNOT(), ctrl=controls[0], target=target[1]
            )
        outputs = {"selection": bb.join(controls, dtype=q.QAny(2))}
        if self.controlled:
            outputs["control"] = control
        outputs.update(
            {"target": target}
            if self.shaped
            else {"first": target[0], "second": target[1]}
        )
        return outputs

    cls = attrs.frozen(
        type(
            "PrivateEncoder",
            (SelectOracle,),
            {
                "__module__": __name__,
                "__annotations__": {
                    "shaped": bool,
                    "controlled": bool,
                    "reverse_targets": bool,
                },
                "signature": property(
                    lambda self: q.Signature(
                        [
                            *self.control_registers,
                            *self.selection_registers,
                            *(
                                reversed(self.target_registers)
                                if self.reverse_targets
                                else self.target_registers
                            ),
                        ]
                    )
                ),
                "control_registers": property(
                    lambda self: (
                        (q.Register("control", q.QBit()),) if self.controlled else ()
                    )
                ),
                "selection_registers": property(selections),
                "target_registers": property(targets),
                "build_composite_bloq": build,
            },
        )
    )
    return cls(shaped, controlled, reverse_targets)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_private_measured_encoder_exact_inverse_multiple_targets(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle

    source = ComplexPhaseOracle(_private_source_encoder(), 1)
    program = _resolve(source, depth, monkeypatch)
    with pytest.raises(TypeError, match="ClassicallyControlledOperation"):
        _reference_operator(source)
    reference = _reference_operator(source, encoder_tensors=True)
    counts = _check(source, program, monkeypatch, reference_operator=reference)
    assert counts["measure"] == 5


def test_exact_context_replacement_without_sdks_and_metadata_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Callable
    from dataclasses import replace

    from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
    from qualtran.bloqs.mean_estimation.complex_phase_oracle_test import ExampleSelect

    from flagquantum.compiler.formats.legacy_complex_phase_factories import (
        ComplexPhaseContext,
        complex_phase_factories,
    )
    from flagquantum.compiler.registers.lookup.arctan_lookup_component import (
        arctan_lookup_program,
    )
    from flagquantum.errors import SerializationError

    source = ComplexPhaseOracle(ExampleSelect(1), 1)
    encoder = _capture(source.encoder, "encoder")
    inverse = _capture(source.encoder.adjoint(), "inverse")
    contexts: list[Callable[[FunctionRecord, int], ComplexPhaseContext]] = []
    module = import_module("flagquantum.ecosystem.cirq._complex_phase_library")
    original = module.complex_phase_factories

    def observe(
        owner: Callable[[FunctionRecord, int], ComplexPhaseContext],
        *,
        on_pair: Any = None,
    ) -> Any:
        contexts.append(owner)
        return original(owner, on_pair=on_pair)

    with monkeypatch.context() as patch:
        patch.setattr(module, "complex_phase_factories", observe)
        first_factory = next(
            iter(
                complex_phase_library_factories(
                    encoder_owner=lambda _: (encoder, inverse),
                    target_names=lambda _: ("target",),
                ).values()
            )
        )
    arguments: dict[str, ResolvedArgument] = {
        "encoder": encoder.functions[0],
        "arctan_bitsize": 1,
    }
    first = first_factory("main", None, arguments)
    context = contexts[0](encoder.functions[0], 1)
    factory = next(iter(complex_phase_factories(lambda *_: context).values()))
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        second = factory("main", None, arguments)
    assert first == second
    assert isinstance(second, ProgramRecord)
    _check(source, second, monkeypatch)
    wrong = replace(context, arctan_pair=(arctan_lookup_program(1, 2),) * 2)
    with pytest.raises(SerializationError, match="precision mismatch"):
        next(iter(complex_phase_factories(lambda *_: wrong).values()))(
            "main", None, arguments
        )
    with pytest.raises(SerializationError, match="register signature"):
        factory("main", (), arguments)
    with pytest.raises(SerializationError, match="encoder and precision"):
        factory("main", None, {})
    default = first_factory(
        "default_precision", None, {"encoder": encoder.functions[0]}
    )
    assert isinstance(default, ProgramRecord) and verify_program_record(default) == ()


def test_original_shaped_target_allocation_refusal_is_retained() -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle

    from flagquantum.errors import SerializationError

    source = ComplexPhaseOracle(_private_source_encoder(shaped=True), 1)
    with pytest.raises(ValueError, match="shape"):
        _source_operations(source)
    encoder = _capture(source.encoder, "encoder")
    factory = next(
        iter(
            complex_phase_library_factories(
                encoder_owner=lambda _: (encoder, encoder),
                target_names=lambda _: ("target",),
            ).values()
        )
    )
    with pytest.raises(SerializationError, match="scalar target words"):
        factory("main", None, {"encoder": encoder.functions[0], "arctan_bitsize": 1})


def _phase_aliases(instructions: Any) -> Any:
    from dataclasses import replace
    from math import pi

    angles = {"t": pi / 4, "tdg": -pi / 4, "s": pi / 2, "sdg": -pi / 2}
    return tuple(
        (
            replace(op, name="phase", params={"theta": angles[op.name]})
            if op.name in angles
            else op
        )
        for op in instructions
    )


@pytest.mark.parametrize("power", range(32))
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("sign,value", [(0, 1), (1, -1)])
def test_each_literal_phase_recipe_matches_source_sdk_wires_feedback_and_angles(
    power: int, sign: int, value: int, adjoint: bool
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    from flagquantum.ecosystem.cirq._component_library import _gate_program
    from flagquantum.ecosystem.cirq._gate_metadata import cirq_gate_signature
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
    from tests.unit.register_source_graph import assert_source_wire_streams

    gate = (cirq.Z ** (value / 2 ** (1 + power))).controlled(control_values=[sign])
    source = cirq_gate_to_bloq(gate)
    if adjoint:
        source, gate = source.adjoint(), cirq.inverse(gate)
    record = _gate_program(gate, cirq_gate_signature(gate), "main")
    program = record if isinstance(record, ProgramRecord) else ProgramRecord((record,))
    actual = bind_register_program(program, ((0,), (1,)), clean=(2, 3))
    expected = import_register_emitter(source).snapshots[0]
    # S/T aliases name the same single physical phase instruction; no gates are
    # removed, fused or resynthesized for this per-wire comparison.
    assert_source_wire_streams(
        _phase_aliases(actual.circuit.instructions),
        _phase_aliases(expected.instructions),
    )
    for wire in range(4):
        actual_wire = [op for op in actual.circuit.instructions if wire in op.wires]
        source_wire = [op for op in expected.instructions if wire in op.wires]
        for observed, target in zip(actual_wire, source_wire, strict=True):
            assert (observed.matrix is None) == (target.matrix is None)
            if observed.matrix is not None:
                torch.testing.assert_close(
                    observed.matrix, target.matrix, atol=0, rtol=0
                )


def test_private_encoder_pair_matches_both_independent_source_snapshots() -> None:
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
    from tests.unit.register_source_graph import assert_source_wire_streams

    source = _private_source_encoder()
    reference = import_register_emitter(source).snapshots
    for index, child in enumerate((source, source.adjoint())):
        program = _capture(child, "main")
        actual = bind_register_program(program, ((0, 1), (2,), (3,)), clean=(4,))
        assert_source_wire_streams(
            _phase_aliases(actual.circuit.instructions),
            _phase_aliases(reference[index].instructions),
        )


def test_source_control_port_is_live_and_inactive_branch_is_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle

    source = ComplexPhaseOracle(_private_source_encoder(controlled=True), 1)
    program = _resolve(source, 1, monkeypatch)
    reference = _reference_operator(source, encoder_tensors=True)
    torch.testing.assert_close(
        reference[:4, :4], torch.eye(4, dtype=torch.complex128), atol=1e-11, rtol=0
    )
    _check(source, program, monkeypatch, reference_operator=reference)


def test_declared_target_order_survives_a_different_encoder_signature_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle

    source = ComplexPhaseOracle(_private_source_encoder(reverse_targets=True), 1)
    assert tuple(r.name for r in source.encoder.signature)[-2:] == ("second", "first")
    assert tuple(r.name for r in source.encoder.target_registers) == ("first", "second")
    program = _resolve(source, 0, monkeypatch)
    _check(
        source,
        program,
        monkeypatch,
        reference_operator=_reference_operator(source, encoder_tensors=True),
    )
