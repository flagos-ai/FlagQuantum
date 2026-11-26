"""Serialized local gates retain exact phase, decomposition and native ownership."""

import sys
from cmath import exp
from collections.abc import Mapping
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_arguments import ExternalGatePayload
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem.cirq._component_library import cirq_gate_library_factories
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_stored_rotation_array_resolution import _factories

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("locked_source"),
]
cirq = pytest.importorskip("cirq")


@pytest.mark.parametrize("kind", ("zpow", "cxpow", "matrix1", "matrix2", "scalar"))
def test_complete_local_gate_payloads_depths_phase_and_workspace(
    kind: str, locked_source: Any
) -> None:
    gates = {
        "zpow": cirq.ZPowGate(exponent=0.375),
        "cxpow": cirq.CXPowGate(exponent=-0.375),
        "matrix1": cirq.MatrixGate(
            exp(0.19j) * cirq.testing.random_unitary(2, random_state=17)
        ),
        "matrix2": cirq.MatrixGate(
            exp(-0.27j) * cirq.testing.random_unitary(4, random_state=23)
        ),
        "scalar": cirq.GlobalPhaseGate(exp(0.31j)),
    }
    gate = gates[kind]
    source = locked_source.CirqGateAsBloq(gate)
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    assert verify_program_record(record) == ()
    assert (
        resolve_component_program(_library(source, 1), 0, cirq_gate_library_factories())
        == record
    )
    width = cirq.num_qubits(gate)
    ports = tuple((i,) for i in range(width))
    bound = bind_register_program(record, ports, clean=tuple(range(width, width + 3)))
    assert bound.outputs == ports
    torch.testing.assert_close(
        logical_operator(bound.circuit, width),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
            1 << width, 1 << width
        ),
        atol=1e-11,
        rtol=0,
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit


def test_symbolic_gate_payload_and_execution_without_source_import(
    locked_source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sympy

    theta = sympy.Symbol("theta")
    gate = cirq.XPowGate(exponent=theta)
    library = _library(locked_source.CirqGateAsBloq(gate), 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        record = resolve_component_program(library, 0, cirq_gate_library_factories())
        bound = bind_register_program(record, ((0,),))
        circuit = Circuit(1, dtype=torch.complex128).append(
            FieldOperation(1, bound.circuit.instructions)
        )
        actual = get_unitary(circuit.bind_parameters({"theta": 0.37}))
    expected = cirq.unitary(cirq.resolve_parameters(gate, {"theta": 0.37}))
    torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-11, rtol=0)
    nonlinear = cirq.XPowGate(exponent=sympy.sin(theta) + theta**2 / 3)
    with pytest.raises(TypeError, match="JSON serializable"):
        _library(locked_source.CirqGateAsBloq(nonlinear), 0)


@pytest.mark.parametrize("width", (1, 2))
def test_stored_arrays_with_arbitrary_local_payloads(width: int) -> None:
    from qualtran.bloqs.rotations.programmable_rotation_gate_array import (
        ProgrammableRotationGateArray,
    )

    rotation = cirq.MatrixGate(
        exp(0.19j) * cirq.testing.random_unitary(1 << width, random_state=17)
    )
    interleaver = cirq.MatrixGate(
        exp(-0.27j) * cirq.testing.random_unitary(1 << width, random_state=23)
    )
    source = ProgrammableRotationGateArray(
        (1, 3),
        (2, 1),
        kappa=2,
        rotation_gate=rotation,
        interleaved_unitaries=(interleaver,),
    )
    record = resolve_component_program(
        _library(source, 1), 0, _factories() | cirq_gate_library_factories()
    )
    logical = width + 3
    bound = bind_register_program(
        record,
        ((0,), (1, 2), tuple(range(3, logical))),
        clean=tuple(range(logical, logical + 3)),
    )
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

    captured = import_register_emitter(source)
    expected = logical_operator(captured.snapshots[0], logical)
    if width == 2:
        with pytest.raises(ValueError, match="non-thru registers in C\\[And\\]"):
            source.tensor_contract()
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        expected,
        atol=2e-10,
        rtol=0,
    )
    assert bound.outputs == ((0,), (1, 2), tuple(range(3, logical)))


@pytest.mark.parametrize("values", (0, 1, (0, 1)))
def test_external_control_polarities_and_grouped_layout(
    values: Any, locked_source: Any
) -> None:
    from qualtran._infra.controlled import CtrlSpec

    gate = cirq.MatrixGate(exp(0.19j) * cirq.testing.random_unitary(2, random_state=17))
    source = locked_source.CirqGateAsBloq(gate).controlled(CtrlSpec(cvs=values))
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    inputs = []
    offset = 0
    for register in record.functions[0].registers or ():
        for _ in register.all_idxs():
            inputs.append(tuple(range(offset, offset + int(register.bitsize))))
            offset += int(register.bitsize)
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(offset, offset + 3))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, offset),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )


def test_invalid_payload_and_missing_unitary_fail_explicitly(
    locked_source: Any,
) -> None:
    factory = cirq_gate_library_factories()[
        "qualtran.cirq_interop._cirq_to_bloq.CirqGateAsBloq"
    ]
    for arguments in ({}, {"gate": 3}, {"gate": ExternalGatePayload(b"invalid")}):
        with pytest.raises(SerializationError, match="(payload|constructor)"):
            factory("main", None, arguments)
    source = locked_source.CirqGateAsBloq(cirq.ResetChannel())
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    assert not record.functions[0].blocks
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, ((0,),))


def test_nested_external_controls_retain_new_control_first(locked_source: Any) -> None:
    from qualtran._infra.controlled import Controlled, CtrlSpec

    gate = cirq.MatrixGate(exp(0.19j) * cirq.testing.random_unitary(2, random_state=17))
    source = Controlled(
        Controlled(locked_source.CirqGateAsBloq(gate), CtrlSpec(cvs=0)), CtrlSpec(cvs=1)
    )
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    bound = bind_register_program(record, ((0,), (1,), (2,)), clean=(3, 4, 5))
    expected = cirq.unitary(
        gate.controlled(control_values=(0,)).controlled(control_values=(1,))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, 3), torch.tensor(expected), atol=1e-11, rtol=0
    )


def test_other_controlled_children_delegate_to_their_existing_owner() -> None:
    from qualtran._infra.controlled import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates.hadamard import Hadamard

    from flagquantum.compiler.formats.legacy_controlled_factories import (
        controlled_via_and_factories,
    )
    from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
        fixed_gate_factories,
    )

    source = Controlled(Hadamard(), CtrlSpec())
    fallback = controlled_via_and_factories()[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]
    factories = fixed_gate_factories() | cirq_gate_library_factories(
        controlled_factory=fallback
    )
    record = resolve_component_program(_library(source, 0), 0, factories)
    bound = bind_register_program(record, ((0,), (1,)), clean=(2,))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )


@pytest.mark.parametrize("kind", ("cxpow", "swappow", "matrix1", "matrix2"))
@pytest.mark.parametrize("double", (False, True))
def test_explicit_adjoint_matches_original_graph_and_all_small_histories(
    kind: str,
    double: bool,
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    from collections import Counter
    from dataclasses import replace
    from itertools import product
    from math import pi, sqrt

    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates.global_phase import GlobalPhase

    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.ecosystem.cirq._named_gate_definitions import _NamedGateDefinitions
    from flagquantum.ecosystem.cirq._register_import import (
        _operation_ports,
        import_cirq_register_optree,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.register_source_graph import assert_source_wire_streams

    gates = {
        "cxpow": cirq.CXPowGate(exponent=-0.375),
        "swappow": cirq.SwapPowGate(exponent=0.375),
        "matrix1": cirq.MatrixGate(
            exp(0.19j) * cirq.testing.random_unitary(2, random_state=17)
        ),
        "matrix2": cirq.MatrixGate(
            exp(-0.27j) * cirq.testing.random_unitary(4, random_state=23)
        ),
    }
    gate = gates[kind]
    original = Adjoint(locked_source.CirqGateAsBloq(gate))
    source = Adjoint(original) if double else original
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from flagquantum.compiler._component_trace.qec_gate_cost import (
        QECGateCost,
        query_qec_gate_cost,
    )
    from flagquantum.ecosystem.cirq.resource_keys import cirq_mapped_gate_resource_key

    model = QECGateCost()
    from flagquantum.ecosystem.cirq._decompose import decompose_cirq_once

    operations = decompose_cirq_once(gate)
    assert operations is not None
    for operation in operations:
        key = cirq_mapped_gate_resource_key(operation.gate)
        assert key is not None
        model = model.plus(query_qec_gate_cost(key), 1)
    source_model = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(model, category) == getattr(source_model, category)
    library = _library(source, 0)
    record = resolve_component_program(library, 0, cirq_gate_library_factories())
    assert verify_program_record(record) == ()
    n = cirq.num_qubits(gate)
    signature = Signature(record.functions[0].registers or ())
    incoming, outgoing = _operation_ports(signature, tuple(cirq.LineQubit.range(n)))
    graph = source.decompose_bloq()
    scalar_angles = [
        pi * float(instance.bloq.exponent)
        for instance in graph.bloq_instances
        if isinstance(instance.bloq, GlobalPhase)
    ]
    assert len(scalar_angles) <= 1
    exported, returned = graph.to_cirq_circuit_and_quregs(
        qubit_manager=cirq.GreedyQubitManager("source"), **incoming
    )
    for name in outgoing:
        assert (returned[name] == outgoing[name]).all()
    expected_record = import_cirq_register_optree(
        exported,
        signature=signature,
        in_quregs=incoming,
        out_quregs=returned,
        gate_definitions=_NamedGateDefinitions(),
    )
    bindings = tuple((i,) for i in range(n))
    clean = tuple(range(n, n + 4))
    expected_bound = bind_register_program(expected_record, bindings, clean=clean)
    source_import = builtins.__import__
    blocked = {"qualtran"}

    def native_only(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in blocked:
            raise AssertionError("Serialized native gate execution imported source SDK")
        return source_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", native_only)
    replay = resolve_component_program(library, 0, cirq_gate_library_factories())
    assert replay == record
    bound = bind_register_program(replay, bindings, clean=clean)

    def phase_aliases(ir: CircuitIR) -> tuple[Instruction, ...]:
        # One-to-one opcode aliases only; no fusion, deletion or reordering.
        aliases = {"s": pi / 2, "sdg": -pi / 2, "t": pi / 4, "tdg": -pi / 4}
        return tuple(
            (
                replace(op, name="phase", params={"theta": aliases[op.name]})
                if op.name in aliases
                else op
            )
            for op in ir.instructions
        )

    def without_scalar_carrier(ir: CircuitIR) -> tuple[Instruction, ...]:
        normalized = phase_aliases(ir)
        if not scalar_angles:
            return normalized
        from flagquantum.compiler.registers.primitives.global_phase import (
            scalar_phase_ir,
        )

        positions = [i for i, op in enumerate(normalized) if op.wires == (n,)]
        candidates = [
            positions[i : i + 4]
            for i in range(len(positions) - 3)
            if [normalized[j].name for j in positions[i : i + 4]]
            == ["x", "phase", "x", "phase"]
        ]
        assert len(candidates) == 1
        carrier = candidates[0]
        assert_source_wire_streams(
            tuple(normalized[i] for i in carrier),
            scalar_phase_ir(n, scalar_angles[0]),
        )
        # The source scalar is a disconnected zero-port graph node. Its
        # frozenset tie has no portable order relative to the logical calls.
        # Compare its complete four-gate carrier separately, retaining its phase.
        return tuple(op for i, op in enumerate(normalized) if i not in carrier)

    assert_source_wire_streams(
        without_scalar_carrier(bound.circuit),
        without_scalar_carrier(expected_bound.circuit),
    )
    assert Counter(op.name for op in phase_aliases(bound.circuit)) == Counter(
        op.name for op in phase_aliases(expected_bound.circuit)
    )
    assert len(bound.circuit.instructions) == len(expected_bound.circuit.instructions)
    assert bound.outputs == bindings
    assert (
        Counter(op.name for op in bound.circuit.instructions)["measure"]
        == {
            "cxpow": 1,
            "swappow": 1,
            "matrix1": 0,
            "matrix2": 3,
        }[kind]
    )
    operator = torch.tensor(
        cirq.unitary(gate if double else cirq.inverse(gate)), dtype=torch.complex128
    )
    blocked.add("cirq")
    torch.testing.assert_close(
        logical_operator(bound.circuit, n), operator, atol=2e-11, rtol=0
    )
    count = sum(op.name == "measure" for op in bound.circuit.instructions)
    stride = 1 << (bound.circuit.n_wires - n)
    vector = torch.complex(
        torch.arange(1, 2**n + 1, dtype=torch.float64),
        torch.arange(2**n, 0, -1, dtype=torch.float64),
    )
    vector /= vector.norm()
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    initial[::stride] = vector
    target = torch.zeros_like(initial)
    target[::stride] = operator @ vector
    for history in product((0, 1), repeat=count):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, target / sqrt(2**count), atol=2e-11, rtol=0)
        assert torch.count_nonzero(actual.reshape(-1, stride)[:, 1:]) == 0
    circuit = Circuit(bound.circuit.n_wires, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(bound.circuit.n_wires, bound.circuit.instructions))
    restored = CircuitIR.from_json(circuit.to_ir().to_json())
    assert restored.instructions == bound.circuit.instructions
    for history in product((0, 1), repeat=count):
        actual = forced_branch(initial, restored.instructions, history)
        torch.testing.assert_close(actual, target / sqrt(2**count), atol=2e-11, rtol=0)
    if not count:
        torch.testing.assert_close(
            circuit.state().flatten(), target, atol=2e-11, rtol=0
        )


@pytest.mark.parametrize("kind", ("zpow", "scalar", "reset"))
@pytest.mark.parametrize("double", (False, True))
def test_explicit_source_bodyless_adjoints_remain_declarations(
    kind: str, double: bool, locked_source: Any
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran._infra.bloq import DecomposeNotImplementedError

    gates = {
        "zpow": cirq.ZPowGate(exponent=0.375),
        "scalar": cirq.GlobalPhaseGate(1j),
        "reset": cirq.ResetChannel(),
    }
    source = Adjoint(locked_source.CirqGateAsBloq(gates[kind]))
    if double:
        source = Adjoint(source)
    with pytest.raises(DecomposeNotImplementedError, match="decomposition"):
        source.decompose_bloq()
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    assert not record.functions[0].blocks
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(
            record, tuple((i,) for i in range(cirq.num_qubits(gates[kind])))
        )


def test_external_adjoint_validation_and_independent_owner_delegation() -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates.s_gate import SGate

    from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
        fixed_gate_factories,
    )
    from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register

    fixed = fixed_gate_factories()
    calls = []

    def fallback(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        calls.append(arguments)
        inverse = fixed["qualtran.bloqs.basic_gates.s_gate.SGate"](
            name, registers, {"is_adjoint": True}
        )
        assert isinstance(inverse, FunctionRecord)
        return inverse

    factories = fixed | cirq_gate_library_factories(adjoint_factory=fallback)
    record = resolve_component_program(_library(Adjoint(SGate()), 0), 0, factories)
    bound = bind_register_program(record, ((0,),))
    assert [op.name for op in bound.circuit.instructions] == ["sdg"]
    assert len(calls) == 1
    factory = cirq_gate_library_factories()["qualtran._infra.adjoint.Adjoint"]
    with pytest.raises(SerializationError, match="exactly one subbloq"):
        factory("main", None, {})
    with pytest.raises(SerializationError, match="constructor context"):
        factory("main", None, {"subbloq": record.functions[0]})
    mapping = cirq_gate_library_factories()
    forward = mapping["qualtran.cirq_interop._cirq_to_bloq.CirqGateAsBloq"](
        "child", None, {"gate": ExternalGatePayload(cirq.to_json_gzip(cirq.X))}
    )
    child = forward.functions[0] if isinstance(forward, ProgramRecord) else forward
    with pytest.raises(SerializationError, match="signature differs"):
        mapping["qualtran._infra.adjoint.Adjoint"](
            "main", (Register("other", quantum_bit_type()),), {"subbloq": child}
        )


@pytest.mark.parametrize("kind", ("cxpow", "swappow", "scalar"))
def test_generic_control_wrappers_do_not_borrow_raw_sdk_adjoint_recipes(
    kind: str, locked_source: Any
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran._infra.controlled import Controlled, CtrlSpec

    gate = {
        "cxpow": cirq.CXPowGate(exponent=-0.375),
        "swappow": cirq.SwapPowGate(exponent=0.375),
        "scalar": cirq.GlobalPhaseGate(1j),
    }[kind]
    source = Adjoint(Controlled(locked_source.CirqGateAsBloq(gate), CtrlSpec(cvs=0)))
    with pytest.raises(
        SerializationError, match="exact adjoint owner.*control-wrapper"
    ):
        resolve_component_program(_library(source, 0), 0, cirq_gate_library_factories())


def test_raw_controlled_gate_has_its_own_original_adjoint_decomposition(
    locked_source: Any,
) -> None:
    from qualtran._infra.adjoint import Adjoint

    raw = cirq.ControlledGate(cirq.GlobalPhaseGate(1j), control_values=(0,))
    source = Adjoint(locked_source.CirqGateAsBloq(raw))
    assert tuple(source.decompose_bloq().iter_bloqnections())
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    bound = bind_register_program(record, ((0,),), clean=(1, 2))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 1),
        torch.tensor(cirq.unitary(cirq.inverse(raw)), dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
