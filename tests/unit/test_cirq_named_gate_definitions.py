"""Named SDK children select the locked native gate component hierarchy."""

from itertools import product
from math import pi, sqrt
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_z_power_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._named_gate_definitions import _NamedGateDefinitions
from flagquantum.ecosystem.cirq._register_definition import _native_definition_module
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("locked_source"),
]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")
SDKGate: Any = cirq.Gate


def _gates() -> tuple[Any, ...]:
    return (
        cirq.T,
        cirq.T**-1,
        cirq.S,
        cirq.S**-1,
        cirq.H,
        cirq.ControlledGate(cirq.H),
        cirq.CNOT,
        cirq.TOFFOLI,
        cirq.X,
        cirq.Y,
        cirq.ControlledGate(cirq.Y),
        cirq.Z,
        cirq.CZ,
        cirq.SWAP,
        cirq.ControlledGate(cirq.SWAP),
        cirq.CSWAP,
        cirq.I,
        cirq.rx(0.37),
        cirq.ry(-0.24),
        cirq.rz(0.19),
        cirq.XPowGate(exponent=0.37),
        cirq.YPowGate(exponent=-0.24),
        cirq.ZPowGate(exponent=0.19),
        cirq.CZPowGate(exponent=0.375),
        cirq.GlobalPhaseGate(complex(0.6, 0.8)),
        cirq.ControlledGate(cirq.ZPowGate(exponent=0.37)),
        cirq.ControlledGate(cirq.rz(0.37)),
        cirq.ControlledGate(cirq.GlobalPhaseGate(complex(0.6, 0.8))),
    )


@pytest.mark.parametrize("gate", _gates())
def test_ordered_named_selection_complex_operator(
    gate: Any, locked_source: Any
) -> None:
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    definitions = _NamedGateDefinitions()
    root = definitions[gate]
    source = cirq_gate_to_bloq(gate)
    assert root.registers is not None
    ports = Signature(root.registers)
    assert tuple((r.name, r.shape, int(r.dtype.num_qubits)) for r in ports) == tuple(
        (r.name, r.shape, int(r.bitsize)) for r in source.signature
    )
    width = cirq.num_qubits(gate)
    program = ProgramRecord(tuple(dict.fromkeys(definitions.values())))
    bound = bind_register_program(
        program,
        tuple((i,) for i in range(width)),
        clean=tuple(range(width, width + 3)),
        entry=root.name,
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, width),
        torch.tensor(cirq.unitary(gate), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    assert definitions[gate] is root


def test_decomposition_child_retains_named_compute_phase_erasure() -> None:
    class Outer(SDKGate):  # type: ignore[misc]
        def _num_qubits_(self) -> int:
            return 2

        def _decompose_(self, qubits: Any) -> Any:
            return (cirq.CZPowGate(exponent=0.375).on(*reversed(qubits)),)

    gate = Outer()
    definitions = _NamedGateDefinitions()
    program = import_cirq_register_gate(gate, gate_definitions=definitions)
    assert program is not None
    bound = bind_register_program(program, ((0,), (1,)), clean=(2,))
    expected = bind_register_program(
        controlled_z_power_program(0.375), ((1,), (0,)), clean=(2,)
    )
    assert bound.circuit == expected.circuit
    from qualtran.bloqs.basic_gates.rotation import CZPowGate

    from tests.unit.register_source_graph import assert_source_wire_streams
    from tests.unit.test_register_phase_gradient import source_gradient_gates

    reference, phases = source_gradient_gates(
        CZPowGate(exponent=0.375), {"q": (1, 0)}, {"q": (1, 0)}, work=2
    )
    assert_source_wire_streams(bound.circuit.instructions, reference)
    assert len(phases) == 1 and phases[0].exponent == 0.375
    assert bound.events == expected.events
    assert all(op.name != "cphase" for op in bound.circuit.instructions)
    amplitudes = torch.complex(
        torch.arange(1, 5, dtype=torch.float64),
        torch.tensor([0.4, -0.7, 1.3, 0.2], dtype=torch.float64),
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = amplitudes
    reference = (Instruction("cphase", (1, 0), params={"theta": pi * 0.375}),)
    for history in product((0, 1), repeat=1):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, forced_branch(initial, reference, ()) / sqrt(2), atol=1e-11, rtol=0
        )
        assert torch.count_nonzero(actual[1::2]) == 0


def test_shifted_unknown_and_generic_control_selection_remains_explicit() -> None:
    definitions = _NamedGateDefinitions()
    for gate in (
        cirq.XPowGate(exponent=0.37, global_shift=0.2),
        cirq.MatrixGate(np.array([[0, 1], [1, 0]])),
    ):
        assert definitions.get(gate) is None
    assert len(definitions) == 0


@pytest.mark.parametrize("kind", ("h", "y", "zpow", "rz", "scalar"))
@pytest.mark.parametrize("values", ((0,), (0, 1), (1, 0, 1)))
def test_named_predicate_controls_complex_branches_and_source(
    kind: str, values: tuple[int, ...]
) -> None:
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from tests.unit.test_legacy_controlled_factories import _check_state
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    child = {
        "h": cirq.H,
        "y": cirq.Y,
        "zpow": cirq.ZPowGate(exponent=0.375),
        "rz": cirq.rz(0.37),
        "scalar": cirq.GlobalPhaseGate(complex(0.6, 0.8)),
    }[kind]
    gate = cirq.ControlledGate(child, num_controls=len(values), control_values=values)
    definitions = _NamedGateDefinitions()
    root = definitions[gate]
    program = ProgramRecord(
        tuple(_native_definition_module((root,), definitions).values())
    )
    source = cirq_gate_to_bloq(gate)
    assert root.registers is not None
    width = cirq.num_qubits(gate)
    bound = bind_register_program(
        program,
        tuple((i,) for i in range(width)),
        clean=tuple(range(width, width + len(values) + 1)),
        entry=root.name,
    )
    assert tuple((r.name, r.shape) for r in root.registers) == tuple(
        (r.name, r.shape) for r in source.signature
    )
    work = bound.circuit.n_wires - width
    _check_state(source, bound.circuit.instructions, width, work)
    model = get_cost_value(source, QECGatesCost())
    measured = sum(op.name == "measure" for op in bound.circuit.instructions)
    assert measured == model.measurement
    if kind in ("h", "y"):
        offset = 0
        ports = {}
        for register in source.signature:
            n = int(register.total_bits())
            ports[register.name] = tuple(range(offset, offset + n))
            offset += n
        expected = reference_arithmetic_gates(
            source,
            ports,
            tuple(range(width, width + work)),
            MeasurementSlots(bound.circuit.instructions),
            keep_pauli_leaves=True,
        )
        assert_source(bound.circuit.instructions, expected)


def test_disjunctive_named_control_keeps_original_refusal() -> None:
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    gate = cirq.ControlledGate(
        cirq.H, num_controls=2, control_values=cirq.SumOfProducts(((0, 1), (1, 0)))
    )
    for convert in (cirq_gate_to_bloq, _NamedGateDefinitions().__getitem__):
        with pytest.raises(ValueError, match="single AND clause"):
            convert(gate)


def test_inverse_wrapper_uses_exact_measured_component_pairs() -> None:
    class Outer(SDKGate):  # type: ignore[misc]
        def _num_qubits_(self) -> int:
            return 2

        def _decompose_(self, qubits: Any) -> Any:
            return (
                cirq.H(qubits[0]),
                cirq.CZPowGate(exponent=0.375).on(*reversed(qubits)),
                cirq.rz(0.19)(qubits[1]),
            )

    original = Outer()
    inverse = cirq.inverse(original)
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    source = cirq_gate_to_bloq(inverse)
    from qualtran._infra.adjoint import Adjoint
    from qualtran.cirq_interop._cirq_to_bloq import CirqGateAsBloq

    assert isinstance(source, Adjoint)
    assert isinstance(source.subbloq, CirqGateAsBloq)
    assert source.subbloq.gate is original
    definitions = _NamedGateDefinitions()
    root = definitions[inverse]
    program = definitions._module(root)
    bound = bind_register_program(program, ((0,), (1,)), clean=(2,), entry=root.name)
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.tensor(cirq.unitary(inverse)),
        atol=1e-11,
        rtol=0,
    )
    assert sum(op.name == "measure" for op in bound.circuit.instructions) == 1
    amplitudes = torch.complex(
        torch.arange(1, 5, dtype=torch.float64), torch.arange(5, 9, dtype=torch.float64)
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = amplitudes
    expected = torch.zeros_like(initial)
    expected[::2] = torch.tensor(cirq.unitary(inverse)) @ amplitudes
    for history in ((0,), (1,)):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, expected / sqrt(2), atol=1e-11, rtol=0)
        assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("controlled", (False, True))
def test_symbolic_scalar_coefficients_keep_complex_phase(controlled: bool) -> None:
    import sympy

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit

    theta = sympy.Symbol("theta")
    gate = cirq.GlobalPhaseGate(sympy.exp(sympy.I * (sympy.sin(theta) + theta**2 / 3)))
    if controlled:
        gate = cirq.ControlledGate(gate)
    definitions = _NamedGateDefinitions()
    root = definitions[gate]
    program = definitions._module(root)
    width = int(controlled)
    bound = bind_register_program(
        program,
        ((0,),) if controlled else (),
        clean=() if controlled else (0,),
        entry=root.name,
    )
    circuit = Circuit(1, dtype=torch.complex128).append(
        FieldOperation(1, bound.circuit.instructions)
    )
    from flagquantum.simulation.unitary import get_unitary

    for value in (-7.0, -0.3, 0.0, 4.5):
        actual = get_unitary(circuit.bind_parameters({"theta": value}))
        expected = torch.tensor(
            cirq.unitary(cirq.resolve_parameters(gate, {"theta": value})),
            dtype=torch.complex128,
        )
        if width == 0:
            expected = expected.reshape(()) * torch.eye(2, dtype=torch.complex128)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


def test_atomic_control_source_body_absence_stays_distinct_from_sdk_execution() -> None:
    from qualtran._infra.bloq import DecomposeTypeError
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    gate = cirq.ControlledGate(cirq.XPowGate(exponent=0.37))
    source = cirq_gate_to_bloq(gate)
    with pytest.raises(DecomposeTypeError, match="atomic"):
        source.decompose_bloq()
    assert _NamedGateDefinitions().get(gate) is None


def test_attached_native_atom_passes_through_before_sdk_semantics() -> None:
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.core.ir import CircuitIR
    from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree

    class Native(SDKGate):  # type: ignore[misc]
        flagquantum_register_definition = bit_gate_function(
            "attached",
            tuple(Signature.build(q=1)),
            CircuitIR(1, (Instruction("h", (0,)),)),
        )

        def _num_qubits_(self) -> int:
            return 1

        def _unitary_(self) -> Any:
            return np.eye(2)

    program = import_cirq_register_optree(
        cirq.Circuit(Native().on(cirq.LineQubit(0))),
        gate_definitions=_NamedGateDefinitions(),
    )
    bound = bind_register_program(program, ((0,),))
    assert bound.circuit.instructions == (Instruction("h", (0,)),)


@pytest.mark.parametrize("child", (cirq.H, cirq.ZPowGate(exponent=0.375)))
def test_nested_named_controls_keep_new_control_first(child: Any) -> None:
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    from tests.unit.test_legacy_controlled_factories import _check_state

    gate = cirq.ControlledGate(
        cirq.ControlledGate(child, control_values=(0,)), control_values=(1,)
    )
    definitions = _NamedGateDefinitions()
    root = definitions[gate]
    bound = bind_register_program(
        definitions._module(root), ((0,), (1,), (2,)), clean=(3, 4, 5), entry=root.name
    )
    _check_state(
        cirq_gate_to_bloq(gate),
        bound.circuit.instructions,
        3,
        bound.circuit.n_wires - 3,
    )


def test_inverse_wrapper_refuses_measurement_without_exact_component_pair() -> None:
    class Measured(SDKGate):  # type: ignore[misc]
        def _num_qubits_(self) -> int:
            return 1

        def _decompose_(self, qubits: Any) -> Any:
            return (cirq.measure(qubits[0], key="unowned"),)

    with pytest.raises(TypeError, match="isn't invertible"):
        cirq.inverse(Measured())
    unchecked = cirq.ops.raw_types._InverseCompositeGate(Measured())
    with pytest.raises(ValueError, match="Missing exact component adjoint"):
        _NamedGateDefinitions()[unchecked]


def test_exported_source_graph_replay_keeps_replacement_bodies_local(
    locked_source: Any,
) -> None:
    from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree

    gate = cirq.CXPowGate(exponent=-0.375)
    source = locked_source.CirqGateAsBloq(gate)
    qubits = tuple(cirq.LineQubit.range(2))
    incoming = {"q": np.asarray(qubits, dtype=object).reshape(2, 1)}
    exported, returned = source.decompose_bloq().to_cirq_circuit_and_quregs(
        qubit_manager=cirq.GreedyQubitManager("source"), **incoming
    )
    np.testing.assert_array_equal(returned["q"], incoming["q"])
    signature = Signature([Register("q", quantum_bit_type(), shape=(2,))])
    program = import_cirq_register_optree(
        exported,
        signature=signature,
        in_quregs=incoming,
        out_quregs=returned,
        gate_definitions=_NamedGateDefinitions(),
    )
    bound = bind_register_program(program, ((0,), (1,)), clean=(2,))
    assert sum(op.name == "measure" for op in bound.circuit.instructions) == 1
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.tensor(cirq.unitary(gate), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    initial = torch.complex(
        torch.arange(1, 9, dtype=torch.float64),
        torch.arange(8, 0, -1, dtype=torch.float64),
    )
    initial[1::2] = 0
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    expected[::2] = torch.tensor(cirq.unitary(gate)) @ initial[::2]
    for outcome in (0, 1):
        result = forced_branch(initial, bound.circuit.instructions, (outcome,))
        torch.testing.assert_close(result, expected / sqrt(2), atol=1e-11, rtol=0)
        assert torch.count_nonzero(result[1::2]) == 0
