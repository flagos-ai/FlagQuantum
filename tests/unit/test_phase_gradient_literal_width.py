"""Concrete symbolic unitary widths retain the original direct SDK recipe."""

from collections import Counter
from dataclasses import fields
from itertools import product
from math import pi
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import cirq
    import numpy as np
    import sympy
    import torch
    from qualtran.bloqs.basic_gates.rotation import CZPowGate, ZPowGate
    from qualtran.bloqs.rotations.phase_gradient import (
        PhaseGradientState,
        PhaseGradientUnitary,
    )
    from qualtran.cirq_interop._bloq_to_cirq import BloqAsCirqGate
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_phase_gradient_factories import (
    phase_gradient_factories,
)
from flagquantum.compiler.registers.phase.phase_gradient import (
    source_symbolic_phase_gradient_cost,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,controlled,exponent",
    tuple(product((1, 2, 3), (False, True), (0.37, -0.375))),
)
def test_literal_unitary_sdk_stream_complex_branches_and_exact_inverse(
    width: int, controlled: bool, exponent: float
) -> None:
    source = PhaseGradientUnitary(sympy.Integer(width), exponent, controlled)
    with pytest.raises(TypeError):
        source.decompose_bloq()
    factory = phase_gradient_factories()[
        "qualtran.bloqs.rotations.phase_gradient.PhaseGradientUnitary"
    ]
    arguments = {
        "bitsize": sympy.Integer(width),
        "exponent": exponent,
        "is_controlled": controlled,
    }
    program = factory("main", None, arguments)
    assert isinstance(program, ProgramRecord) and verify_program_record(program) == ()
    assert program == factory("main", None, {**arguments, "bitsize": width})
    logical = width + controlled
    inputs = ((0,), tuple(range(1, logical))) if controlled else (tuple(range(width)),)
    clean = (logical,) if controlled else ()
    bound = bind_register_program(program, inputs, clean=clean)
    ports = dict(
        zip(
            ("ctrl", "phase_grad") if controlled else ("phase_grad",),
            inputs,
            strict=True,
        )
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    quregs = {
        name: np.array([cirq.LineQubit(wire) for wire in wires])
        for name, wires in ports.items()
    }
    gates = tuple(
        cirq.flatten_op_tree(
            source.decompose_from_registers(
                context=cirq.DecompositionContext(cirq.SimpleQubitManager()), **quregs
            )
        )
    )
    phases = tuple(gate for gate in bound.circuit.instructions if gate.name == "phase")
    assert len(gates) == len(phases) == width
    tensor_circuit = cirq.Circuit()
    source_stream: list[Instruction] = []
    for index, (operation, phase_gate) in enumerate(zip(gates, phases, strict=True)):
        assert isinstance(operation, cirq.Operation)
        assert isinstance(operation.gate, BloqAsCirqGate)
        leaf = operation.gate.bloq
        assert isinstance(leaf, (ZPowGate, CZPowGate))
        tensor_circuit.append(
            cirq.MatrixGate(leaf.tensor_contract()).on(*operation.qubits)
        )
        assert type(leaf).__name__ == ("CZPowGate" if controlled else "ZPowGate")
        assert leaf.exponent == exponent / 2**index
        assert leaf.eps == 1e-10 / sympy.Integer(width)
        leaf_ports: dict[str, tuple[int, ...]] = {
            "q": ((0, index + 1) if controlled else (index,))
        }
        leaf_stream, _ = source_gradient_gates(
            leaf,
            leaf_ports,
            leaf_ports,
            work=logical if controlled else None,
            bit=index,
        )
        source_stream.extend(leaf_stream)
        wires = (0, index + 1) if controlled else (index,)
        assert operation.qubits == tuple(cirq.LineQubit(wire) for wire in wires)
        assert phase_gate.wires == ((logical,) if controlled else (index,))
        assert phase_gate.params["theta"] == pytest.approx(pi * float(leaf.exponent))
    assert_source(bound.circuit.instructions, tuple(source_stream))
    if controlled:
        # The original measured CZ recipe has no Cirq unitary protocol. Its
        # unchanged tensor contract provides the independent complete leaf map.
        with pytest.raises(TypeError):
            cirq.Circuit(gates).unitary(qubit_order=cirq.LineQubit.range(logical))
    unitary = torch.tensor(
        tensor_circuit.unitary(qubit_order=cirq.LineQubit.range(logical)),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), unitary, atol=8e-12, rtol=0
    )
    # A coherent spectator occupies the last logical wire; clean work follows it.
    clean = (logical + 1,) if controlled else ()
    forward = bind_register_program(program, inputs, clean=clean)
    inverse = factory("main", None, {**arguments, "exponent": -exponent})
    assert isinstance(inverse, ProgramRecord)
    backward = bind_register_program(inverse, inputs, clean=clean)
    total = logical + 1 + controlled
    raw = torch.arange(1, 2 ** (logical + 1) + 1, dtype=torch.float64)
    state = torch.complex(raw, 2 - 3 * raw)
    state /= state.norm()
    evolved = torch.kron(unitary, torch.eye(2, dtype=torch.complex128)) @ state
    stride = 2 if controlled else 1
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    final = torch.zeros_like(initial)
    initial[::stride], final[::stride] = state, evolved
    for bound_program, start, target in (
        (forward, initial, final),
        (backward, final, initial),
    ):
        circuit = DynamicCircuit(total).append(
            FieldOperation(total, bound_program.circuit.instructions)
        )
        count = sum(gate.name == "measure" for gate in circuit.to_ir().instructions)
        assert count == width * controlled
        for history in product((0, 1), repeat=count):
            result = forced_branch(start, circuit.to_ir().instructions, history)
            torch.testing.assert_close(
                result, target / 2 ** (count / 2), atol=8e-12, rtol=0
            )
            if controlled:
                assert torch.count_nonzero(result[1::stride]) == 0
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    assert counts["phase"] == width
    assert counts["measure"] == width * controlled
    model = source_symbolic_phase_gradient_cost(
        sympy.Integer(width), controlled=controlled
    )
    original = get_cost_value(source, QECGatesCost())
    for field in fields(model):
        if field.name != "logical_qubits":
            assert getattr(model, field.name) == getattr(original, field.name)
    assert model.logical_qubits == get_cost_value(source, QubitCount())


def test_state_and_unresolved_widths_keep_source_refusals() -> None:
    factories = phase_gradient_factories()
    for name, width in (
        ("PhaseGradientState", sympy.Integer(3)),
        ("PhaseGradientState", sympy.Symbol("n")),
        ("PhaseGradientUnitary", sympy.Symbol("n")),
    ):
        result = factories["qualtran.bloqs.rotations.phase_gradient." + name](
            "main", None, {"bitsize": width}
        )
        assert isinstance(result, FunctionRecord) and not result.blocks
    with pytest.raises(ValueError, match="Symbolic Bitsize"):
        list(
            PhaseGradientState(sympy.Integer(3)).decompose_from_registers(
                context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                phase_grad=cirq.LineQubit.range(3),
            )
        )


def test_literal_native_constructor_append_is_sdk_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError("Source SDK imported during native gradient execution")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    factory = phase_gradient_factories()[
        "qualtran.bloqs.rotations.phase_gradient.PhaseGradientUnitary"
    ]
    record = factory("main", None, {"bitsize": sympy.Integer(3)})
    assert isinstance(record, ProgramRecord)
    bound = bind_register_program(record, ((2, 0, 1),))
    circuit = DynamicCircuit(3).append(FieldOperation(3, bound.circuit.instructions))
    assert circuit.to_ir().instructions == bound.circuit.instructions
