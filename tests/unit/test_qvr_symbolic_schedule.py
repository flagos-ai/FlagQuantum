"""Source-positive expression QVR schedules retain literal branch semantics."""

from importlib import import_module
from itertools import product
from math import pi
from typing import Any

from tests._reference_dependencies import unavailable_reference

# Typed-register owners require NumPy before their modules can be collected.
try:
    import_module("numpy")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_qvr_factories import qvr_factories
from flagquantum.compiler.registers.phase.qvr_zpow import symbolic_qvr_zpow_schedule
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FORMATS = ((2, 0, False), (2, 2, False), (3, 2, False), (2, 0, True), (3, 1, True))
COEFFICIENTS = (
    sympy.Integer(1),
    sympy.Rational(1, 4),
    sympy.Float(0.375),
    sympy.Rational(-3, 8),
)


def _source_phase_contract(
    source: Any, wires: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Evaluate original Z-power contracts, not a successful SDK unitary.

    Cirq retains constant SymPy exponents as parameterized. The unmodified
    source graph and each original scalar exponent are still available.
    """
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    raw = cirq.Circuit(
        cirq.decompose(
            source.on(*(cirq.LineQubit(w) for w in wires)),
            keep=lambda op: isinstance(op.gate, cirq.ZPowGate),
        )
    )
    gates = []
    for operation in raw.all_operations():
        assert isinstance(operation.gate, cirq.ZPowGate)
        assert operation.gate.global_shift == 0
        qubit = operation.qubits[0]
        assert isinstance(qubit, cirq.LineQubit)
        gates.append(
            Instruction(
                "phase",
                (qubit.x,),
                params={"theta": pi * float(operation.gate.exponent)},
            )
        )
    assert cirq.unitary(raw, default=None) is None
    return tuple(gates)


@pytest.mark.parametrize(
    "format_,gamma,eps",
    tuple(product(FORMATS, COEFFICIENTS, (0.2, sympy.Rational(1, 5))))
    + tuple((format_, 0.375, sympy.Rational(1, 5)) for format_ in FORMATS),
)
def test_literal_source_calls_gates_complex_append_and_no_workspace(
    format_: tuple[int, int, bool], gamma: float | sympy.Expr, eps: float | sympy.Expr
) -> None:
    try:
        from qualtran._infra.data_types import QFxp
        from qualtran._infra.registers import Register as SourceRegister
        from qualtran.bloqs.basic_gates.rotation import ZPowGate
        from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    width, fraction, signed = format_
    source = QvrZPow(
        SourceRegister("energy", QFxp(width, fraction, signed=signed)), gamma, eps
    )
    dtype = fixed_point_register_type(width, fraction, signed=signed)
    factory = qvr_factories()[
        "qualtran.bloqs.rotations.quantum_variable_rotation.QvrZPow"
    ]
    record = factory(
        "main",
        None,
        {"cost_reg": Register("energy", dtype), "gamma": gamma, "eps": eps},
    )
    assert isinstance(record, ProgramRecord)
    assert verify_program_record(record) == ()
    composite = source.decompose_bloq()
    assert _native_trace(record) == _source_trace(SourceCalls, composite)
    phases = [
        instance.bloq
        for instance, _, _ in composite.iter_bloqsoqs()
        if isinstance(instance.bloq, ZPowGate)
    ]
    schedule = symbolic_qvr_zpow_schedule(
        tuple(range(width)),
        cost_fraction_bits=fraction,
        coefficient=gamma,
        epsilon=eps,
        signed=signed,
    )
    assert len(phases) == len(schedule)
    for phase, (_, exponent, tolerance) in zip(phases, schedule, strict=True):
        assert sympy.simplify(phase.exponent - exponent).is_zero is True
        assert sympy.simplify(phase.eps - tolerance).is_zero is True
    wires = tuple(reversed(range(width)))
    bound = bind_register_program(record, (wires,))
    expected = _source_phase_contract(source, wires)
    assert_source(bound.circuit.instructions, expected)
    assert len(bound.circuit.instructions) == width + int(signed)
    assert all(gate.name == "phase" for gate in bound.circuit.instructions)
    assert bound.outputs == (wires,)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    raw = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    initial = torch.complex(raw, 2 - 3 * raw)
    initial /= initial.norm()
    actual = (
        Circuit(width + 1, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(width + 1, bound.circuit.instructions))
        .state()
    )
    reference = (
        Circuit(width + 1, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(width + 1, expected))
        .state()
    )
    torch.testing.assert_close(actual, reference, atol=2e-12, rtol=0)
    restored = (
        Circuit(width + 1, inputs=actual, dtype=torch.complex128)
        .append(
            FieldOperation(
                width + 1,
                _source_phase_contract(source.adjoint(), wires),
            )
        )
        .state()
    )
    torch.testing.assert_close(restored.flatten(), initial, atol=2e-12, rtol=0)


def test_symbolic_coefficient_with_finite_source_indices_binds_exactly() -> None:
    try:
        from qualtran._infra.data_types import QFxp
        from qualtran._infra.registers import Register as SourceRegister
        from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    gamma = sympy.Symbol("gamma", positive=True)
    eps = sympy.pi * gamma / 8
    source = QvrZPow(SourceRegister("energy", QFxp(3, 2)), gamma, eps)
    factory = qvr_factories()[
        "qualtran.bloqs.rotations.quantum_variable_rotation.QvrZPow"
    ]
    record = factory(
        "main",
        None,
        {
            "cost_reg": Register("energy", fixed_point_register_type(3, 2)),
            "gamma": gamma,
            "eps": eps,
        },
    )
    assert isinstance(record, ProgramRecord)
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    bound = bind_register_program(record, ((2, 0, 1),))
    numeric = (
        Circuit(3)
        .append(FieldOperation(3, bound.circuit.instructions))
        .bind_parameters({"gamma": 0.25})
        .to_ir()
    )
    resolved = QvrZPow(
        source.cost_reg, sympy.Rational(1, 4), eps.subs(gamma, sympy.Rational(1, 4))
    )
    expected = _source_phase_contract(resolved, (2, 0, 1))
    assert_source(numeric.instructions, expected)


@pytest.mark.parametrize(
    "fraction", (sympy.Integer(2), sympy.Symbol("f", negative=True, integer=True))
)
def test_raw_storage_approval_does_not_replace_source_symbolic_split_refusal(
    fraction: sympy.Expr,
) -> None:
    try:
        from qualtran._infra.data_types import QFxp
        from qualtran._infra.registers import Register as SourceRegister
        from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    source = QvrZPow(SourceRegister("energy", QFxp(3, fraction)), 0.375, 0.2)
    cost = Register("energy", fixed_point_register_type(3, fraction))
    factory = qvr_factories()[
        "qualtran.bloqs.rotations.quantum_variable_rotation.QvrZPow"
    ]
    record = factory("main", None, {"cost_reg": cost, "gamma": 0.375, "eps": 0.2})
    assert isinstance(record, FunctionRecord) and not record.blocks
    assert record.registers == (cost,)
    with pytest.raises(ValueError, match="symbolic data type"):
        source.decompose_bloq()


def test_original_unresolved_and_out_of_range_indices_are_not_invented() -> None:
    try:
        from qualtran._infra.data_types import QFxp
        from qualtran._infra.registers import Register as SourceRegister
        from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    for gamma in (sympy.Symbol("g"), sympy.Rational(1, 100000)):
        source = QvrZPow(SourceRegister("energy", QFxp(3, 2)), gamma, 0.2)
        with pytest.raises(IndexError):
            source.decompose_bloq()
        with pytest.raises(ValueError, match="source rotation"):
            symbolic_qvr_zpow_schedule(
                (2, 0, 1), cost_fraction_bits=2, coefficient=gamma, epsilon=0.2
            )


@pytest.mark.parametrize("format_", ((2, 0, True), (3, 2, False)))
def test_actual_lossless_rational_library_reaches_the_finite_body(
    format_: tuple[int, int, bool],
) -> None:
    try:
        from qualtran._infra.data_types import QFxp
        from qualtran._infra.registers import Register as SourceRegister
        from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    width, fraction, signed = format_
    source = QvrZPow(
        SourceRegister("energy", QFxp(width, fraction, signed=signed)),
        sympy.Rational(1, 4),
        sympy.Rational(1, 5),
    )
    records = tuple(
        resolve_component_program(_library(source, depth), 0, qvr_factories())
        for depth in (0, 1)
    )
    assert records[0] == records[1]
    assert _native_trace(records[0]) == _source_trace(
        SourceCalls, source.decompose_bloq()
    )
    wires = tuple(reversed(range(width)))
    bound = bind_register_program(records[0], (wires,))
    assert_source(bound.circuit.instructions, _source_phase_contract(source, wires))


def test_native_expression_constructor_does_not_import_source_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError("Source SDK imported during owned QVR execution")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    factory = qvr_factories()[
        "qualtran.bloqs.rotations.quantum_variable_rotation.QvrZPow"
    ]
    record = factory(
        "main",
        None,
        {
            "cost_reg": Register(
                "energy", fixed_point_register_type(2, 0, signed=True)
            ),
            "gamma": sympy.Integer(1),
            "eps": sympy.Rational(1, 5),
        },
    )
    assert isinstance(record, ProgramRecord)
    bound = bind_register_program(record, ((0, 1),))
    assert (
        len(
            Circuit(2)
            .append(FieldOperation(2, bound.circuit.instructions))
            .to_ir()
            .instructions
        )
        == 3
    )
