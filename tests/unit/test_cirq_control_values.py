"""Exercise full typed control-value transport and source decoder boundaries."""

from itertools import product

import pytest

from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem.cirq._control_values import (
    export_control_values,
    import_control_values,
)

try:
    from tests.unit.test_classical_register_codec import KINDS, pair
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.importorskip("numpy")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")
q = pytest.importorskip("qualtran")


QUANTUM = [kind for kind in KINDS if not kind.startswith("c")]


@pytest.mark.parametrize("kind", QUANTUM)
@pytest.mark.parametrize("shape", [(), (2,), (1, 2)])
def test_complete_small_domains_and_shaped_single_register_roundtrip(
    kind: str, shape: tuple[int, ...]
) -> None:
    dtype, source = pair(kind)
    registers = (Register("enable", dtype, shape),)
    for word in (range(4) if kind == "opaque" else source.get_classical_domain()):
        values = (np.full(shape, word),)
        original = q.CtrlSpec(source, values[0])
        exported = export_control_values(registers, values)
        assert exported == original.to_cirq_cv()
        returned, actual = import_control_values(exported, registers=registers)
        expected = q.CtrlSpec.from_cirq_cv(exported, qdtypes=(source,), shapes=(shape,))
        assert returned is registers
        np.testing.assert_array_equal(actual[0], expected.cvs[0])
        assert export_control_values(returned, actual) == exported


@pytest.mark.parametrize("bits", list(product((0, 1), repeat=4)))
def test_default_layout_matches_original(bits: tuple[int, ...]) -> None:
    for clause in (bits[:1], bits):
        exported = cirq.ProductOfSums(clause)
        registers, actual = import_control_values(exported)
        expected = q.CtrlSpec.from_cirq_cv(exported)
        assert registers[0].shape == expected.shapes[0]
        np.testing.assert_array_equal(actual[0], expected.cvs[0])
        assert export_control_values(registers, actual) == exported.expand()


def test_multi_register_consumption_preserves_each_slice_and_source_bug() -> None:
    unsigned, qu = pair("uint")
    signed, qs = pair("signed")
    registers = (Register("first", unsigned), Register("second", signed, (2,)))
    for values in product(range(4), range(-2, 2), range(-2, 2)):
        arrays = (np.array(values[0]), np.array(values[1:]))
        original = q.CtrlSpec((qu, qs), arrays)
        exported = export_control_values(registers, arrays)
        assert exported == original.to_cirq_cv()
        returned, actual = import_control_values(exported, registers=registers)
        assert returned is registers
        for lhs, rhs in zip(actual, arrays, strict=True):
            np.testing.assert_array_equal(lhs, rhs)
        assert export_control_values(returned, actual) == exported
    clause = cirq.ProductOfSums([0, 1])
    source = q.CtrlSpec.from_cirq_cv(
        clause, qdtypes=(q.QBit(), q.QBit()), shapes=((), ())
    )
    assert [int(value) for value in source.cvs] == [0, 0]
    bit, _ = pair("bit")
    _, native = import_control_values(
        clause, registers=(Register("a", bit), Register("b", bit))
    )
    assert [int(value) for value in native] == [0, 1]


@pytest.mark.parametrize("kind", [kind for kind in KINDS if kind.startswith("c")])
def test_classical_storage_rejected_at_quantum_boundary(kind: str) -> None:
    dtype, _ = pair(kind)
    with pytest.raises(ValueError, match="quantum register"):
        export_control_values((Register("q", dtype),), (np.array(0),))
    with pytest.raises(ValueError, match="quantum register"):
        import_control_values(
            cirq.ProductOfSums([0]), registers=(Register("q", dtype),)
        )


def test_unsupported_clauses_shapes_widths_and_symbolic_metadata() -> None:
    from sympy import Symbol

    from flagquantum.core.ir.program.register_types import unsigned_register_type

    dtype, _ = pair("bit")
    register = Register("q", dtype)
    with pytest.raises(ValueError, match="exactly one"):
        import_control_values(cirq.SumOfProducts([(0,), (1,)]))
    with pytest.raises(ValueError, match="binary"):
        import_control_values(cirq.SumOfProducts([(2,)]))
    with pytest.raises(ValueError, match="bit count"):
        import_control_values(cirq.ProductOfSums([0, 1]), registers=(register,))
    with pytest.raises(ValueError, match="one activation"):
        export_control_values((register,), ())
    with pytest.raises(ValueError, match="shape"):
        export_control_values((register,), (np.array([1]),))
    one_bit_word = Register("q", unsigned_register_type(1))
    with pytest.raises(ValueError, match="domain"):
        export_control_values((one_bit_word,), (np.array(2),))
    unknown = Register("q", type(dtype)("unknown", 1))
    with pytest.raises(ValueError, match="scalar bit codec"):
        export_control_values((unknown,), (np.array(0),))
    from flagquantum.core.ir.program.register_types import fixed_point_register_type

    fixed = Register("q", fixed_point_register_type(2, 1))
    _, fixed_values = import_control_values(
        cirq.ProductOfSums([0, 1]), registers=(fixed,)
    )
    assert (
        export_control_values((fixed,), fixed_values)
        == cirq.ProductOfSums([0, 1]).expand()
    )
    symbolic = Register("q", unsigned_register_type(Symbol("n", positive=True)))
    with pytest.raises(ValueError, match="symbolic"):
        export_control_values((symbolic,), (np.array(1),))


def test_transported_values_control_actual_appended_program() -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from examples.algorithms.controlled_programs import controlled_x_program
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch

    dtype, _ = pair("uint")
    ports = (Register("ctrl", dtype),)
    clause = cirq.ProductOfSums([0, 1])
    registers, values = import_control_values(clause, registers=ports)
    pattern = ControlRegister(2, tuple(int(x) for x in values[0].flat))
    record = controlled_x_program(registers, (pattern,))
    bound = bind_register_program(record, ((0, 1), (2,)), clean=(3,))
    from collections import Counter
    from dataclasses import asdict

    from qualtran.bloqs.basic_gates import XGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
        estimate_component_resources,
    )
    from flagquantum.compiler.registers.control.controlled_component_resource_model import (
        controlled_component_cost,
    )
    from tests.unit.test_register_control_tutorial import _source_gates
    from tests.unit.test_register_subtraction import assert_source

    source = XGate().controlled(q.CtrlSpec(q.QUInt(2), 1))
    assert_source(
        bound.circuit.instructions,
        _source_gates(source, {"ctrl": (0, 1), "q": (2,)}, (3,), {3: 0}),
    )
    observed = Counter(gate.name for gate in bound.circuit.instructions)
    assert observed["t"] + observed["tdg"] == 4
    assert observed["measure"] == 1
    model = controlled_component_cost(
        ComponentCost("cx", 2, GateCost(clifford=1)), (pattern,), target_width=1
    )
    actual_model = asdict(estimate_component_resources(model).gates)
    expected_model = get_cost_value(source, QECGatesCost())
    assert actual_model == {key: getattr(expected_model, key) for key in actual_model}
    vector = torch.complex(torch.arange(1, 17).double(), torch.arange(16).double())
    vector /= vector.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    initial.reshape(8, 2, 2)[:, 0, :] = vector.reshape(8, 2)
    op = cirq.X.controlled(num_controls=2, control_values=clause)
    expected_values = torch.as_tensor(cirq.unitary(op).copy()) @ vector.reshape(8, 2)
    expected = torch.zeros_like(initial)
    expected.reshape(8, 2, 2)[:, 0, :] = expected_values
    for branch in (0, 1):
        actual = forced_branch(initial, bound.circuit.instructions, (branch,))
        torch.testing.assert_close(actual, expected * 2**-0.5, rtol=0, atol=3e-14)
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(5, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=2, seed=11, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=3e-14
    )


@pytest.mark.parametrize("polynomial", [False, True])
@pytest.mark.parametrize("shape", [(), (2,)])
def test_field_and_polynomial_transport(
    polynomial: bool, shape: tuple[int, ...]
) -> None:
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
    from flagquantum.simulation.finite_field_polynomial_encoding import (
        FiniteFieldPolynomialEncoding,
    )

    field = FiniteFieldEncoding(2, 2)
    encoding = FiniteFieldPolynomialEncoding(1, field) if polynomial else field
    source = q.QGFPoly(1, q.QGF(2, 2)) if polynomial else q.QGF(2, 2)
    registers = (Register("field", encoding.register_type(), shape),)
    for value in source.get_classical_domain():
        array = np.empty(shape, dtype=object)
        array.fill(value)
        original = q.CtrlSpec(source, array)
        exported = export_control_values(registers, (array,))
        assert exported == original.to_cirq_cv()
        returned, actual = import_control_values(exported, registers=registers)
        assert export_control_values(returned, actual) == exported
