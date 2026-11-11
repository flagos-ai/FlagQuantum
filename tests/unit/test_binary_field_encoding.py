"""Native field metadata reconstruction retains the pinned dtype's identity."""

from collections import Counter
from dataclasses import replace
from typing import SupportsInt

import numpy as np
import pytest
import sympy
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.binary_field_encoding import resolve_binary_field_encoding
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from tests.unit.test_classical_field_registers import field_add_function

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("polynomial", ["default", "none"])
def test_admitted_projection_restores_source_class_and_complete_scalar_codecs(
    width: int, polynomial: str
) -> None:
    from qualtran import QGF

    original = (
        FiniteFieldEncoding(2, width)
        if polynomial == "default"
        else FiniteFieldEncoding(2, width, None)
    )
    source = QGF(2, width) if polynomial == "default" else QGF(2, width, None)
    dtype = original.register_type()
    restored = resolve_binary_field_encoding(dtype)
    assert restored.register_type() == dtype
    assert "gf_type" not in restored.__dict__
    assert restored.gf_type is source.gf_type is original.gf_type
    if polynomial == "none":
        assert restored.irreducible_poly is None
    else:
        assert restored.irreducible_poly == source.irreducible_poly
    for value in source.get_classical_domain():
        assert restored.to_bits(value) == source.to_bits(value)
        result = restored.from_bits(source.to_bits(value))
        assert type(result) is type(value) and result == value
    np.testing.assert_array_equal(
        restored.gf_type.elements * restored.gf_type(1), source.gf_type.elements
    )


@pytest.mark.parametrize(
    "characteristic,degree,primitive,modulus",
    [
        (2, 1, None, None),
        (3, 1, None, None),
        (7, 1, 3, None),
        (7, 1, 5, None),
        (2, 3, None, 11),
        (2, 3, None, 13),
    ],
)
def test_coefficient_field_and_ignored_target_polynomial_are_not_collapsed(
    characteristic: int, degree: int, primitive: int | None, modulus: int | None
) -> None:
    import galois
    from qualtran import QGF

    coefficient = galois.GF(
        characteristic,
        degree,
        primitive_element=primitive,
        irreducible_poly=modulus,
        compile="python-calculate",
    )
    polynomial = galois.Poly.Int(3, field=coefficient)
    original = FiniteFieldEncoding(2, 1, polynomial)
    source = QGF(2, 1, polynomial)
    restored = resolve_binary_field_encoding(original.register_type())
    assert restored.register_type() == original.register_type()
    assert restored.irreducible_poly.field is polynomial.field
    assert restored.irreducible_poly == source.irreducible_poly
    assert restored.gf_type is source.gf_type
    assert list(restored.get_domain()) == list(source.get_classical_domain())


@pytest.mark.parametrize("modulus", [11, 13])
def test_nondefault_binary_modulus_preserves_source_multiplication(
    modulus: int,
) -> None:
    import galois
    from qualtran import QGF

    polynomial = galois.Poly.Int(modulus)
    field = FiniteFieldEncoding(2, 3, polynomial)
    source = QGF(2, 3, polynomial)
    restored = resolve_binary_field_encoding(field.register_type())
    assert restored.gf_type is source.gf_type
    left = restored.gf_type.elements[:, None]
    right = restored.gf_type.elements[None, :]
    np.testing.assert_array_equal(
        left * right,
        source.gf_type.elements[:, None] * source.gf_type.elements[None, :],
    )
    assert int(restored.gf_type(2) * restored.gf_type(4)) == (3 if modulus == 11 else 5)


@pytest.mark.parametrize(
    "defect",
    [
        "integer",
        "classical",
        "odd",
        "width",
        "symbolic",
        "bool",
        "order",
        "extra",
        "duplicate",
        "reordered",
        "partial_poly",
    ],
)
def test_core_admission_precedes_any_optional_library_import(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import flagquantum.simulation.binary_field_encoding as owner

    field = FiniteFieldEncoding(2, 3)
    dtype = field.register_type()
    base = dtype.parameters[:3]
    mutated = {
        "integer": RegisterType("uint", 3),
        "classical": field.register_type(classical=True),
        "odd": RegisterType(
            "finite_field",
            4,
            parameters=(("characteristic", 3), ("degree", 2), ("order", 9)),
        ),
        "width": replace(dtype, quantum_bits=2),
        "symbolic": replace(dtype, quantum_bits=sympy.Symbol("n")),
        "bool": replace(
            dtype, parameters=(("characteristic", True), *dtype.parameters[1:])
        ),
        "order": replace(dtype, parameters=((*base[:2], ("order", 7)))),
        "extra": replace(dtype, parameters=(*dtype.parameters, ("extra", 1))),
        "duplicate": replace(dtype, parameters=(*dtype.parameters, ("order", 8))),
        "reordered": replace(dtype, parameters=tuple(reversed(dtype.parameters))),
        "partial_poly": replace(dtype, parameters=(*base, ("polynomial", 11))),
    }[defect]
    monkeypatch.setattr(
        owner, "import_module", lambda name: pytest.fail("optional import")
    )
    with pytest.raises(ValueError):
        owner.resolve_binary_field_encoding(mutated)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("polynomial_field_modulus", 5),
        ("polynomial_field_primitive_element", 2),
        ("polynomial_field_degree", -1),
    ],
)
def test_inconsistent_numerical_metadata_is_rejected_not_repaired(
    parameter: str, value: int
) -> None:
    dtype = FiniteFieldEncoding(2, 1).register_type()
    mutated = replace(
        dtype,
        parameters=tuple(
            (name, value if name == parameter else item)
            for name, item in dtype.parameters
        ),
    )
    with pytest.raises(ValueError):
        resolve_binary_field_encoding(mutated)


def fixture(field: FiniteFieldEncoding) -> ProgramRecord:
    leaf = field_add_function(field.register_type())
    builder = ComponentProgramBuilder(Signature(leaf.registers or ()), (leaf,))
    return builder.finalize(builder.call_dict("field_add", builder.inputs))


def test_graph_resolves_once_per_dtype_and_explicit_override_remains_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.component_classical_state as owner

    field = FiniteFieldEncoding(2, 3)
    record = fixture(field)
    dtype = field.register_type()
    calls: list[RegisterType] = []

    def resolve(declaration: RegisterType) -> FiniteFieldEncoding:
        calls.append(declaration)
        return resolve_binary_field_encoding(declaration)

    monkeypatch.setattr(owner, "resolve_binary_field_encoding", resolve)

    def action(function: object, values: dict[str, object]) -> object:
        return values

    state = owner.ComponentClassicalState(
        record, (0,), {"x": field.gf_type(1), "y": field.gf_type(2)}, action
    )
    assert calls == [dtype]
    state.simulate()
    assert calls == [dtype]
    calls.clear()
    state = owner.ComponentClassicalState(
        record,
        (0,),
        {"x": field.gf_type(1), "y": field.gf_type(2)},
        action,
        field_encodings={dtype: field},
    )
    state.simulate()
    assert not calls
    with pytest.raises(ValueError, match="differs"):
        owner.ComponentClassicalState(
            record, (0,), {}, action, field_encodings={dtype: FiniteFieldEncoding(2, 2)}
        )
    assert not calls


def test_integer_graph_does_not_request_a_field_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.component_classical_state as owner
    from tests.unit.test_component_classical_state import add_values, order
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add

    monkeypatch.setattr(
        owner,
        "resolve_binary_field_encoding",
        lambda dtype: pytest.fail("field resolution"),
    )
    record, _ = wrapped_add(1)
    output = owner.ComponentClassicalState(
        record, order(record), {"a": 1, "b": 2, "spectator": 1}, add_values
    ).simulate()
    assert output == {"a": 1, "b": 4, "spectator": 1}


@pytest.mark.parametrize(
    "width,polynomial",
    [(1, "default"), (2, "default"), (3, "default"), (3, "none"), (3, "11"), (3, "13")],
)
def test_automatic_field_graph_matches_source_full_matrix_gate_stream_and_append(
    width: int, polynomial: str
) -> None:
    import cirq
    import galois
    from qualtran import QGF
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.cirq_interop import BloqAsCirqGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.classical_sim import (
        ClassicalSimState,
        get_classical_truth_table,
    )

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.simulation.classical_truth_table import classical_truth_table

    poly = (
        None
        if polynomial == "none"
        else galois.Poly.Int(int(polynomial)) if polynomial.isdecimal() else None
    )
    field = (
        FiniteFieldEncoding(2, width)
        if polynomial == "default"
        else FiniteFieldEncoding(2, width, poly)
    )
    source_dtype = QGF(2, width) if polynomial == "default" else QGF(2, width, poly)
    source = GF2Addition(source_dtype)
    record = fixture(field)
    signature = Signature(record.functions[0].registers or ())

    def action(function: object, values: dict[str, object]) -> object:
        x, y = values["x"], values["y"]
        assert isinstance(x, field.gf_type) and isinstance(y, field.gf_type)
        return {"x": x, "y": x + y}

    def evaluate(values: dict[str, object]) -> tuple[object, ...]:
        state = ComponentClassicalState(record, (0,), values, action)
        output = state.simulate()
        reference = ClassicalSimState.from_cbloq(
            source.as_composite_bloq(), values
        ).simulate()
        assert type(output["x"]) is type(reference["x"]) and type(output["y"]) is type(
            reference["y"]
        )
        assert output == reference
        return output["x"], output["y"]

    table = classical_truth_table(signature, lambda r: field.get_domain(), evaluate)
    assert table == get_classical_truth_table(source)
    size = 1 << (2 * width)
    matrix = torch.zeros((size, size), dtype=torch.complex128)

    def integer(value: object) -> int:
        assert isinstance(value, SupportsInt)
        return int(value)

    for inputs, outputs in table[2]:
        matrix[
            (integer(outputs[0]) << width) | integer(outputs[1]),
            (integer(inputs[0]) << width) | integer(inputs[1]),
        ] = 1
    torch.testing.assert_close(
        matrix, torch.from_numpy(source.tensor_contract()), atol=1e-12, rtol=0
    )
    bound = bind_register_program(
        record,
        (tuple(range(width)), tuple(range(width, 2 * width))),
        clean=(2 * width,),
    )
    wires = np.asarray(cirq.LineQubit.range(2 * width))
    source_circuit, _ = source.decompose_bloq().to_cirq_circuit_and_quregs(
        x=wires[:width], y=wires[width:]
    )
    expected = []
    for gate in source_circuit.all_operations():
        assert gate.gate in {cirq.CNOT, BloqAsCirqGate(CNOT())}
        expected.append(("cx", tuple(qubit.x for qubit in gate.qubits)))
    assert [(gate.name, gate.wires) for gate in bound.circuit.instructions] == expected
    assert Counter(gate.name for gate in bound.circuit.instructions) == {"cx": width}
    assert get_cost_value(source, QECGatesCost()).clifford == width
    initial = torch.zeros(2 * size, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, size + 1, dtype=torch.float64),
        torch.arange(size, 0, -1, dtype=torch.float64),
    )
    initial[::2] = amplitudes / amplitudes.norm()
    output = torch.zeros_like(initial)
    output[::2] = matrix @ initial[::2]
    op = FieldOperation(2 * width + 1, bound.circuit.instructions)
    actual = (
        Circuit(2 * width + 1, inputs=initial, dtype=torch.complex128)
        .append(op)
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, output, atol=1e-12, rtol=0)
    assert torch.count_nonzero(actual[1::2]) == 0
    restored = (
        Circuit(2 * width + 1, inputs=actual, dtype=torch.complex128)
        .append(op)
        .state()
        .flatten()
    )
    torch.testing.assert_close(restored, initial, atol=1e-12, rtol=0)


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_automatic_array_graph_preserves_field_storage_and_empty_gap(
    shape: tuple[int, ...],
) -> None:
    from tests.unit.test_classical_field_registers import source_state

    field = FiniteFieldEncoding(2, 3)
    signature = Signature([Register("x", field.register_type(), shape)])
    leaf_builder = ComponentProgramBuilder(signature, name="identity")
    leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
    builder = ComponentProgramBuilder(signature, (leaf,))
    record = builder.finalize(builder.call_dict("identity", builder.inputs))
    value = field.gf_type.Ones(shape) if shape else field.gf_type(1)
    state = ComponentClassicalState(record, (0,), {"x": value}, lambda f, v: v)
    output = state.simulate()["x"]
    assert isinstance(output, field.gf_type)
    np.testing.assert_array_equal(output, value)
    reference, _ = source_state(field, shape, {"x": value})
    if shape == (0,):
        with pytest.raises(KeyError, match="RightDangle"):
            reference.simulate()
    else:
        expected = reference.simulate()["x"]
        assert type(output) is type(expected)
        np.testing.assert_array_equal(output, expected)
