"""Field assignment and SSA execution match the pinned source numerical chain."""

from collections import Counter
from collections.abc import Mapping
from typing import Any, SupportsInt

import numpy as np
import pytest
import sympy
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ValueRef,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_distributions import ClassicalDistribution
from flagquantum.simulation.classical_field_registers import (
    bind_field_register_values,
    collect_field_register_value,
)
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_state(
    field: FiniteFieldEncoding,
    shape: tuple[int, ...],
    values: Mapping[str, object],
    fixed: object = None,
) -> tuple[Any, Any]:
    from qualtran import QGF, Bloq
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import PhasedClassicalSimState

    register = SourceRegister(
        "x",
        QGF(
            field.characteristic, field.degree, irreducible_poly=field.irreducible_poly
        ),
        shape=shape,
    )
    identity: Any = type(
        "FieldIdentity",
        (Bloq,),
        {
            "signature": property(lambda self: SourceSignature([register])),
            "on_classical_vals": lambda self, **values: values,
        },
    )
    return (
        PhasedClassicalSimState.from_cbloq(
            identity().as_composite_bloq(), values, fixed_random_vals={-1: fixed}
        ),
        register,
    )


@pytest.mark.parametrize("characteristic,degree", [(2, 1), (2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_field_binding_collection_types_and_scalar_identity(
    characteristic: int, degree: int, shape: tuple[int, ...]
) -> None:
    from qualtran import LeftDangle, Soquet
    from qualtran.simulation.classical_sim import _get_in_vals

    field = FiniteFieldEncoding(characteristic, degree)
    value = field.gf_type.Ones(shape) if shape else field.gf_type(1)
    state, register = source_state(field, shape, {"x": value})
    actual = bind_field_register_values("x", {"x": value}, field, shape=shape)
    assert tuple(actual) == tuple(register.all_idxs())
    for index, assigned in actual.items():
        reference = state.soq_assign[Soquet(LeftDangle, register, index)]
        assert type(assigned) is type(reference)
        np.testing.assert_array_equal(assigned, reference)
    output = collect_field_register_value(actual, field, shape=shape)
    expected = _get_in_vals(LeftDangle, register, state.soq_assign)
    assert type(output) is type(expected)
    np.testing.assert_array_equal(output, expected)
    if not shape:
        assert actual[()] is value and output is value
    else:
        assert isinstance(output, field.gf_type)
        assert output is not value
        np.testing.assert_array_equal(value, field.gf_type.Ones(shape))


@pytest.mark.parametrize(
    "kind",
    [
        "integer",
        "other_field",
        "expr",
        "missing",
        "shape",
        "negative",
        "large",
        "float_array",
        "nan_array",
        "plain_array",
    ],
)
def test_validation_and_collection_failure_boundaries_match_source(kind: str) -> None:
    from qualtran import LeftDangle
    from qualtran.simulation.classical_sim import _get_in_vals

    field = FiniteFieldEncoding(2, 3)
    shape = (
        (2,)
        if kind
        in {"shape", "negative", "large", "float_array", "nan_array", "plain_array"}
        else ()
    )
    values: dict[str, object] = {
        "x": {
            "integer": 1,
            "other_field": FiniteFieldEncoding(2, 2).gf_type(1),
            "expr": sympy.Symbol("v"),
            "missing": None,
            "shape": [1],
            "negative": [-1, 0],
            "large": [0, 8],
            "float_array": [0.5, 1.0],
            "nan_array": [float("nan"), 0],
            "plain_array": [0, 7],
        }[kind]
    }
    if kind == "missing":
        values.clear()
    try:
        state, register = source_state(field, shape, values)
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)):
            bind_field_register_values("x", values, field, shape=shape)
        return
    actual = bind_field_register_values("x", values, field, shape=shape)
    try:
        expected = _get_in_vals(LeftDangle, register, state.soq_assign)
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)):
            collect_field_register_value(actual, field, shape=shape)
        return
    result = collect_field_register_value(actual, field, shape=shape)
    assert type(result) is type(expected)
    np.testing.assert_array_equal(result, expected)


def test_missing_collection_index_is_not_filled_with_zero() -> None:
    field = FiniteFieldEncoding(2, 2)
    with pytest.raises(KeyError):
        collect_field_register_value({(0,): field.gf_type(1)}, field, shape=(2,))


def test_shared_traversal_is_the_replaced_owner_for_both_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.classical_field_registers as fields
    import flagquantum.simulation.classical_registers as integers

    field = FiniteFieldEncoding(2, 2)
    bound: dict[tuple[int, ...], object] = {(): field.gf_type(1)}

    def bind(*args: object, **kwargs: object) -> dict[tuple[int, ...], object]:
        return bound

    monkeypatch.setattr(integers, "bind_classical_register_values", bind)
    monkeypatch.setattr(fields, "bind_classical_register_values", bind)
    assert fields.bind_field_register_values("x", {}, field) is bound
    assert integers.bind_integer_register_values("x", {}, 2) is bound
    result = object()

    def collect(*args: object, **kwargs: object) -> object:
        return result

    monkeypatch.setattr(integers, "collect_classical_register_value", collect)
    monkeypatch.setattr(fields, "collect_classical_register_value", collect)
    assert fields.collect_field_register_value({}, field, shape=(2,)) is result
    assert integers.collect_integer_register_value({}, 2, shape=(2,)) is result


@pytest.mark.parametrize("mode", ["field", "integer", "expr", "banned"])
def test_field_distribution_resolves_once_then_validates(mode: str) -> None:
    field = FiniteFieldEncoding(2, 2)
    sampled: object = (
        field.gf_type(1)
        if mode == "field"
        else 1 if mode == "integer" else sympy.Symbol("s")
    )
    distribution = ClassicalDistribution([0, 1], [0.5, 0.5])
    seen: list[ClassicalDistribution] = []

    def resolve(value: ClassicalDistribution) -> object:
        seen.append(value)
        return sampled

    if mode == "field":
        result = bind_field_register_values(
            "x", {"x": distribution}, field, resolve_distribution=resolve
        )
        assert result[()] is sampled
    else:
        with pytest.raises(ValueError):
            bind_field_register_values(
                "x",
                {"x": distribution},
                field,
                resolve_distribution=None if mode == "banned" else resolve,
            )
    assert seen == ([] if mode == "banned" else [distribution])


def field_add_function(dtype: RegisterType) -> FunctionRecord:
    """Build only the native typed leaf; source decomposition remains independent."""
    width = int(dtype.num_bits)
    x, y = ValueRef("entry", 0, "register"), ValueRef("entry", 1, "register")
    next_id = 2

    def refs(count: int, kind: str) -> tuple[ValueRef, ...]:
        nonlocal next_id
        result = tuple(ValueRef("entry", next_id + i, kind) for i in range(count))
        next_id += count
        return result

    xb, yb = refs(width, "qubit"), refs(width, "qubit")
    operations = [
        Operation("split_register", (x,), xb),
        Operation("split_register", (y,), yb),
    ]
    xx, yy = [], []
    for a, b in zip(xb, yb, strict=True):
        result = refs(2, "qubit")
        operations.append(Operation("cx", (a, b), result))
        xx.append(result[0])
        yy.append(result[1])
    outputs = refs(2, "register")
    operations.extend(
        [
            Operation("join_register", tuple(xx), (outputs[0],), register_type=dtype),
            Operation("join_register", tuple(yy), (outputs[1],), register_type=dtype),
        ]
    )
    return FunctionRecord(
        "field_add",
        (
            BlockRecord(
                "entry", (x, y), operations=tuple(operations), return_values=outputs
            ),
        ),
        registers=(Register("x", dtype), Register("y", dtype)),
    )


@pytest.mark.parametrize("width", [1, 2, 3])
def test_binary_field_graph_full_table_matrix_and_append(width: int) -> None:
    import cirq
    from qualtran import QGF
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.cirq_interop import BloqAsCirqGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.simulation.classical_truth_table import classical_truth_table

    field = FiniteFieldEncoding(2, width)
    dtype = field.register_type()
    leaf = field_add_function(dtype)
    builder = ComponentProgramBuilder(Signature(leaf.registers or ()), (leaf,))
    record = builder.finalize(builder.call_dict("field_add", builder.inputs))
    source = GF2Addition(QGF(2, width))

    def action(function: FunctionRecord, values: dict[str, object]) -> object:
        x, y = values["x"], values["y"]
        assert isinstance(x, field.gf_type) and isinstance(y, field.gf_type)
        return {"x": x, "y": x + y}

    def evaluate(values: dict[str, object]) -> tuple[object, ...]:
        state = ComponentClassicalState(
            record, (0,), values, action, field_encodings={dtype: field}
        )
        outputs = state.simulate()
        from qualtran.simulation.classical_sim import ClassicalSimState

        reference = ClassicalSimState.from_cbloq(
            source.as_composite_bloq(), values
        ).simulate()
        assert isinstance(outputs["x"], field.gf_type) and isinstance(
            outputs["y"], field.gf_type
        )
        assert outputs == reference
        return outputs["x"], outputs["y"]

    table = classical_truth_table(
        Signature(leaf.registers or ()), lambda r: field.get_domain(), evaluate
    )
    size = 1 << (2 * width)

    def integer(value: object) -> int:
        assert isinstance(value, SupportsInt)
        return int(value)

    matrix = torch.zeros((size, size), dtype=torch.complex128)
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
    circuit, _ = source.decompose_bloq().to_cirq_circuit_and_quregs(
        x=wires[:width], y=wires[width:]
    )
    expected = []
    for op in circuit.all_operations():
        assert op.gate in {cirq.CNOT, BloqAsCirqGate(CNOT())}
        expected.append(("cx", tuple(q.x for q in op.qubits)))
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == expected
    assert Counter(g.name for g in bound.circuit.instructions) == {"cx": width}
    assert get_cost_value(source, QECGatesCost()).clifford == width
    initial = torch.zeros(2 * size, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, size + 1, dtype=torch.float64),
        torch.arange(size, 0, -1, dtype=torch.float64),
    )
    initial[::2] = amplitudes / amplitudes.norm()
    expected_state = torch.zeros_like(initial)
    expected_state[::2] = matrix @ initial[::2]
    operation = FieldOperation(2 * width + 1, bound.circuit.instructions)
    actual = (
        Circuit(2 * width + 1, inputs=initial, dtype=torch.complex128)
        .append(operation)
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected_state, atol=1e-12, rtol=0)
    assert torch.count_nonzero(actual[1::2]) == 0
    restored = (
        Circuit(2 * width + 1, inputs=actual, dtype=torch.complex128)
        .append(operation)
        .state()
        .flatten()
    )
    torch.testing.assert_close(restored, initial, atol=1e-12, rtol=0)


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_field_graph_array_passthrough_and_empty_source_gap(
    shape: tuple[int, ...],
) -> None:
    field = FiniteFieldEncoding(2, 3)
    dtype = field.register_type()
    signature = Signature([Register("x", dtype, shape)])
    leaf_builder = ComponentProgramBuilder(signature, name="identity")
    leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
    builder = ComponentProgramBuilder(signature, (leaf,))
    record = builder.finalize(builder.call_dict("identity", builder.inputs))
    value = field.gf_type.Ones(shape) if shape else field.gf_type(1)
    state = ComponentClassicalState(
        record, (0,), {"x": value}, lambda f, v: v, field_encodings={dtype: field}
    )
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
    assert record.functions[0].registers == tuple(signature)


def test_field_metadata_mismatch_is_refused_before_callbacks() -> None:
    field = FiniteFieldEncoding(2, 3)
    dtype = field.register_type()
    leaf = field_add_function(dtype)
    builder = ComponentProgramBuilder(Signature(leaf.registers or ()), (leaf,))
    record = builder.finalize(builder.call_dict("field_add", builder.inputs))
    with pytest.raises(ValueError, match="differs"):
        ComponentClassicalState(
            record,
            (0,),
            {},
            lambda f, v: pytest.fail("action"),
            field_encodings={dtype: FiniteFieldEncoding(2, 2)},
        )
