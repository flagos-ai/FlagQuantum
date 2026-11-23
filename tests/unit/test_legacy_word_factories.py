"""Complete bitwise source libraries execute through native typed word owners."""

from collections import Counter
from dataclasses import replace
from importlib import import_module
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentLibrary,
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import word_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.ecosystem._component_library import bitwise_library_factories
from flagquantum.errors import SerializationError
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
DTYPES = (
    "QAny",
    "QUInt",
    "QInt",
    "QMontgomeryUInt",
    "BQUInt",
    "QFxp",
    "QIntOnesComp",
    "QBit",
    "QGF",
)


def _dtype(kind: str) -> Any:
    source = import_module("qualtran")
    if kind == "QBit":
        return source.QBit()
    if kind == "QGF":
        return source.QGF(2, 3)
    if kind == "QFxp":
        return source.QFxp(3, 1)
    return getattr(source, kind)(3, 5) if kind == "BQUInt" else getattr(source, kind)(3)


def _library(source: Any, depth: int = 1) -> ComponentLibrary:
    codec = import_module("qualtran.serialization.bloq")
    return deserialize_component_library(
        codec.bloqs_to_proto(source, max_depth=depth).SerializeToString()
    )


def _factories() -> dict[str, ComponentFactory]:
    return {
        **fixed_gate_factories(),
        **bookkeeping_factories(),
        **bitwise_library_factories(),
    }


@pytest.mark.parametrize("kind", ("Xor", "XorK", "BitwiseNot"))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", (torch.complex64, torch.complex128))
def test_full_constructor_amplitudes_msb_gate_stream_resources_and_involution(
    kind: str, dtype: str, precision: torch.dtype
) -> None:
    source_dtype = _dtype(dtype)
    width = source_dtype.num_qubits
    operations = import_module("qualtran.bloqs.arithmetic.bitwise")
    constant = -1 if dtype in ("QInt", "QIntOnesComp") else 1
    original = (
        operations.XorK(source_dtype, constant)
        if kind == "XorK"
        else getattr(operations, kind)(source_dtype)
    )
    if dtype == "QGF":
        if kind == "XorK":
            with pytest.raises(ValueError, match="val should be a"):
                _library(original)
        else:
            with pytest.raises(TypeError, match="Data type.*QGF.*not recognized"):
                _library(original)
        return
    library = _library(original)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    program = resolve_component_program(library, 0, _factories())
    assert verify_program_record(program) == ()
    n = width * (2 if kind == "Xor" else 1)
    wires = tuple(reversed(range(n)))
    ports = (wires[:width], wires[width:]) if kind == "Xor" else (wires,)
    bound = bind_register_program(program, ports)
    assert bound.outputs == ports and not bound.events
    source_ports = dict(
        zip(("x", "y") if kind == "Xor" else ("x",), ports, strict=True)
    )
    expected = reference_arithmetic_gates(original, source_ports, (), {})
    assert_source(bound.circuit.instructions, expected)
    decomposed = resolve_component_program(
        _library(original.decompose_bloq()), 0, _factories()
    )
    assert_source(
        bound.circuit.instructions,
        bind_register_program(decomposed, ports).circuit.instructions,
    )
    coeff = torch.arange(1, (1 << (n + 1)) + 1, dtype=torch.float64)
    state = torch.complex(coeff, coeff.flip(0)).to(precision)
    state /= state.norm()
    circuit = Circuit(n + 1, inputs=state, dtype=precision)
    operation = FieldOperation(n + 1, bound.circuit.instructions)
    circuit.append(operation)
    # Apply the independently contracted original matrix in the caller's reversed order.
    matrix = torch.as_tensor(original.tensor_contract(), dtype=precision).reshape(
        1 << n, 1 << n
    )
    order = (*wires, n)
    reverse = tuple(order.index(q) for q in range(n + 1))
    target = (
        (matrix @ state.reshape((2,) * (n + 1)).permute(order).reshape(1 << n, 2))
        .reshape((2,) * (n + 1))
        .permute(reverse)
        .flatten()
    )
    tolerance = 5e-7 if precision == torch.complex64 else 3e-12
    torch.testing.assert_close(
        circuit.state().flatten(), target, atol=tolerance, rtol=0
    )
    circuit.append(operation)
    torch.testing.assert_close(circuit.state().flatten(), state, atol=tolerance, rtol=0)
    counts = Counter(g.name for g in bound.circuit.instructions)
    expected_count = sum(source_dtype.to_bits(constant)) if kind == "XorK" else width
    assert counts == (
        {"cx" if kind == "Xor" else "x": expected_count} if expected_count else {}
    )
    resource = import_module("qualtran.resource_counting")
    cost = resource.get_cost_value(original, resource.QECGatesCost())
    assert cost.clifford == expected_count
    assert cost.total_t_count() == cost.measurement == 0


@pytest.mark.parametrize(
    "example", ("_xork", "_xor", "_xor_symb", "_bitwise_not", "_bitwise_not_symb")
)
def test_all_original_examples_and_depths(example: str) -> None:
    source = getattr(import_module("qualtran.bloqs.arithmetic.bitwise"), example).make()
    for depth in (0, 1, 2):
        program = resolve_component_program(_library(source, depth), 0, _factories())
        assert verify_program_record(program) == ()
        if "symb" in example:
            assert not program.functions[0].blocks


@pytest.mark.parametrize("gate", ("XGate", "Hadamard", "SGate", "TGate"))
@pytest.mark.parametrize("n", (0, 1, 3))
def test_on_each_complete_source_examples_and_phase(gate: str, n: int) -> None:
    basic = import_module("qualtran.bloqs.basic_gates")
    original = basic.OnEach(n, getattr(basic, gate)())
    program = resolve_component_program(_library(original), 0, _factories())
    bound = bind_register_program(program, (tuple(reversed(range(n))),), borrowed=(n,))
    assert not bound.events and bound.outputs == (tuple(reversed(range(n))),)
    assert len(bound.circuit.instructions) == n
    coefficient = torch.arange(1, (1 << (n + 1)) + 1, dtype=torch.float64)
    state = torch.complex(coefficient, 2 - coefficient)
    state /= state.norm()
    circuit = Circuit(n + 1, inputs=state, dtype=state.dtype).append(
        FieldOperation(n + 1, bound.circuit.instructions)
    )
    if n:
        matrix = torch.as_tensor(original.tensor_contract(), dtype=state.dtype).reshape(
            1 << n, 1 << n
        )
        # Identical one-bit factors make reversal invariant while the spectator stays untouched.
        expected = (matrix @ state.reshape(1 << n, 2)).flatten()
    else:
        expected = state
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=3e-12, rtol=0)


def test_validation_and_explicit_encoder_replacement() -> None:
    factories = bitwise_library_factories()
    xor = factories["qualtran.bloqs.arithmetic.bitwise.Xor"]
    constant = factories["qualtran.bloqs.arithmetic.bitwise.XorK"]
    each = factories["qualtran.bloqs.basic_gates.on_each.OnEach"]
    for args in ({}, {"dtype": 1}, {"dtype": unsigned_register_type(3), "other": 1}):
        with pytest.raises(SerializationError):
            xor("x", None, args)
    with pytest.raises(SerializationError, match="signature"):
        xor("x", (), {"dtype": unsigned_register_type(3)})
    for k in (-1, 8):
        with pytest.raises(ValueError):
            constant("x", None, {"dtype": unsigned_register_type(3), "k": k})

    def alternate(dtype: RegisterType, value: object) -> tuple[int, ...]:
        assert dtype == unsigned_register_type(3) and value == 1
        return (0, 0, 1)

    replacement = word_factories(encode_constant=alternate)
    args = {"dtype": unsigned_register_type(3), "k": 1}
    assert constant("x", None, args) == replacement[
        "qualtran.bloqs.arithmetic.bitwise.XorK"
    ]("x", None, args)

    def invalid(dtype: RegisterType, value: object) -> tuple[int, ...]:
        return (2,)

    with pytest.raises(SerializationError, match="binary MSB"):
        word_factories(encode_constant=invalid)[
            "qualtran.bloqs.arithmetic.bitwise.XorK"
        ]("x", None, args)
    child_program = resolve_component_program(
        _library(import_module("qualtran.bloqs.basic_gates").XGate()),
        0,
        fixed_gate_factories(),
    )
    child = child_program.functions[0]
    for args in (
        {},
        {"n": True, "gate": child},
        {"n": 3, "gate": 1},
        {"n": 3, "gate": replace(child, registers=())},
        {"n": 3, "gate": child, "target_dtype": unsigned_register_type(2)},
    ):
        with pytest.raises(SerializationError):
            each("x", None, args)
    with pytest.raises(SerializationError, match="signature"):
        each("x", (), {"n": 3, "gate": child})
    with pytest.raises(ValueError):
        each("x", None, {"n": -1, "gate": child})
    declared = each("x", None, {"n": sympy.Symbol("n"), "gate": child})
    assert isinstance(declared, FunctionRecord) and not declared.blocks
    symbolic = constant(
        "x", None, {"dtype": unsigned_register_type(3), "k": sympy.Symbol("k")}
    )
    assert isinstance(symbolic, FunctionRecord) and not symbolic.blocks
    for k in (0, 3, 7):
        program = constant("x", None, {"dtype": opaque_register_type(3), "k": k})
        assert isinstance(program, ProgramRecord)


@pytest.mark.parametrize(
    "dtype",
    ("QUInt", "QInt", "QMontgomeryUInt", "BQUInt", "QFxp", "QIntOnesComp", "QBit"),
)
def test_constant_encoding_entire_small_source_domain_and_refusals(dtype: str) -> None:
    operations = import_module("qualtran.bloqs.arithmetic.bitwise")
    source_dtype = _dtype(dtype)
    constant_factory = bitwise_library_factories()[
        "qualtran.bloqs.arithmetic.bitwise.XorK"
    ]
    decoded = _library(operations.XorK(source_dtype, 0), 0).entries[0].declaration
    native_dtype = next(arg.value for arg in decoded.arguments if arg.name == "dtype")
    assert isinstance(native_dtype, RegisterType)
    width = source_dtype.num_qubits
    for value in range(-(1 << width) - 1, (1 << width) + 1):
        original = operations.XorK(source_dtype, value)
        try:
            bits = source_dtype.to_bits(value)
        except ValueError:
            with pytest.raises(ValueError):
                constant_factory("main", None, {"dtype": native_dtype, "k": value})
            continue
        program = constant_factory("main", None, {"dtype": native_dtype, "k": value})
        assert isinstance(program, ProgramRecord)
        gates = bind_register_program(
            program, (tuple(range(width)),)
        ).circuit.instructions
        assert tuple(g.wires[0] for g in gates) == tuple(
            i for i, bit in enumerate(bits) if bit
        )
        resource = import_module("qualtran.resource_counting")
        assert resource.get_cost_value(
            original, resource.QECGatesCost()
        ).clifford == len(gates)
    # Sympy numeric literals retain the source symbolic-decomposition boundary.
    for value in (sympy.Integer(0), sympy.Integer(1)):
        program = constant_factory("main", None, {"dtype": native_dtype, "k": value})
        assert isinstance(program, FunctionRecord) and not program.blocks


def test_on_each_rejects_unresolved_child_dependencies_before_namespacing() -> None:
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.core.ir.program.signatures import Signature

    source = import_module("qualtran.bloqs.basic_gates").XGate()
    primitive = resolve_component_program(
        _library(source), 0, fixed_gate_factories()
    ).functions[0]
    primitive = replace(primitive, name="leaf")
    builder = ComponentProgramBuilder(Signature.build(q=1), (primitive,), name="nested")
    nested = builder.finalize(
        builder.call_dict("leaf", {"q": builder.inputs["q"]})
    ).functions[0]
    factory = bitwise_library_factories()["qualtran.bloqs.basic_gates.on_each.OnEach"]
    with pytest.raises(SerializationError, match="self-contained"):
        factory("main", None, {"n": 3, "gate": nested})
