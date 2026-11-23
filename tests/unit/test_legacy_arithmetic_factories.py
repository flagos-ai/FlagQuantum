"""Complete integer source constructors retain native typed execution semantics."""

from collections import Counter
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_arithmetic_factories import (
    arithmetic_factories,
)
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
from flagquantum.compiler.formats.legacy_conjunction_factories import (
    conjunction_factories,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
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
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._component_library import (
    arithmetic_library_factories,
    bitwise_library_factories,
)
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _factories() -> dict[str, ComponentFactory]:
    return {
        **fixed_gate_factories(),
        **bookkeeping_factories(),
        **conjunction_factories(),
        **bitwise_library_factories(),
        **arithmetic_library_factories(),
    }


def _library(original: Any, depth: int = 1) -> ComponentLibrary:
    codec = import_module("qualtran.serialization.bloq")
    return deserialize_component_library(
        codec.bloqs_to_proto(original, max_depth=depth).SerializeToString()
    )


def _resolve(original: Any, depth: int = 1) -> ProgramRecord:
    library = _library(original, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    program = resolve_component_program(library, 0, _factories())
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("kind", ("Add", "AddK", "Negate"))
@pytest.mark.parametrize("dtype", ("QUInt", "QInt", "QMontgomeryUInt"))
@pytest.mark.parametrize("width", (1, 2, 3, 5))
def test_named_and_decomposed_libraries_source_gates_operands_feedback_and_resources(
    kind: str, dtype: str, width: int
) -> None:
    source = import_module("qualtran")
    arithmetic = import_module("qualtran.bloqs.arithmetic.addition")
    native_type = getattr(source, dtype)(width)
    if kind == "Negate":
        original = import_module("qualtran.bloqs.arithmetic.negate").Negate(native_type)
    elif kind == "AddK":
        original = arithmetic.AddK(native_type, -1)
    else:
        original = arithmetic.Add(
            native_type, getattr(source, dtype)(width + (width > 1))
        )
    if kind == "Negate" and dtype == "QInt" and width == 1:
        with pytest.raises(ValueError, match="Too-large"):
            _library(original)
        with pytest.raises(SerializationError, match="outside|large"):
            _resolve(original, 0)
        return
    program = _resolve(original)
    signature = original.signature
    logical = int(signature.n_qubits())
    offset = 0
    ports = {}
    for register in signature.lefts():
        count = int(register.bitsize)
        ports[register.name] = tuple(reversed(range(offset, offset + count)))
        offset += count
    right = int(signature.get_left("b").bitsize) if kind == "Add" else width
    count = right - 1 if kind == "Add" else 2 * width - 1
    work = tuple(range(logical + 1, logical + 1 + count))
    bound = bind_register_program(program, tuple(ports.values()), clean=work)
    slots = {
        g.wires[0]: int(g.metadata["classical_bit"])
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    expected = reference_arithmetic_gates(original, ports, work, slots)
    assert_source(bound.circuit.instructions, expected)
    direct = bind_register_program(
        _resolve(original.decompose_bloq()), tuple(ports.values()), clean=work
    )
    assert_source(bound.circuit.instructions, direct.circuit.instructions)
    assert bound.outputs == tuple(ports.values())
    model = import_module("qualtran.resource_counting")
    cost = model.get_cost_value(original, model.QECGatesCost())
    primitives = Counter(g.name for g in bound.circuit.instructions)
    assert (
        primitives["t"] + primitives["tdg"] == cost.total_t_count() == 4 * (right - 1)
    )
    assert primitives["measure"] == primitives["reset"] == cost.measurement == right - 1


@pytest.mark.parametrize("kind", ("Add", "AddK", "Negate"))
@pytest.mark.parametrize("dtype", ("QUInt", "QInt", "QMontgomeryUInt"))
@pytest.mark.parametrize("width", (1, 2, 3))
def test_full_source_complex_amplitudes_every_history_clean_work_and_append(
    kind: str, dtype: str, width: int
) -> None:
    source = import_module("qualtran")
    arithmetic = import_module("qualtran.bloqs.arithmetic.addition")
    native_type = getattr(source, dtype)(width)
    if kind == "Negate" and dtype == "QInt" and width == 1:
        with pytest.raises(ValueError, match="Too-large"):
            _library(
                import_module("qualtran.bloqs.arithmetic.negate").Negate(native_type)
            )
        return
    original = (
        arithmetic.Add(native_type)
        if kind == "Add"
        else (
            arithmetic.AddK(native_type, -1)
            if kind == "AddK"
            else import_module("qualtran.bloqs.arithmetic.negate").Negate(native_type)
        )
    )
    logical = width * (2 if kind == "Add" else 1)
    scratch = width - 1 if kind == "Add" else 2 * width - 1
    ports = (
        (tuple(range(width)), tuple(range(width, logical)))
        if kind == "Add"
        else (tuple(range(width)),)
    )
    work = tuple(range(logical + 1, logical + 1 + scratch))
    bound = bind_register_program(_resolve(original), ports, clean=work)
    coeff = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coeff, 5 - 2 * coeff)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + scratch), dtype=torch.complex128)
    initial[:: 1 << scratch] = amplitudes
    matrix = torch.as_tensor(
        original.tensor_contract(), dtype=torch.complex128
    ).reshape(1 << logical, 1 << logical)
    expected = torch.zeros_like(initial)
    expected[:: 1 << scratch] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    for history in product((0, 1), repeat=width - 1):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected / 2 ** ((width - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << scratch)[:, 1:]) == 0
    for precision in (torch.complex64, torch.complex128):
        circuit = DynamicCircuit(
            logical + 1 + scratch, inputs=initial.to(precision), dtype=precision
        )
        circuit.append(FieldOperation(circuit.n_qubits, bound.circuit.instructions))
        result = run_dynamic(
            circuit, shots=8, seed=67, strategy="trajectory"
        ).final_states.reshape(8, -1)
        torch.testing.assert_close(
            result,
            expected.to(precision).expand(8, -1),
            atol=2e-6 if precision == torch.complex64 else 3e-12,
            rtol=0,
        )


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("include_msb", (False, True))
def test_complete_directional_output_isometry_all_erasure_branches_and_resources(
    width: int, include_msb: bool
) -> None:
    source = import_module("qualtran.bloqs.arithmetic.addition")
    original = source.OutOfPlaceAdder(width, include_most_significant_bit=include_msb)
    count = width + int(include_msb)
    data = (tuple(range(width)), tuple(range(width, 2 * width)))
    outputs = tuple(range(2 * width + 1, 2 * width + 1 + count))
    forward = bind_register_program(
        _resolve(original), data, clean=outputs[::-1], allow_workspace_outputs=True
    )
    backward = bind_register_program(
        _resolve(original.adjoint()),
        (*data, outputs),
        input_ownership=tuple((w, False) for w in outputs),
    )
    assert forward.outputs == (*data, outputs) and backward.outputs == data
    resource = import_module("qualtran.resource_counting")
    for op, bound, ports, allocations, returned in (
        (
            original,
            forward,
            {"a": data[0], "b": data[1]},
            outputs[::-1],
            {"a": data[0], "b": data[1], "c": outputs},
        ),
        (
            original.adjoint(),
            backward,
            {"a": data[0], "b": data[1], "c": outputs},
            (),
            {"a": data[0], "b": data[1]},
        ),
    ):
        slots = {
            g.wires[0]: int(g.metadata["classical_bit"])
            for g in bound.circuit.instructions
            if g.name == "measure"
        }
        expected_gates = reference_arithmetic_gates(
            op, ports, allocations, slots, expected_outputs=returned
        )
        assert_source(bound.circuit.instructions, expected_gates)
        direct = bind_register_program(
            _resolve(op.decompose_bloq()),
            tuple(ports.values()),
            clean=allocations,
            allow_workspace_outputs=not op.is_adjoint,
            input_ownership=tuple((w, False) for w in outputs) if op.is_adjoint else (),
        )
        assert_source(bound.circuit.instructions, direct.circuit.instructions)
        counts = Counter(g.name for g in bound.circuit.instructions)
        cost = resource.get_cost_value(op, resource.QECGatesCost())
        assert counts["t"] + counts["tdg"] == cost.total_t_count()
        assert counts["measure"] == counts["reset"] == cost.measurement
    coeff = torch.arange(1, (1 << (2 * width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coeff, 3 - coeff)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (2 * width + 1 + count), dtype=torch.complex128)
    initial[:: 1 << count] = amplitudes
    matrix = torch.as_tensor(
        original.tensor_contract(), dtype=torch.complex128
    ).reshape(1 << (2 * width + count), 1 << (2 * width))
    expected = (
        (matrix @ amplitudes.reshape(-1, 2))
        .reshape(1 << (2 * width), 1 << count, 2)
        .permute(0, 2, 1)
        .flatten()
    )
    circuit = Circuit(
        2 * width + 1 + count, inputs=initial, dtype=initial.dtype
    ).append(FieldOperation(2 * width + 1 + count, forward.circuit.instructions))
    actual = circuit.state().flatten()
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)
    for history in product((0, 1), repeat=count - 1):
        recovered = forced_branch(actual, backward.circuit.instructions, history)
        torch.testing.assert_close(
            recovered, initial / 2 ** ((count - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(recovered.reshape(-1, 1 << count)[:, 1:]) == 0


@pytest.mark.parametrize(
    "module,example",
    (
        *(
            ("addition", name)
            for name in (
                "_add_symb",
                "_add_small",
                "_add_large",
                "_add_diff_size_regs",
                "_add_oop_symb",
                "_add_oop_small",
                "_add_oop_large",
                "_add_k",
                "_add_k_small",
                "_add_k_large",
            )
        ),
        ("negate", "_negate"),
        ("negate", "_negate_symb"),
    ),
)
def test_all_complete_file_examples_at_all_serialized_depths(
    module: str, example: str
) -> None:
    original = getattr(
        import_module(f"qualtran.bloqs.arithmetic.{module}"), example
    ).make()
    for depth in (0, 1, 2):
        program = _resolve(original, depth)
        if "symb" in example or example == "_add_k":
            assert not program.functions[0].blocks
            with pytest.raises(ValueError, match="component bodies"):
                bind_register_program(program, ())


def test_always_preserves_real_nested_closure_and_rejects_invalid_arguments() -> None:
    source = import_module("qualtran")
    original = import_module("qualtran.bloqs.bookkeeping.always").Always(
        source.bloqs.arithmetic.Add(source.QUInt(2))
    )
    nested = type(original)(original)
    gates = []
    for op in (original.subbloq, original, nested):
        bound = bind_register_program(_resolve(op), ((0, 1), (2, 3)), clean=(4,))
        gates.append(bound.circuit.instructions)
    assert gates[0] == gates[1] == gates[2]
    factory = bookkeeping_factories()["qualtran.bloqs.bookkeeping.always.Always"]
    for arguments in ({}, {"subbloq": 1}, {"subbloq": FunctionRecord("x", ())}):
        with pytest.raises(SerializationError):
            factory("main", None, arguments)
    child = _resolve(original.subbloq).functions[0]
    with pytest.raises(SerializationError, match="signature"):
        factory("wrapper", (), {"subbloq": child})


def test_arithmetic_validation_symbolic_constants_and_injected_encoder_replacement() -> (
    None
):
    mapping = arithmetic_library_factories()
    prefix = "qualtran.bloqs.arithmetic.addition."
    dtype = unsigned_register_type(3)
    add = mapping[prefix + "Add"]
    constant = mapping[prefix + "AddK"]
    for arguments in (
        {},
        {"a_dtype": 1},
        {"a_dtype": opaque_register_type(3)},
        {"a_dtype": dtype, "unknown": 1},
    ):
        with pytest.raises(SerializationError):
            add("main", None, arguments)
    with pytest.raises(SerializationError, match="signature"):
        add("main", (), {"a_dtype": dtype})
    assert isinstance(add("main", None, {"a_dtype": dtype}), ProgramRecord)
    for value in (sympy.Symbol("k"), sympy.Integer(1)):
        result = constant("main", None, {"dtype": dtype, "k": value})
        assert isinstance(result, FunctionRecord) and not result.blocks
    for value in (8, "bad"):
        with pytest.raises(ValueError):
            constant("main", None, {"dtype": dtype, "k": value})

    def replacement(dtype: RegisterType, value: object) -> tuple[int, ...]:
        assert dtype == unsigned_register_type(3) and value == 7
        return (1, 1, 1)

    constant_args = {"dtype": dtype, "k": -1}
    assert constant("main", None, constant_args) == arithmetic_factories(
        encode_constant=replacement
    )[prefix + "AddK"]("main", None, constant_args)
    output = mapping[prefix + "OutOfPlaceAdder"]
    for invalid_args in (
        {},
        {"bitsize": True},
        {"bitsize": 3, "is_adjoint": 2},
        {"bitsize": 3, "include_most_significant_bit": "yes"},
        {"bitsize": 3, "unknown": 1},
    ):
        with pytest.raises(SerializationError):
            output("main", None, invalid_args)
    with pytest.raises(SerializationError, match="signature"):
        output("main", (), {"bitsize": 3})
    with pytest.raises(ValueError):
        output("main", None, {"bitsize": 0})


@pytest.mark.parametrize("dtype", ("QUInt", "QInt", "QMontgomeryUInt"))
def test_constant_admission_and_unsigned_negative_reduction_match_source(
    dtype: str,
) -> None:
    source = import_module("qualtran")
    addition = import_module("qualtran.bloqs.arithmetic.addition")
    original_dtype = getattr(source, dtype)(3)
    declaration = _library(addition.AddK(original_dtype, 0), 0).entries[0].declaration
    native_dtype = next(
        arg.value for arg in declaration.arguments if arg.name == "dtype"
    )
    assert isinstance(native_dtype, RegisterType)
    factory = arithmetic_library_factories()["qualtran.bloqs.arithmetic.addition.AddK"]
    for value in range(-17, 18):
        original = addition.AddK(original_dtype, value)
        try:
            bits = original._load_k_bloq._bits_k
        except ValueError:
            with pytest.raises(ValueError):
                factory("main", None, {"dtype": native_dtype, "k": value})
            continue
        program = factory("main", None, {"dtype": native_dtype, "k": value})
        assert isinstance(program, ProgramRecord)
        bound = bind_register_program(program, ((0, 1, 2),), clean=(3, 4, 5, 6, 7))
        primitives = Counter(g.name for g in bound.circuit.instructions)
        assert primitives["x"] == 2 * sum(bits)
