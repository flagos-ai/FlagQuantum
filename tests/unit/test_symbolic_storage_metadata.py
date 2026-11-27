"""Known-size raw storage retains unresolved interpretation parameters."""

from dataclasses import replace
from typing import Any

import pytest
import sympy
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.control.component_predicate import (
    validate_control_ports,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    classical_bounded_unsigned_register_type,
    fixed_point_register_type,
    montgomery_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.register_verifier import (
    concrete_logical_register_width,
    executable_classical_register_width,
    executable_register_width,
)
from flagquantum.core.ir.program.registers import Register, RegisterType, TypeParameter
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.errors import SerializationError
from tests.unit.register_source_workspace import SourceWorkspace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
P = sympy.Symbol("p", integer=True, positive=True)
F = sympy.Symbol("f", integer=True)


def storage_pair(kind: str) -> tuple[Any, RegisterType]:
    from qualtran._infra.data_types import BQUInt, QFxp, QMontgomeryUInt

    if kind == "montgomery":
        return QMontgomeryUInt(3, P), montgomery_register_type(3, P)
    if kind == "fixed":
        return QFxp(3, F), fixed_point_register_type(3, F)
    return BQUInt(3, P), bounded_unsigned_register_type(3, P)


@pytest.mark.parametrize("kind", ("montgomery", "fixed", "bounded"))
@pytest.mark.parametrize("dirty", (False, True))
def test_source_lifetimes_cast_and_append_preserve_entangled_state(
    kind: str, dirty: bool
) -> None:
    from qualtran.bloqs.bookkeeping.allocate import Allocate
    from qualtran.bloqs.bookkeeping.free import Free
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    original, dtype = storage_pair(kind)
    assert (
        executable_register_width(dtype) == concrete_logical_register_width(dtype) == 3
    )
    functions = (
        allocate_word_function("allocate", dtype, dirty),
        free_word_function("free", dtype, dirty),
        cast_word_function("out", dtype, opaque_register_type(3)),
        cast_word_function("back", opaque_register_type(3), dtype),
    )
    builder = ComponentProgramBuilder(Signature(()), functions)
    (word,) = builder.call_tuple("allocate", {})
    (raw,) = builder.call_tuple("out", {"reg": word})
    (word,) = builder.call_tuple("back", {"reg": raw})
    builder.call_tuple("free", {"reg": word})
    record = builder.finalize({})
    assert verify_program_record(record) == ()
    wires = (2, 0, 1)
    bound = bind_register_program(
        record, (), clean=() if dirty else wires, borrowed=wires if dirty else ()
    )
    assert bound.outputs == () and bound.circuit.instructions == ()
    manager = SourceWorkspace(() if dirty else wires, wires if dirty else ())
    source_alloc, source_free = Allocate(original, dirty), Free(original, dirty)
    operation, outputs = source_alloc.as_cirq_op(manager)
    assert operation is None
    assert tuple(int(q.x) for q in outputs["reg"]) == wires
    assert source_free.as_cirq_op(manager, **outputs) == (None, {})
    assert manager.live == {}
    assert manager.events == [(e.kind, e.wires, e.dirty) for e in bound.events]
    assert all(e.wires == wires and e.dirty == dirty for e in bound.events)
    for source in (source_alloc, source_free):
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
        assert get_cost_value(source, QubitCount()) == 3
    initial = torch.randn(
        16, dtype=torch.complex128, generator=torch.Generator().manual_seed(130)
    )
    if not dirty:
        initial[2:] = 0
    initial /= initial.norm()
    circuit = fq.Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(4, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    assert functions[0].registers is not None
    assert functions[0].registers[0].dtype is dtype


@pytest.mark.parametrize("kind", ("montgomery", "fixed", "bounded"))
def test_actual_constructor_storage_and_source_split_refusal(kind: str) -> None:
    from qualtran.bloqs.bookkeeping.join import Join
    from qualtran.bloqs.bookkeeping.split import Split

    source, dtype = storage_pair(kind)
    factories = bookkeeping_factories()
    prefix = "qualtran.bloqs.bookkeeping."
    constructor = factories[prefix + "cast.Cast"]
    cast = constructor(
        "main", None, {"inp_dtype": dtype, "out_dtype": opaque_register_type(3)}
    )
    assert isinstance(cast, FunctionRecord)
    assert cast.blocks
    bound = bind_register_program(ProgramRecord((cast,)), ((2, 0, 1),))
    assert bound.outputs == ((2, 0, 1),) and bound.circuit.instructions == ()
    for name, original in (("Split", Split), ("Join", Join)):
        factory = factories[prefix + name.lower() + "." + name]
        if kind != "montgomery":
            with pytest.raises(ValueError):
                original(source)
            with pytest.raises(SerializationError, match="symbolic source dtype"):
                factory("main", None, {"dtype": dtype})
        else:
            source_view = original(source)
            function = factory("main", None, {"dtype": dtype})
            assert isinstance(function, FunctionRecord)
            inputs = ((2,), (0,), (1,)) if name == "Join" else ((2, 0, 1),)
            view = bind_register_program(ProgramRecord((function,)), inputs)
            assert view.outputs == (
                ((2, 0, 1),) if name == "Join" else ((2,), (0,), (1,))
            )
            assert source_view.dtype == source


@pytest.mark.parametrize(
    "dtype",
    (
        opaque_register_type(sympy.Integer(3)),
        bounded_unsigned_register_type(sympy.Integer(3), sympy.Integer(5)),
        fixed_point_register_type(sympy.Integer(3), sympy.Integer(1), True),
        montgomery_register_type(sympy.Integer(3), P),
    ),
)
def test_exact_constants_preserve_descriptor_identity(dtype: RegisterType) -> None:
    assert (
        executable_register_width(dtype) == concrete_logical_register_width(dtype) == 3
    )
    function = cast_word_function("main", dtype, opaque_register_type(3))
    bound = bind_register_program(ProgramRecord((function,)), ((2, 0, 1),))
    assert bound.outputs == ((2, 0, 1),)
    assert function.registers is not None and function.registers[0].dtype is dtype


@pytest.mark.parametrize(
    "dtype",
    (
        RegisterType("opaque", sympy.Rational(3, 2)),
        RegisterType("opaque", sympy.oo),
        RegisterType("opaque", sympy.Symbol("n", integer=True)),
        RegisterType("opaque", True),
        RegisterType("uint", 3, parameters=(("bound", sympy.Integer(9)),)),
        RegisterType(
            "fixed_point", 1, parameters=(("fractional_width", F), ("signed", True))
        ),
        RegisterType(
            "fixed_point",
            3,
            parameters=(("fractional_width", sympy.Integer(4)), ("signed", False)),
        ),
        RegisterType(
            "fixed_point",
            3,
            parameters=(("fractional_width", sympy.Rational(1, 2)), ("signed", False)),
        ),
        RegisterType(
            "uint", 3, parameters=(("bound", sympy.Symbol("bad", integer=False)),)
        ),
    ),
)
def test_unknown_sizes_and_known_invalid_metadata_still_refuse(
    dtype: RegisterType,
) -> None:
    with pytest.raises(ValueError):
        executable_register_width(dtype)


@pytest.mark.parametrize("kind", ("fixed", "bounded"))
def test_symbolic_source_control_refusal_and_classical_boundary(kind: str) -> None:
    _, dtype = storage_pair(kind)
    with pytest.raises(ValueError, match="nonsymbolic source dtype"):
        validate_control_ports((Register("ctrl", dtype),), (ControlRegister(3, (1,)),))
    classical = classical_bounded_unsigned_register_type(3, P)
    with pytest.raises(ValueError, match="concrete integral metadata"):
        executable_classical_register_width(classical)
    with pytest.raises(ValueError):
        executable_register_width(classical)
    with pytest.raises(ValueError):
        executable_register_width(replace(dtype, classical_bits=1))


@pytest.mark.parametrize("activation", (0, 7))
def test_symbolic_modulus_control_matches_every_gate_and_erasure_branch(
    activation: int,
) -> None:
    from itertools import product

    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QMontgomeryUInt
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from flagquantum.compiler.registers.control.component_predicate import (
        control_predicate_program,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = CtrlSpecAnd(CtrlSpec(QMontgomeryUInt(3, P), activation))
    name = source.control_registers[0].name
    controls = (Register(name, montgomery_register_type(3, P)),)
    patterns = (ControlRegister(3, (activation,)),)
    forward = control_predicate_program(controls, patterns, name="compute")
    erase = control_predicate_program(controls, patterns, name="erase", uncompute=True)
    builder = ComponentProgramBuilder(Signature(controls))
    computed = builder.call_program(forward, builder.inputs, entry="compute")
    returned = builder.call_program(erase, computed, entry="erase")
    bound = bind_register_program(
        builder.finalize(returned), ((0, 1, 2),), clean=(3, 4)
    )
    assert bound.outputs == ((0, 1, 2),)
    gates = bound.circuit.instructions
    measurements = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    inputs: dict[str, tuple[int, ...]] = {name: (0, 1, 2)}
    outputs = {**inputs, "junk": (3,), "target": (4,)}
    reference = reference_arithmetic_gates(
        source, inputs, (3, 4), measurements, expected_outputs=outputs
    )
    reference += reference_arithmetic_gates(
        source.adjoint(), outputs, (3, 4), measurements, expected_outputs=inputs
    )
    assert_source(gates, reference)
    initial = torch.zeros(64, dtype=torch.complex128)
    initial.reshape(8, 4, 2)[:, 0, :] = torch.randn(
        8, 2, dtype=torch.complex128, generator=torch.Generator().manual_seed(130)
    )
    initial /= initial.norm()
    for outcomes in product((0, 1), repeat=2):
        branch = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            branch, forced_branch(initial, reference, outcomes), atol=2e-12, rtol=0
        )
        torch.testing.assert_close(branch, initial / 2, atol=2e-12, rtol=0)
        assert torch.count_nonzero(branch.reshape(8, 4, 2)[:, 1:, :].abs() > 2e-12) == 0
    # Primitive inventories and analytical models are checked independently.
    assert sum(g.name in ("t", "tdg") for g in gates) == 8
    assert sum(g.name == "measure" for g in gates) == 2
    assert get_cost_value(source, QECGatesCost()).total_t_count() == 8
    assert get_cost_value(source.adjoint(), QECGatesCost()).measurement == 2


@pytest.mark.parametrize("kind", ("montgomery", "fixed", "bounded"))
@pytest.mark.parametrize("shape", ((), (2,), (0, 2)))
@pytest.mark.parametrize("left_only", (False, True))
def test_shaped_partition_wrapper_executes_known_size_symbolic_metadata(
    kind: str, shape: tuple[int, ...], left_only: bool
) -> None:
    from math import prod

    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QAny, QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.t_gate import TGate
    from qualtran.bloqs.bookkeeping.auto_partition import (
        AutoPartition as SourceAutoPartition,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
    from flagquantum.compiler.registers.layout.component_auto_partition import (
        auto_partition_component_program,
    )
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.simulation.unitary import get_unitary

    source_dtype, dtype = storage_pair(kind)
    width = 3 * prod(shape)
    original, inputs = BloqBuilder.from_signature(
        SourceSignature(
            (SourceRegister("q", source_dtype, shape), SourceRegister("s", QBit()))
        )
    )
    spectator = original.add(Hadamard(), q=inputs["s"])
    spectator = original.add(TGate(), q=spectator)
    child = original.finalize(q=inputs["q"], s=spectator)
    source = SourceAutoPartition(
        child,
        (
            (SourceRegister("a", QAny(width)), ("q",)),
            (SourceRegister("b", QAny(1)), ("s",)),
        ),
        left_only=left_only,
    )
    leaves = tuple(
        bit_gate_function(
            op,
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction(op, (0,)),)),
        )
        for op in ("h", "t")
    )
    builder = ComponentProgramBuilder(
        Signature((Register("q", dtype, shape), Register("s", quantum_bit_type()))),
        leaves,
    )
    spectator = builder.call_dict("h", {"q": builder.inputs["s"]})["q"]
    spectator = builder.call_dict("t", {"q": spectator})["q"]
    native = builder.finalize({"q": builder.inputs["q"], "s": spectator})
    module = auto_partition_component_program(
        native,
        AutoPartition(
            (("a", (("q", width),)), ("b", (("s", 1),))), left_only=left_only
        ),
    )
    packed = next(f for f in module.functions if f.name == "packed")
    assert packed.registers is not None
    assert [
        (r.name, r.total_bits(), r.shape_symbolic, r.side.value)
        for r in packed.registers
    ] == [
        (r.name, r.total_bits(), r.shape_symbolic, r.side.value)
        for r in source.signature
    ]
    calls = [op for op in packed.blocks[0].operations if op.opcode == "call"]
    assert (
        len(calls)
        == (3 if left_only else 5)
        == len(source.decompose_bloq().bloq_instances)
    )
    wires = tuple(range(width))
    bound = bind_register_program(module, (wires, (width,)), entry="packed")
    expected = tuple(wires[i : i + 3] for i in range(0, width, 3)) + ((width,),)
    assert bound.outputs == (expected if left_only else (wires, (width,)))
    assert bound.events == ()
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == [
        ("h", (width,)),
        ("t", (width,)),
    ]
    actual = get_unitary(
        fq.Circuit(width + 1, dtype=torch.complex128).append(
            FieldOperation(width + 1, bound.circuit.instructions)
        )
    )
    torch.testing.assert_close(
        actual,
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == model.t == 1 and model.measurement == 0


@pytest.mark.parametrize("polynomial", (False, True))
def test_exact_field_constants_prove_size_without_changing_classical_admission(
    polynomial: bool,
) -> None:
    parameters: tuple[tuple[str, TypeParameter], ...] = (
        ("characteristic", sympy.Integer(2)),
        ("degree", sympy.Integer(3)),
        ("order", sympy.Integer(8)),
    )
    width = 6 if polynomial else 3
    if polynomial:
        parameters = (("polynomial_degree", sympy.Integer(1)), *parameters)
    dtype = RegisterType(
        "finite_field_polynomial" if polynomial else "finite_field",
        sympy.Integer(width),
        parameters=parameters,
    )
    assert executable_register_width(dtype) == width
    classical = RegisterType(
        dtype.encoding, 0, width, parameters=(("storage", "classical"), *parameters)
    )
    with pytest.raises(ValueError):
        executable_classical_register_width(classical)
