"""Repeated one-bit typed children reuse source-ordered quantum compatibility."""

from typing import Any

import pytest
import sympy
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_word_factories import word_factories
from flagquantum.compiler.registers.bits.on_each import on_each_word_program
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    classical_bit_type,
    fixed_point_register_type,
    montgomery_register_type,
    opaque_register_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_legacy_word_factories import _factories, _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KINDS = (
    "opaque",
    "unsigned",
    "signed",
    "bounded",
    "montgomery",
    "fixed",
    "negative_fixed",
    "field",
    "polynomial",
    "fixed_symbolic",
    "montgomery_symbolic",
)


def child_types(kind: str) -> tuple[Any, RegisterType]:
    from qualtran._infra.data_types import (
        QGF,
        BQUInt,
        QAny,
        QFxp,
        QGFPoly,
        QInt,
        QMontgomeryUInt,
        QUInt,
    )

    if kind == "opaque":
        return QAny(1), opaque_register_type(1)
    if kind == "unsigned":
        return QUInt(1), unsigned_register_type(1)
    if kind == "signed":
        return QInt(1), signed_register_type(1)
    if kind == "bounded":
        return BQUInt(1, 2), bounded_unsigned_register_type(1, 2)
    if kind.startswith("montgomery"):
        modulus = sympy.Symbol("p") if kind.endswith("symbolic") else None
        return QMontgomeryUInt(1, modulus), montgomery_register_type(1, modulus)
    if "fixed" in kind:
        fraction = (
            sympy.Symbol("f")
            if kind.endswith("symbolic")
            else -2 if kind == "negative_fixed" else 0
        )
        return QFxp(1, fraction), fixed_point_register_type(1, fraction)
    parameters = (("characteristic", 2), ("degree", 1), ("order", 2))
    if kind == "field":
        return QGF(2, 1), RegisterType("finite_field", 1, parameters=parameters)
    return QGFPoly(0, QGF(2, 1)), RegisterType(
        "finite_field_polynomial", 1, parameters=(("polynomial_degree", 0), *parameters)
    )


def source_child(dtype: Any) -> Any:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.registers import Register as OriginalRegister
    from qualtran._infra.registers import Signature
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.t_gate import TGate

    builder, inputs = BloqBuilder.from_signature(
        Signature((OriginalRegister("q", dtype),))
    )
    q = builder.add(Hadamard(), q=inputs["q"])
    q = builder.add(TGate(), q=q)
    return builder.finalize(q=q)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width", (1, 3))
def test_complete_one_bit_child_calls_gates_complex_operator_and_resource(
    kind: str, width: int
) -> None:
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    original_dtype, dtype = child_types(kind)
    child = source_child(original_dtype)
    source = OnEach(width, child)
    gates = (Instruction("h", (0,)), Instruction("t", (0,)))
    native_child = bit_gate_function(
        "child", (Register("q", dtype),), CircuitIR(1, gates)
    )
    program = on_each_word_program(width, native_child)
    assert verify_program_record(program) == ()
    root = program.functions[0]
    calls = [op for op in root.blocks[0].operations if op.opcode == "call"]
    assert len(calls) == width + 2
    assert [op.callee for op in calls[1:-1]] == ["child"] * width
    original_calls = sorted(
        source.decompose_bloq().bloq_instances, key=lambda instance: instance.i
    )
    assert [type(instance.bloq).__name__ for instance in original_calls] == [
        "Split",
        *(["CompositeBloq"] * width),
        "Join",
    ]
    wires = tuple(reversed(range(width)))
    bound = bind_register_program(program, (wires,), borrowed=(width,))
    assert bound.outputs == (wires,) and bound.events == ()
    assert bound.circuit.instructions == tuple(
        Instruction(op, (q,)) for q in wires for op in ("h", "t")
    )
    source_matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    # Reverse caller placement is represented by the same ordered source ports.
    actual = get_unitary(
        fq.Circuit(width, dtype=torch.complex128).append(
            FieldOperation(width, bound.circuit.instructions)
        )
    )
    torch.testing.assert_close(actual, source_matrix, rtol=0, atol=2e-12)
    initial = torch.randn(
        1 << (width + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(246),
    )
    initial /= initial.norm()
    circuit = fq.Circuit(width + 1, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 1, bound.circuit.instructions))
    expected = (source_matrix @ initial.reshape(1 << width, 2)).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    inverse_child = bit_gate_function(
        "inverse",
        (Register("q", dtype),),
        CircuitIR(1, (Instruction("tdg", (0,)), Instruction("h", (0,)))),
    )
    inverse = bind_register_program(
        on_each_word_program(width, inverse_child), (wires,), borrowed=(width,)
    )
    circuit.append(FieldOperation(width + 1, inverse.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == model.t == width and model.measurement == 0
    assert get_cost_value(source, QubitCount()) == width
    if kind in ("field", "polynomial"):
        with pytest.raises(TypeError, match="not recognized"):
            _library(source, 0)
    elif kind == "montgomery_symbolic":
        # The original codec drops modulus metadata; direct execution above is
        # independent evidence, not a claim of faithful source reconstruction.
        from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

        assert bloqs_from_proto(bloqs_to_proto(source, max_depth=1))[0] != source
    else:
        from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

        assert bloqs_from_proto(bloqs_to_proto(source, max_depth=2))[0] == source
        restored = resolve_component_program(_library(source, 2), 0, _factories())
        resolved = bind_register_program(restored, (wires,), borrowed=(width,))
        assert resolved.circuit.instructions == bound.circuit.instructions
        assert resolved.outputs == bound.outputs and resolved.events == ()


@pytest.mark.parametrize(
    "register",
    (
        Register("q", opaque_register_type(1), (1,)),
        Register("q", opaque_register_type(1), side=RegisterSide.LEFT),
        Register("q", opaque_register_type(1), side=RegisterSide.RIGHT),
        Register("q", opaque_register_type(2)),
        Register("r", opaque_register_type(1)),
        Register("q", classical_bit_type()),
    ),
)
def test_source_incompatible_child_shapes_directions_domains_and_widths_refuse(
    register: Register,
) -> None:
    child = FunctionRecord("child", (), registers=(register,))
    with pytest.raises(ValueError):
        on_each_word_program(3, child)
    factory = word_factories(encode_constant=lambda dtype, value: ())[
        "qualtran.bloqs.basic_gates.on_each.OnEach"
    ]
    with pytest.raises(SerializationError):
        factory("main", None, {"n": 3, "gate": child})


@pytest.mark.parametrize("depth", (0, 1, 2, 3))
def test_composite_child_transport_retains_actual_body_depth_boundary(
    depth: int,
) -> None:
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = OnEach(3, source_child(child_types("opaque")[0]))
    proto = bloqs_to_proto(source, max_depth=depth)
    restored = bloqs_from_proto(proto)[0]
    if depth < 2:
        assert restored != source
        with pytest.raises(KeyError, match="q"):
            restored.tensor_contract()
        with pytest.raises(SerializationError, match="Missing connection"):
            resolve_component_program(_library(source, depth), 0, _factories())
    else:
        assert restored == source
        bound = bind_register_program(
            resolve_component_program(_library(source, depth), 0, _factories()),
            ((2, 0, 1),),
        )
        assert bound.outputs == ((2, 0, 1),)
        assert bound.circuit.instructions == tuple(
            Instruction(op, (wire,)) for wire in (2, 0, 1) for op in ("h", "t")
        )
