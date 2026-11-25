"""Proposal 125 preserves literal raw field storage and numerical boundaries."""

import builtins
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import sympy
import torch
from qualtran import QGF, QGFPoly
from qualtran.bloqs.bookkeeping import Join, Split
from qualtran.resource_counting import QECGatesCost, get_cost_value
from qualtran.serialization.bloq import bloqs_to_proto

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_component import (
    field_polynomial_component_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.register_verifier import executable_register_width
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tools.check_dependency_policy import load_toml

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PROFILES = ((3, 1), (5, 1), (3, 2))


def test_approved_contract_supersedes_only_the_quantum_storage_limit() -> None:
    contracts = Path(__file__).resolve().parents[2] / "contracts"
    approved = load_toml(contracts / "nonbinary-quantum-field-storage-contract.toml")
    previous = load_toml(contracts / "nonbinary-classical-field-contract.toml")
    assert approved["approval"] == "explicit_user_approval_2026-11-25"
    assert approved["public_methods_added"] == []
    assert approved["admission"]["storage"] == "quantum_raw_bits"
    assert approved["boundaries"]["classical_semantics"] == "unchanged"
    assert previous["admission"]["quantum_field_scope"] == "existing_binary_only"


@pytest.mark.parametrize("characteristic,degree", PROFILES)
@pytest.mark.parametrize("polynomial", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_source_raw_operator_aliases_and_native_source_free_append(
    characteristic: int,
    degree: int,
    polynomial: bool,
    inverse: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    field = FiniteFieldEncoding(characteristic, degree)
    dtype = (
        FiniteFieldPolynomialEncoding(1, field).register_type()
        if polynomial
        else field.register_type()
    )
    source_field = QGF(characteristic, degree)
    source_dtype = QGFPoly(1, source_field) if polynomial else source_field
    source = (Join if inverse else Split)(source_dtype)
    width = executable_register_width(dtype)
    assert width == source_dtype.num_qubits
    function = (join_word_function if inverse else split_word_function)("view", dtype)
    assert function.registers is not None
    assert [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in function.registers
    ] == [(r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in source.signature]
    assert any(reg.dtype is dtype for reg in function.registers)
    parameters = dict(dtype.parameters)
    assert parameters["characteristic"] == source_field.characteristic
    assert parameters["degree"] == source_field.degree
    assert parameters["order"] == source_field.order
    assert parameters["polynomial"] == int(source_field.irreducible_poly)
    program = ProgramRecord((function,))
    assert verify_program_record(program) == ()
    wires = tuple(reversed(range(width)))
    inputs = tuple((w,) for w in wires) if inverse else (wires,)
    bound = bind_register_program(program, inputs, entry="view")
    assert bound.outputs == ((wires,) if inverse else tuple((w,) for w in wires))
    assert not bound.circuit.instructions and not bound.events
    operation, aliases = source.as_cirq_op(
        None, reg=np.asarray(wires).reshape(width, 1) if inverse else np.asarray(wires)
    )
    assert operation is None and tuple(aliases["reg"].flat) == wires
    matrix = torch.as_tensor(
        source.tensor_contract().reshape(1 << width, 1 << width), dtype=torch.complex128
    )
    torch.testing.assert_close(
        matrix, torch.eye(1 << width, dtype=torch.complex128), atol=0, rtol=0
    )
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    with pytest.raises(TypeError, match="QGF"):
        bloqs_to_proto(source, max_depth=0)
    generator = torch.Generator().manual_seed(125 + width)
    before = torch.randn(1 << (width + 1), dtype=torch.complex128, generator=generator)
    before /= before.norm()
    order = (*wires, width)
    axes = tuple(order.index(i) for i in range(width + 1))
    expected = (
        (
            matrix
            @ before.reshape((2,) * (width + 1)).permute(order).reshape(1 << width, 2)
        )
        .reshape((2,) * (width + 1))
        .permute(axes)
        .flatten()
    )
    original_import = builtins.__import__

    def source_free(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "galois", "kickmix"}:
            raise AssertionError(f"Raw field execution imported {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", source_free)
        artifact = CircuitIR.from_json(bound.circuit.to_json())
        actual = (
            Circuit(width + 1, inputs=before, dtype=torch.complex128)
            .append(FieldOperation(width + 1, artifact.instructions))
            .state()
            .flatten()
        )
        torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0)


@pytest.mark.parametrize("characteristic,degree", PROFILES)
def test_packed_polynomial_coefficient_views_retain_nominal_order(
    characteristic: int, degree: int
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolySplit

    field = FiniteFieldEncoding(characteristic, degree)
    coefficient = field.register_type()
    width = executable_register_width(coefficient)
    packed = FiniteFieldPolynomialEncoding(1, field).register_type()
    layout = FieldPolynomialLayout(1, width)
    wires = tuple(reversed(range(2 * width)))
    split = field_polynomial_component_program(layout, coefficient, packed_dtype=packed)
    bound = bind_register_program(split, (wires,))
    assert bound.outputs == (wires[:width], wires[width:])
    source = GFPolySplit(QGFPoly(1, QGF(characteristic, degree)))
    operation, ports = source.as_cirq_op(None, reg=np.asarray(wires))
    assert operation is None
    assert bound.outputs == tuple(tuple(word) for word in ports["reg"])
    join = field_polynomial_component_program(
        layout, coefficient, reverse=True, packed_dtype=packed
    )
    restored = bind_register_program(join, bound.outputs)
    assert restored.outputs == (wires,)
    operation, ports = source.adjoint().as_cirq_op(None, reg=ports["reg"])
    assert operation is None and tuple(ports["reg"]) == wires
    assert not bound.events and not restored.events
    assert bound.circuit.instructions == restored.circuit.instructions == ()


@pytest.mark.parametrize("characteristic,degree", PROFILES)
@pytest.mark.parametrize("polynomial", (False, True))
def test_malformed_symbolic_and_classical_metadata_stay_refused(
    characteristic: int, degree: int, polynomial: bool
) -> None:
    field = FiniteFieldEncoding(characteristic, degree)
    encoding = FiniteFieldPolynomialEncoding(1, field) if polynomial else field
    dtype = encoding.register_type()
    invalid = (
        replace(dtype, quantum_bits=dtype.quantum_bits + 1),
        replace(dtype, quantum_bits=True),
        replace(dtype, quantum_bits=sympy.Symbol("n")),
        replace(dtype, classical_bits=1),
        replace(dtype, parameters=dtype.parameters[::-1]),
        replace(
            dtype,
            parameters=tuple(
                (name, False if name == "characteristic" else value)
                for name, value in dtype.parameters
            ),
        ),
        encoding.register_type(classical=True),
    )
    for bad in invalid:
        with pytest.raises(ValueError, match="unsupported executable"):
            executable_register_width(bad)


@pytest.mark.parametrize("characteristic,degree", PROFILES)
def test_numeric_decode_still_refuses_unused_raw_encodings(
    characteristic: int, degree: int
) -> None:
    field = FiniteFieldEncoding(characteristic, degree)
    source = QGF(characteristic, degree)
    width = executable_register_width(field.register_type())
    bits = [(int(field.order) >> i) & 1 for i in reversed(range(width))]
    with pytest.raises(ValueError):
        field.from_bits(bits)
    with pytest.raises(ValueError):
        source.from_bits(bits)


@pytest.mark.parametrize("characteristic,degree", PROFILES)
@pytest.mark.parametrize("polynomial", (False, True))
def test_raw_field_allocation_and_free_preserve_the_complete_spectator_state(
    characteristic: int, degree: int, polynomial: bool
) -> None:
    from qualtran import BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.resource_counting import QubitCount

    field = FiniteFieldEncoding(characteristic, degree)
    dtype = (
        FiniteFieldPolynomialEncoding(1, field).register_type()
        if polynomial
        else field.register_type()
    )
    source_dtype = (
        QGFPoly(1, QGF(characteristic, degree))
        if polynomial
        else QGF(characteristic, degree)
    )
    width = executable_register_width(dtype)
    leaves = tuple(
        bit_gate_function(
            gate,
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction(gate, (0,)),)),
        )
        for gate in ("h", "t")
    )
    builder = ComponentProgramBuilder(Signature.build(s=1), leaves)
    word = builder.allocate(dtype=dtype)
    builder.free(word)
    s = builder.call_dict("h", {"q": builder.inputs["s"]})["q"]
    s = builder.call_dict("t", {"q": s})["q"]
    program = builder.finalize({"s": s})
    assert verify_program_record(program) == ()
    bound = bind_register_program(program, ((width,),), clean=tuple(range(width)))
    assert bound.outputs == ((width,),) and len(bound.events) == 2
    assert all(width not in event.wires for event in bound.events)
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == [
        ("h", (width,)),
        ("t", (width,)),
    ]
    source_builder, inputs = BloqBuilder.from_signature(SourceSignature.build(s=1))
    temporary = source_builder.allocate(dtype=source_dtype)
    source_builder.free(temporary)
    source_s = source_builder.add(Hadamard(), q=inputs["s"])
    source_s = source_builder.add(TGate(), q=source_s)
    source = source_builder.finalize(s=source_s)
    spectator = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128) / 15**0.5
    before = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    before[:2] = spectator
    expected = torch.zeros_like(before)
    expected[:2] = (
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128) @ spectator
    )
    actual = (
        Circuit(width + 1, inputs=before, dtype=torch.complex128)
        .append(FieldOperation(width + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0)
    model = get_cost_value(source, QECGatesCost())
    assert model.t == model.clifford == 1 and model.measurement == 0
    # Retain the original DAG estimate separately from this explicitly live
    # caller spectator and the supplied physical work pool.
    assert get_cost_value(source, QubitCount()) == width
    assert bound.circuit.n_wires == width + 1
