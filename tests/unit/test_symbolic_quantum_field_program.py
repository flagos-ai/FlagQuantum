"""Symbolic field ports are logical requirements, never physical feasibility."""

from __future__ import annotations

from dataclasses import replace

import pytest
import sympy

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
    executable_register_width,
    logical_register_width,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tests.unit.test_logical_register_types import identity

pytestmark = pytest.mark.unit
P = sympy.Symbol("p", integer=True, positive=True)
M = sympy.Symbol("m", integer=True, positive=True)
N = sympy.Symbol("n", integer=True, nonnegative=True)
PROFILES = ((2, M, None), (P, 2, None), (P, M, None), (2, M, 2), (P, 2, 1), (P, M, N))


def declaration(
    characteristic: SymbolicInt,
    degree: SymbolicInt,
    polynomial_degree: SymbolicInt | None,
) -> RegisterType:
    field = FiniteFieldEncoding(characteristic, degree)
    return (
        field.register_type()
        if polynomial_degree is None
        else FiniteFieldPolynomialEncoding(polynomial_degree, field).register_type()
    )


@pytest.mark.parametrize("characteristic,degree,polynomial_degree", PROFILES)
@pytest.mark.parametrize("shape", ((), (2, 2), (0, 2)))
def test_declarations_and_typed_identity_calls_retain_exact_metadata_and_refuse_binding(
    characteristic: SymbolicInt,
    degree: SymbolicInt,
    polynomial_degree: SymbolicInt | None,
    shape: tuple[int, ...],
) -> None:
    dtype = declaration(characteristic, degree, polynomial_degree)
    assert logical_register_width(dtype) == dtype.quantum_bits
    function = replace(identity(dtype, shape=shape).functions[0], name="identity")
    assert verify_program_record(ProgramRecord((replace(function, blocks=()),))) == ()
    builder = ComponentProgramBuilder(
        Signature((Register("word", dtype, shape),)), (function,)
    )
    record = builder.finalize(builder.call_dict("identity", builder.inputs))
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == function.registers
    with pytest.raises(ValueError):
        executable_register_width(dtype)
    with pytest.raises(ValueError):
        bind_register_program(record, (), borrowed=(0,))
    classical = replace(
        dtype,
        quantum_bits=0,
        classical_bits=dtype.quantum_bits,
        parameters=(("storage", "classical"), *dtype.parameters),
    )
    with pytest.raises(ValueError):
        executable_classical_register_width(classical)
    assert verify_program_record(
        ProgramRecord(
            (FunctionRecord("decl", (), registers=(Register("c", classical),)),)
        )
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("characteristic,degree,polynomial_degree", PROFILES)
def test_all_six_unmodified_source_identity_graphs(
    characteristic: SymbolicInt,
    degree: SymbolicInt,
    polynomial_degree: SymbolicInt | None,
) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QGF, QGFPoly
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.testing import assert_valid_cbloq

    field = QGF(characteristic, degree)
    source_dtype = (
        field if polynomial_degree is None else QGFPoly(polynomial_degree, field)
    )
    dtype = declaration(characteristic, degree, polynomial_degree)
    assert dtype.quantum_bits == source_dtype.num_qubits
    builder, inputs = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("word", source_dtype),))
    )
    original = builder.finalize(**inputs)
    assert_valid_cbloq(original)
    assert len(original.bloq_instances) == 0
    assert verify_program_record(identity(dtype)) == ()
    assert dict(dtype.parameters)["order"] == field.order


def test_malformed_and_partly_concrete_invalid_metadata_still_fail() -> None:
    base = declaration(P, M, N)
    parameters = dict(base.parameters)
    for invalid in (
        replace(base, quantum_bits=base.quantum_bits + 1),
        replace(base, quantum_bits=True),
        replace(base, classical_bits=True),
        replace(base, classical_bits=1),
        replace(base, parameters=base.parameters[::-1]),
        replace(base, parameters=base.parameters + (("degree", M),)),
        *(
            replace(
                base,
                parameters=tuple(
                    (name, value if name == target else old)
                    for name, old in base.parameters
                ),
            )
            for target, value in (
                ("polynomial_degree", -1),
                ("polynomial_degree", True),
                ("characteristic", 1),
                ("degree", 0),
                ("order", 2),
                ("degree", sympy.Rational(1, 2)),
            )
        ),
    ):
        assert verify_program_record(identity(invalid))
    assert parameters["polynomial_degree"] == N


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("characteristic,degree,polynomial_degree", PROFILES)
@pytest.mark.parametrize("dirty", (False, True))
def test_source_symbolic_field_lifetime_calls_remain_logical(
    characteristic: SymbolicInt,
    degree: SymbolicInt,
    polynomial_degree: SymbolicInt | None,
    dirty: bool,
) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QGF, QGFPoly
    from qualtran.bloqs.bookkeeping.allocate import Allocate
    from qualtran.bloqs.bookkeeping.free import Free
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.serialization.bloq import bloqs_to_proto
    from qualtran.testing import assert_valid_cbloq

    field = QGF(characteristic, degree)
    source_dtype = (
        field if polynomial_degree is None else QGFPoly(polynomial_degree, field)
    )
    allocation, release = Allocate(source_dtype, dirty), Free(source_dtype, dirty)
    source = BloqBuilder()
    source_word = source.add(allocation)
    source.add(release, reg=source_word)
    reference = source.finalize()
    assert_valid_cbloq(reference)
    assert reference.bloq_counts() == {allocation: 1, release: 1}
    assert get_cost_value(reference, QECGatesCost()).asdict() == {}
    assert allocation.adjoint() == release and release.adjoint() == allocation
    with pytest.raises(TypeError, match="is not recognized"):
        bloqs_to_proto(reference)

    dtype = declaration(characteristic, degree, polynomial_degree)
    builder = ComponentProgramBuilder(Signature(()))
    native_word = builder.allocate(dtype=dtype, dirty=dirty)
    builder.free(native_word, dirty=dirty)
    program = builder.finalize({})
    assert verify_program_record(program) == ()
    assert len(program.functions[0].blocks[0].operations) == 2
    operations = tuple(
        op for f in program.functions[1:] for b in f.blocks for op in b.operations
    )
    assert [op.opcode for op in operations] == ["allocate_register", "free_register"]
    assert all(op.workspace_dirty is dirty for op in operations)
    assert program.functions[1].registers is not None
    assert program.functions[2].registers is not None
    assert (
        program.functions[1].registers[0].dtype
        == program.functions[2].registers[0].dtype
        == dtype
    )
    with pytest.raises(ValueError):
        bind_register_program(program, (), borrowed=(0,))
