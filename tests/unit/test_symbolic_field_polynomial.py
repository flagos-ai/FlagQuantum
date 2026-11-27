"""Original symbolic polynomial calls remain typed declarations, never fake gates."""

from dataclasses import replace
from typing import Any

import pytest
from sympy import Integer, Symbol

from examples.algorithms.field_polynomial_decomposition import (
    field_polynomial_child,
    field_polynomial_program,
    field_polynomial_score,
)
from flagquantum.compiler.registers.finite_field.polynomial_addition_component import (
    logical_field_polynomial_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
M = Symbol("m", positive=True, integer=True)


@pytest.mark.parametrize("degree", (0, 1, 4))
@pytest.mark.parametrize("field_degree", (M, M + 1, 2 * M, Integer(3)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_original_symbolic_graph_connections_and_declarations(
    degree: int, field_degree: Any, adjoint: bool
) -> None:
    from qualtran._infra.data_types import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import GF2PolyAdd

    field = FiniteFieldEncoding(2, field_degree)
    packed = FiniteFieldPolynomialEncoding(degree, field).register_type()
    program = logical_field_polynomial_addition_program(
        degree, field.register_type(), packed, adjoint=adjoint
    )
    original: Any = GF2PolyAdd(QGFPoly(degree, QGF(2, field_degree)))
    if adjoint:
        original = original.adjoint()
    composite = original.decompose_bloq()
    assert verify_program_record(program) == ()
    assert _native_trace(program) == _source_trace(SourceCalls, composite)
    definitions = {f.name: f for f in program.functions}
    operations = program.functions[0].blocks[0].operations
    assert len(operations) == degree + 5
    assert all(not f.blocks for f in program.functions[1:])
    assert len({op.callee for op in operations}) == 3
    for native, (source, _, _) in zip(
        operations, composite.iter_bloqsoqs(), strict=True
    ):
        assert native.callee is not None
        registers = definitions[native.callee].registers
        assert registers is not None
        assert [(r.name, r.bitsize, r.shape, r.side.value) for r in registers] == [
            (r.name, r.bitsize, r.shape, r.side.value) for r in source.bloq.signature
        ]
    if not adjoint:
        signature = Signature.build_from_dtypes(f_x=packed, g_x=packed)
        assert field_polynomial_program("GF2PolyAdd", signature, {}) == program
        assert field_polynomial_score("GF2PolyAdd", signature, {}) is not None
    # Unknown width or a signature-only child must fail binding, including
    # symbolic integer literals whose raw storage width can otherwise resolve.
    if field_degree == Integer(3):
        width = 3 * (degree + 1)
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_register_program(
                program, (tuple(range(width)), tuple(range(width, 2 * width)))
            )
    else:
        with pytest.raises(ValueError):
            bind_register_program(program, ((0,), (1,)))


@pytest.mark.parametrize("different_field", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_symbolic_integer_constant_preserves_actual_source_values_and_sharing(
    different_field: bool, adjoint: bool
) -> None:
    import galois
    from qualtran._infra.data_types import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import GF2PolyAddK

    field = FiniteFieldEncoding(2, Integer(3))
    packed = FiniteFieldPolynomialEncoding(2, field).register_type()
    constant = galois.Poly([1, 2], field=galois.GF(5 if different_field else 8))
    original: Any = GF2PolyAddK(QGFPoly(2, QGF(2, Integer(3))), constant)
    if adjoint:
        original = original.adjoint()
    program = logical_field_polynomial_addition_program(
        2, field.register_type(), packed, constants=(0, 1, 2), adjoint=adjoint
    )
    composite = original.decompose_bloq()
    assert _native_trace(program) == _source_trace(SourceCalls, composite)
    assert all(not f.blocks for f in program.functions[1:])
    constant_entries = [
        f.name for f in program.functions if f.name.startswith("field_add_constant_")
    ]
    assert len(constant_entries) == 3
    assert all(name.endswith("_adjoint") == adjoint for name in constant_entries)
    assert len({op.callee for op in program.functions[0].blocks[0].operations}) == 5
    if not adjoint:
        signature = Signature.build_from_dtypes(f_x=packed)
        actual = field_polynomial_program("GF2PolyAddK", signature, {"g_x": constant})
        assert actual == program
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_register_program(actual, (tuple(range(9)),))


def test_opaque_function_names_keep_constant_roles_and_operand_changes_refuse() -> None:
    import galois

    field = FiniteFieldEncoding(2, Integer(3))
    packed = FiniteFieldPolynomialEncoding(2, field).register_type()
    ports = Signature.build_from_dtypes(f_x=packed)
    parameters = {"g_x": galois.Poly([1, 2], field=galois.GF(8))}
    program = field_polynomial_program("GF2PolyAddK", ports, parameters)
    assert program is not None
    names = {f.name: f"opaque_{i}" for i, f in enumerate(program.functions)}
    renamed = replace(
        program,
        functions=tuple(
            replace(
                f,
                name=names[f.name],
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(
                            replace(op, callee=names[op.callee]) if op.callee else op
                            for op in b.operations
                        ),
                    )
                    for b in f.blocks
                ),
            )
            for f in program.functions
        ),
    )
    selected = []
    for function in program.functions:
        before = field_polynomial_child(program, function.name, ports, parameters)
        after = field_polynomial_child(renamed, names[function.name], ports, parameters)
        assert (None if before is None else before[:2]) == (
            None if after is None else after[:2]
        )
        if before is not None:
            selected.append(before[1].functions[0].name)
    assert set(selected) == {
        "field_add_constant_0",
        "field_add_constant_1",
        "field_add_constant_2",
    }

    root = program.functions[0]
    block = root.blocks[0]
    # Swapping field-addition call positions preserves word types and SSA
    # validity here but breaks the complete source coefficient ordering.
    operations = list(block.operations)
    operations[1], operations[2] = operations[2], operations[1]
    changed = replace(
        program,
        functions=(
            replace(root, blocks=(replace(block, operations=tuple(operations)),)),
            *program.functions[1:],
        ),
    )
    assert verify_program_record(changed) == ()
    for function in program.functions:
        if (
            field_polynomial_child(program, function.name, ports, parameters)
            is not None
        ):
            assert (
                field_polynomial_child(changed, function.name, ports, parameters)
                is None
            )


@pytest.mark.parametrize("field_degree", (Symbol("m"), Symbol("m", integer=True)))
def test_original_nominal_symbolic_binding_failure_is_not_an_executable_body(
    field_degree: Any,
) -> None:
    from qualtran._infra.composite_bloq import BloqError
    from qualtran._infra.data_types import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import GF2PolyAdd

    original = GF2PolyAdd(QGFPoly(2, QGF(2, field_degree)))
    with pytest.raises(BloqError, match="dtypes are not consistent"):
        original.decompose_bloq()
    packed = FiniteFieldPolynomialEncoding(
        2, FiniteFieldEncoding(2, field_degree)
    ).register_type()
    assert (
        field_polynomial_program(
            "GF2PolyAdd", Signature.build_from_dtypes(f_x=packed, g_x=packed), {}
        )
        is None
    )


@pytest.mark.parametrize("field_degree", (M, Integer(3)))
@pytest.mark.parametrize("reverse", (False, True))
def test_standalone_source_symbolic_coefficient_view_declaration(
    field_degree: Any, reverse: bool
) -> None:
    from qualtran._infra.data_types import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import (
        GFPolyJoin,
        GFPolySplit,
    )

    from flagquantum.compiler.registers.finite_field.field_polynomial_component import (
        field_polynomial_component_declaration,
    )

    field = FiniteFieldEncoding(2, field_degree)
    source = (GFPolyJoin if reverse else GFPolySplit)(QGFPoly(2, QGF(2, field_degree)))
    record = field_polynomial_component_declaration(
        2, field.register_type(), reverse=reverse
    )
    assert verify_program_record(record) == ()
    assert record.functions[0].blocks == ()
    assert [
        (r.name, r.bitsize, r.shape, r.side.value)
        for r in record.functions[0].registers or ()
    ] == [(r.name, r.bitsize, r.shape, r.side.value) for r in source.signature]
