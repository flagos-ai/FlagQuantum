"""Complete original set-oracle factories preserve bytes and route owned calls."""

from collections.abc import Mapping
from dataclasses import replace

import pytest

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.symmetric_difference_component import (
    symmetric_difference_program,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from tests.unit.test_symmetric_difference import layout

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name", ("_symm_diff", "_symm_diff_symb", "_symm_diff_equal_size_symb")
)
def test_all_actual_source_factories_library_and_native_resolution(name: str) -> None:
    import sympy
    from qualtran.bloqs.arithmetic.lists import symmetric_difference as src
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = getattr(src, name).make()
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert len(library.entries) == (1 if depth == 0 else 11)
        restored = bloqs_from_proto(proto)[0]
        assert (source == restored) == (name == "_symm_diff")
        for field in ("n_lhs", "n_rhs", "n_diff"):
            a, b = getattr(source, field), getattr(restored, field)
            assert str(a) == str(b)
            if isinstance(a, sympy.Basic):
                assert all(s.is_positive and s.is_integer for s in a.free_symbols)
                assert all(
                    s.assumptions0 == {"commutative": True} for s in b.free_symbols
                )
        if name != "_symm_diff":
            a, b = source.dtype.num_qubits, restored.dtype.num_qubits
            assert str(a) == str(b)
            assert all(s.is_positive and s.is_integer for s in a.free_symbols)
            assert all(s.assumptions0 == {"commutative": True} for s in b.free_symbols)
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )
    declaration = library.entries[0].declaration

    def factory(
        function_name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None
        for field in ("n_lhs", "n_rhs", "n_diff"):
            assert str(arguments[field]) == str(getattr(source, field))
        assert arguments["dtype"] == registers[0].dtype
        if name != "_symm_diff":
            return FunctionRecord(function_name, (), registers=registers)
        graph = symmetric_difference_program(4, 2, 4, 4)
        assert graph.functions[0].registers == registers
        graph = replace(
            graph,
            functions=(
                replace(graph.functions[0], name=function_name),
                *graph.functions[1:],
            ),
        )
        return _namespace_helpers(graph, function_name)

    resolved = resolve_component_program(library, 0, {declaration.name: factory})
    if name != "_symm_diff":
        assert resolved.functions[0].registers == declaration.registers
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(resolved, ())
        return
    a, b, out, flag, clean, _ = layout(4, 2, 4, 4)
    ports = (*a, *b, *out, (flag,))
    actual = bind_register_program(resolved, ports, clean=clean)
    expected = bind_register_program(
        symmetric_difference_program(4, 2, 4, 4), ports, clean=clean
    )
    assert actual.circuit == expected.circuit and actual.outputs == expected.outputs


@pytest.mark.parametrize("kind", ("unsigned", "signed", "opaque"))
@pytest.mark.parametrize("shape", ((1, 1, 0, 1), (2, 1, 1, 2), (2, 2, 2, 2)))
def test_finite_dtype_metadata_keeps_explicit_execution_factory(
    kind: str, shape: tuple[int, int, int, int]
) -> None:
    from qualtran import QAny, QInt, QUInt
    from qualtran.bloqs.arithmetic.lists.symmetric_difference import SymmetricDifference
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    lhs, rhs, size, width = shape
    source = SymmetricDifference(
        lhs, rhs, size, {"unsigned": QUInt, "signed": QInt, "opaque": QAny}[kind](width)
    )
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        assert serialize_component_library(deserialize_component_library(wire)) == wire
        assert source == bloqs_from_proto(proto)[0]
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )
    declaration = library.entries[0].declaration

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None and arguments["dtype"] == registers[0].dtype
        if kind != "unsigned":
            return FunctionRecord(name, (), registers=registers)
        graph = symmetric_difference_program(lhs, rhs, width, size)
        assert graph.functions[0].registers == registers
        graph = replace(
            graph,
            functions=(replace(graph.functions[0], name=name), *graph.functions[1:]),
        )
        return _namespace_helpers(graph, name)

    resolved = resolve_component_program(library, 0, {declaration.name: factory})
    assert resolved.functions[0].registers == declaration.registers
    if kind == "unsigned":
        a, b, out, flag, clean, _ = layout(lhs, rhs, width, size)
        ports = (*a, *b, *out, (flag,))
        actual = bind_register_program(resolved, ports, clean=clean)
        expected = bind_register_program(
            symmetric_difference_program(lhs, rhs, width, size), ports, clean=clean
        )
        assert actual.circuit == expected.circuit and actual.outputs == expected.outputs
