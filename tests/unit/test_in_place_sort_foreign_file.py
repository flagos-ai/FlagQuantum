"""Original SortInPlace libraries resolve typed integer execution explicitly."""

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
from flagquantum.compiler.registers.sorting.in_place_sort_component import (
    in_place_sort_program,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from tests.unit.test_in_place_sort import layout

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
sympy = pytest.importorskip("sympy")


@pytest.mark.parametrize("kind", ("unsigned", "signed", "opaque"))
@pytest.mark.parametrize(
    "mode",
    (
        "one",
        "three",
        "word_symbol",
        "length_symbol",
        "plain_symbols",
        "assumed_symbols",
    ),
)
def test_entire_source_library_wire_declaration_and_native_execution(
    kind: str, mode: str
) -> None:
    from qualtran import QAny, QInt, QUInt
    from qualtran.bloqs.arithmetic.lists.sort_in_place import SortInPlace
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    assumed = sympy.symbols("l n", positive=True, integer=True)
    plain = sympy.symbols("l n")
    length, width = {
        "one": (1, 2),
        "three": (3, 2),
        "word_symbol": (3, assumed[1]),
        "length_symbol": (assumed[0], 2),
        "plain_symbols": plain,
        "assumed_symbols": assumed,
    }[mode]
    source = SortInPlace(
        length, {"unsigned": QUInt, "signed": QInt, "opaque": QAny}[kind](width)
    )
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        reconstructed = bloqs_from_proto(proto)[0]
        assert str(reconstructed) == str(source)
        # Original textual serialization drops symbol assumptions itself.
        assert (reconstructed == source) == (mode in {"one", "three", "plain_symbols"})
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )
    declaration = library.entries[0].declaration
    executable = kind in {"unsigned", "signed"} and mode in {"one", "three"}

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None
        assert str(arguments["l"]) == str(source.l)
        assert arguments["dtype"] == registers[0].dtype
        assert all(r.side.name == "THRU" for r in registers)
        if not executable:
            return FunctionRecord(name, (), registers=registers)
        graph = in_place_sort_program(int(length), int(width), signed=kind == "signed")
        assert graph.functions[0].registers == registers
        graph = replace(
            graph,
            functions=(replace(graph.functions[0], name=name), *graph.functions[1:]),
        )
        return _namespace_helpers(graph, name)

    resolved = resolve_component_program(library, 0, {declaration.name: factory})
    if mode in {"length_symbol", "plain_symbols", "assumed_symbols"}:
        assert resolved.functions[0].registers == declaration.registers
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(resolved, ())
        return
    assert resolved.functions[0].registers == declaration.registers
    if executable:
        words, ranks, clean, _ = layout(int(length), int(width))
        actual = bind_register_program(resolved, (*words, *ranks), clean=clean)
        expected = bind_register_program(
            in_place_sort_program(int(length), int(width), signed=kind == "signed"),
            (*words, *ranks),
            clean=clean,
        )
        assert actual.circuit == expected.circuit and actual.outputs == expected.outputs
