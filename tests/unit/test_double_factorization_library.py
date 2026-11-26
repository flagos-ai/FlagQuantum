"""Stored DF constructors retain original root, junk and physical child wiring."""

import sys
from importlib import import_module

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_definition_match import (
    _named_definitions,
    matches_component_definition,
)
from flagquantum.compiler.registers.component_graph import (
    _elements,
    iter_component_calls,
)
from flagquantum.compiler.registers.primitives.identity import register_identity_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._double_factorization_library import (
    double_factorization_factories,
)
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_BASE = "qualtran.bloqs.chemistry.df."


@pytest.mark.parametrize("block", (False, True))
@pytest.mark.parametrize("example", (False, True))
def test_source_libraries_keep_all_calls_and_boundary_outputs(
    block: bool, example: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module(_BASE + "double_factorization")
    source = (
        getattr(module, "_df_block_encoding" if block else "_df_one_body").make()
        if example
        else (
            module.DoubleFactorizationBlockEncoding(6, 50, 250, 12, 8, 3, 7)
            if block
            else module.DoubleFactorizationOneBody(50, 6, 250, 12, 7, 8)
        )
    )
    graph = source.decompose_bloq()
    expected = _named_inputs(_source_trace(SourceCalls, graph))
    source_ir = import_module("qualtran")
    returned = {
        (edge.right.reg.name, edge.right.idx): (
            (
                "entry"
                if edge.left.binst == source_ir.LeftDangle
                else str(edge.left.binst.i)
            ),
            edge.left.reg.name,
            edge.left.idx,
        )
        for edge in graph.connections
        if edge.right.binst == source_ir.RightDangle
    }
    baseline = None
    for depth in (0, 1, 2):
        library = _library(source, depth)
        encoded = serialize_component_library(library)
        assert (
            serialize_component_library(deserialize_component_library(encoded))
            == encoded
        )
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            program = resolve_component_program(
                library, 0, double_factorization_factories()
            )
        assert verify_program_record(program) == ()
        root = program.functions[0]
        assert root.registers == library.entries[0].declaration.registers
        assert _named_inputs(_native_trace(program)) == expected
        body = root.blocks[0]
        assert len(body.operations) == (7 if block else 9)
        assert root.registers is not None
        ports = [(r.name, index) for r in root.registers for index in r.all_idxs()]
        producers = {
            v: ("entry", port, index)
            for v, (port, index) in zip(body.arguments, ports, strict=True)
        }
        for position, _, outputs in iter_component_calls(program):
            for binding in _elements(outputs):
                producers[binding.value] = (
                    str(position),
                    binding.register.name,
                    binding.index,
                )
        assert body.return_values is not None
        assert {
            port: producers[value]
            for port, value in zip(ports, body.return_values, strict=True)
        } == returned
        with pytest.raises(
            ValueError, match="physical lowering requires component bodies"
        ):
            bind_register_program(program, (), clean=tuple(range(100)))
        if baseline is None:
            baseline = program
        else:
            assert program == baseline


@pytest.mark.parametrize(
    "kind",
    (
        "InnerPrepareDoubleFactorization",
        "OuterPrepareDoubleFactorization",
        "OutputIndexedData",
        "ProgRotGateArray",
    ),
)
def test_original_bodyless_leaf_examples(kind: str) -> None:
    module = import_module(
        _BASE + ("select_bloq" if kind == "ProgRotGateArray" else "prepare")
    )
    if kind == "ProgRotGateArray":
        source = module.ProgRotGateArray(50, 10, 250, 1)
    else:
        source = getattr(
            module,
            {
                "InnerPrepareDoubleFactorization": "_prep_inner",
                "OuterPrepareDoubleFactorization": "_prep_outer",
                "OutputIndexedData": "_indexed_data",
            }[kind],
        ).make()
    for depth in (0, 1, 2):
        library = _library(source, depth)
        program = resolve_component_program(
            library, 0, double_factorization_factories()
        )
        assert (
            program.functions[0].registers == library.entries[0].declaration.registers
        )
        assert not program.functions[0].blocks
        assert verify_program_record(program) == ()


@pytest.mark.parametrize("reflection", (False, True))
def test_selected_physical_children_calls_all_complex_histories_and_work(
    reflection: bool,
) -> None:
    from tests.unit.test_component_hierarchy_control import compare_layers
    from tests.unit.test_default_linear_combination_oracles import bind
    from tests.unit.test_multi_control_component import atomic_leaves
    from tests.unit.test_multiplexed_source_binding import logical_operator

    module = import_module(_BASE + "double_factorization")
    source = (
        module.DoubleFactorizationBlockEncoding(4, 50, 250, 12, 8)
        if reflection
        else module.DoubleFactorizationOneBody(50, 4, 250, 12, 8, 8)
    )
    program = resolve_component_program(
        _library(source, 0), 0, double_factorization_factories()
    )
    name = program.functions[0].blocks[0].operations[3 if reflection else 4].callee
    assert name is not None
    record = _rename_program(ProgramRecord(_named_definitions(program, name)), "main")
    expected_kind = "ReflectionUsingPrepare" if reflection else "ApplyControlledZs"
    [original] = [
        i.bloq
        for i, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if type(i.bloq).__name__ == expected_kind
    ]
    identities = {
        f.name
        for f in record.functions
        if matches_component_definition(record, f.name, register_identity_program(1))
    }
    compare_layers(record, original, atomic_leaves(record) | identities)
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.tensor(original.tensor_contract(), dtype=torch.complex128),
        atol=8e-12,
        rtol=0,
    )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live
