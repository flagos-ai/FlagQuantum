"""Original SF constructors retain source graph identities and opaque leaves."""

import sys
from importlib import import_module

import pytest

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.component_graph import (
    _elements,
    iter_component_calls,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._single_factorization_library import (
    single_factorization_factories,
)
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_BASE = "qualtran.bloqs.chemistry.sf."


@pytest.mark.parametrize("block", (False, True))
@pytest.mark.parametrize("variant", (0, 1, 2))
def test_original_root_libraries_and_exact_source_connections(
    block: bool, variant: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module(_BASE + "single_factorization")
    if block:
        source = module.SingleFactorizationBlockEncoding(
            10, 50, 12, 7, 5, kp1=4, kp2=8, kp1_inv=2, kp2_inv=4
        )
    else:
        source = module.SingleFactorizationOneBody(
            50,
            10,
            12,
            7,
            is_adjoint=bool(variant % 2),
            kp1=4,
            kp2=8,
            kp1_inv=2,
            kp2_inv=4,
        )
    library = _library(source, variant)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    expected = _named_inputs(_source_trace(SourceCalls, source.decompose_bloq()))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        program = resolve_component_program(
            library, 0, single_factorization_factories()
        )
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert _named_inputs(_native_trace(program)) == expected
    source_ir = import_module("qualtran")

    graph = source.decompose_bloq()
    expected_returns = {
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
    root = program.functions[0]
    assert root.registers is not None
    body = root.blocks[0]
    ports = [(r.name, index) for r in root.registers for index in r.all_idxs()]
    producers = {
        value: ("entry", port, index)
        for value, (port, index) in zip(body.arguments, ports, strict=True)
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
    } == expected_returns
    assert len(program.functions[0].blocks[0].operations) == (5 if block else 9)
    with pytest.raises(ValueError, match="physical lowering requires component bodies"):
        bind_register_program(program, (), clean=tuple(range(100)))


@pytest.mark.parametrize(
    "kind",
    (
        "InnerPrepareSingleFactorization",
        "OuterPrepareSingleFactorization",
        "SelectSingleFactorization",
    ),
)
@pytest.mark.parametrize("narrow", (False, True))
def test_original_bodyless_leaf_metadata(kind: str, narrow: bool) -> None:
    module = import_module(
        _BASE + ("select_bloq" if kind.startswith("Select") else "prepare")
    )
    spin, aux = (2, 0) if narrow else (10, 50)
    arguments = (
        {"num_spin_orb": spin}
        if kind.startswith("Select")
        else {"num_aux": aux, "num_bits_state_prep": 12}
    )
    if kind.startswith("Inner"):
        arguments.update(num_spin_orb=spin, kp1=4, kp2=8)
    source = getattr(module, kind)(**arguments)
    for depth in (0, 1, 2):
        if narrow and kind == "OuterPrepareSingleFactorization" and depth:
            with pytest.raises(KeyError, match="target"):
                _library(source, depth)
            continue
        library = _library(source, depth)
        program = resolve_component_program(
            library, 0, single_factorization_factories()
        )
        assert (
            program.functions[0].registers == library.entries[0].declaration.registers
        )
        assert not program.functions[0].blocks
        assert verify_program_record(program) == ()


def test_both_select_flags_keep_original_no_system_port() -> None:
    module = import_module(_BASE + "select_bloq")
    programs = []
    for control in (False, True):
        source = module.SelectSingleFactorization(10, additional_control=control)
        program = resolve_component_program(
            _library(source, 0), 0, single_factorization_factories()
        )
        assert tuple(r.name for r in program.functions[0].registers or ()) == (
            "p",
            "q",
            "spin",
            "succ_pq",
            "succ_l",
        )
        programs.append(program)
    assert programs[0] == programs[1]


def test_selected_reflection_dependency_exact_calls_complex_operator_and_cleanup() -> (
    None
):
    import torch

    from flagquantum.compiler.formats.legacy_word_factories import _rename_program
    from flagquantum.compiler.registers.component_definition_match import (
        _named_definitions,
        matches_component_definition,
    )
    from flagquantum.compiler.registers.primitives.identity import (
        register_identity_program,
    )
    from flagquantum.core.ir.program import ProgramRecord
    from tests.unit.test_component_hierarchy_control import compare_layers
    from tests.unit.test_default_linear_combination_oracles import bind
    from tests.unit.test_multi_control_component import atomic_leaves
    from tests.unit.test_multiplexed_source_binding import logical_operator

    module = import_module(_BASE + "single_factorization")
    source = module.SingleFactorizationBlockEncoding(4, 50, 12)
    program = resolve_component_program(
        _library(source, 0), 0, single_factorization_factories()
    )
    name = program.functions[0].blocks[0].operations[2].callee
    assert name is not None
    record = _rename_program(ProgramRecord(_named_definitions(program, name)), "main")
    [original] = [
        instance.bloq
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if type(instance.bloq).__name__ == "ReflectionUsingPrepare"
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
