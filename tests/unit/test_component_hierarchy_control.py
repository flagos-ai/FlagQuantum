"""Stored hierarchy controls match the original protocol at every call boundary."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import (
    control_component_hierarchy,
    control_component_program,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control import add_h, add_phase, body, fixture_program
from tests.unit.test_component_graph import _native_trace, _source, _source_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def hierarchy(
    depth: int, repeats: int, spectator: str = "spectator"
) -> tuple[ProgramRecord, Any]:
    from qualtran import Controlled, CtrlSpec

    base, source = fixture_program(1, spectator)
    if depth == 1:
        return base, source
    function = base.functions[0]
    signature = Signature(function.registers or ())
    previous = "layer0"
    definitions = (replace(function, name=previous), *base.functions[1:])
    source = source.subbloq
    for level in range(1, depth):
        name = "main" if level == depth - 1 else f"layer{level}"
        native = ComponentProgramBuilder(signature, definitions, name=name)
        upstream = _source(signature)
        values, expected = native.inputs, upstream.inputs
        for _ in range(repeats):
            values = native.call_dict(previous, values)
            expected = upstream.builder.add_d(source, **expected)
        record = native.finalize(values)
        definitions, previous, source = (
            record.functions,
            name,
            upstream.builder.finalize(**expected),
        )
    return ProgramRecord(definitions), Controlled(source, CtrlSpec())


def compare_layers(
    record: ProgramRecord, source: Any, terminals: set[str]
) -> dict[str, Any]:
    functions = {f.name: f for f in record.functions}
    pairs = {"main": source}
    pending = ["main"]
    observer = _source(Signature([]))
    while pending:
        name = pending.pop()
        if name in terminals:
            continue
        composite = pairs[name].decompose_bloq()
        assert tuple(
            observer.signature(Signature(functions[name].registers or ()))
        ) == tuple(composite.signature)
        actual = _native_trace(record, name)
        expected = _source_trace(observer, composite)
        # Generic source adders insert the control key last in the input dict.
        # Ordered signatures above and named operands here preserve port order.
        for actual_row, expected_row in zip(actual, expected, strict=True):
            assert isinstance(actual_row, tuple) and isinstance(expected_row, tuple)
            assert actual_row[0] == expected_row[0]
            assert dict(actual_row[1]) == dict(expected_row[1])
            assert actual_row[2] == expected_row[2]
        operations = functions[name].blocks[0].operations
        for instance, _, _ in composite.iter_bloqsoqs():
            target = operations[instance.i].callee
            assert target is not None
            if target in pairs:
                assert pairs[target] == instance.bloq
            else:
                pairs[target] = instance.bloq
                pending.append(target)
    return pairs


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("spectator", ["spectator", "ctrl"])
def test_shared_hierarchy_source_graph_gates_counts_and_complex_measurement_branches(
    depth: int, repeats: int, spectator: str
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, source = hierarchy(depth, repeats, spectator)
    result = control_component_hierarchy(record, {"h": add_h, "p": add_phase})
    pairs = compare_layers(result, source, {"ch", "cp"})
    assert len(result.functions) == len(record.functions) + depth - 1
    assert all(
        a is b
        for a, b in zip(
            record.functions[1:],
            result.functions[1 : len(record.functions)],
            strict=True,
        )
    )
    assert verify_program_record(result) == ()
    count = repeats ** (depth - 1)
    calls = program_component_call_graph(result, keep={"ch", "cp"}.__contains__)
    assert calls.leaf_counts == {"ch": count, "cp": count}
    names = {bloq: name for name, bloq in pairs.items()}
    graph, totals = source.call_graph(keep=lambda b: b in {pairs["ch"], pairs["cp"]})
    assert calls.direct_calls == {
        (names[parent], names[child]): data["n"]
        for parent, child, data in graph.edges.data()
    }
    assert calls.leaf_counts == {names[bloq]: n for bloq, n in totals.items()}
    source_model = get_cost_value(source, QECGatesCost())
    assert (
        source_model.measurement == count and get_cost_value(source, QubitCount()) == 4
    )
    models = {
        name: get_cost_value(pairs[name], QECGatesCost()) for name in ("ch", "cp")
    }
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):

        def cost(name: str, field: str = category) -> int:
            return int(getattr(models[name], field))

        assert calls.estimate_leaf_cost(cost) == getattr(source_model, category)
    bound = bind_register_program(result, ((0,), (1,), (2,)), clean=(3,))
    flat, _ = expand_component_calls(result)
    assert (
        bind_register_program(flat, ((0,), (1,), (2,)), clean=(3,)).circuit
        == bound.circuit
    )
    direct, _ = fixture_program(count, spectator)
    expected = control_component_program(direct, {"h": add_h, "p": add_phase})
    assert (
        bind_register_program(expected, ((0,), (1,), (2,)), clean=(3,)).circuit
        == bound.circuit
    )
    assert Counter(g.name for g in bound.circuit.instructions)["measure"] == count
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = torch.tensor([complex(2 * i - 3, i + 1) for i in range(8)])
    initial /= initial.norm()
    expected_state = torch.zeros_like(initial)
    expected_state[::2] = torch.as_tensor(
        np.einsum(
            "ij,j->i", source.tensor_contract(), initial[::2].numpy(), optimize=False
        )
    )
    # Exhaust small histories and explicitly sample larger graphs, whose
    # complete primitive stream is independently checked above.
    histories = (
        product((0, 1), repeat=count)
        if count <= 4
        else ((0,) * count, (1,) * count, tuple(i % 2 for i in range(count)))
    )
    for history in histories:
        state = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            state, expected_state / 2 ** (count / 2), rtol=0, atol=2e-12
        )
        assert torch.count_nonzero(state[1::2]).item() == 0


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("expand", [False, True])
def test_exact_hierarchical_adjoints_restore_every_forward_inverse_history(
    depth: int, expand: bool
) -> None:
    record, source = hierarchy(depth, 2 if depth == 2 else 1)
    result = control_component_hierarchy(record, {"h": add_h, "p": add_phase})
    inverse_body = body(
        "cp_inverse",
        (Register("q", quantum_bit_type(), (2,)),),
        CircuitIR(3, controlled_z_power_ir(0, 1, -0.25, 2, 0)),
    )
    inverse = adjoint_component_hierarchy(
        replace(result, functions=(*result.functions, *inverse_body)),
        {"ch": "ch", "cp": "cp_inverse"},
    )
    if expand:
        result, _ = expand_component_calls(result)
        inverse, _ = expand_component_calls(inverse)
    gates = bind_register_program(
        result, ((0,), (1,), (2,)), clean=(3,)
    ).circuit.instructions
    backwards = bind_register_program(
        inverse, ((0,), (1,), (2,)), clean=(3,)
    ).circuit.instructions
    count = sum(g.name == "measure" for g in gates)
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = torch.tensor([complex(i + 1, 2 * i - 3) for i in range(8)])
    initial /= initial.norm()
    for first in product((0, 1), repeat=count):
        state = forced_branch(initial, gates, first)
        for second in product((0, 1), repeat=count):
            restored = forced_branch(state, backwards, second)
            torch.testing.assert_close(restored, initial / 2**count, rtol=0, atol=2e-12)
            assert torch.count_nonzero(restored[1::2]).item() == 0
    assert source.ctrl_reg_names == ("ctrl",)


def test_explicit_adder_overrides_derivation_and_preserves_source_call_boundary() -> (
    None
):
    record, source = hierarchy(3, 2)
    base, _ = fixture_program(1)
    special = control_component_program(base, {"h": add_h, "p": add_phase}).functions[0]
    special = replace(special, name="special")
    record = replace(record, functions=(*record.functions, special))

    def add_special(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        outputs = builder.call_dict("special", {"ctrl": control, **values})
        return outputs.pop("ctrl"), tuple(outputs.values())

    result = control_component_hierarchy(record, {"layer0": add_special})
    compare_layers(result, source, {"special"})
    assert len(result.functions) == len(record.functions) + 1
    assert program_component_call_graph(
        result, keep={"special"}.__contains__
    ).leaf_counts == {"special": 4}
    automatic = control_component_hierarchy(record, {"h": add_h, "p": add_phase})
    assert (
        bind_register_program(result, ((0,), (1,), (2,)), clean=(3,)).circuit
        == bind_register_program(automatic, ((0,), (1,), (2,)), clean=(3,)).circuit
    )


def test_generated_names_avoid_existing_definitions() -> None:
    record, source = hierarchy(3, 2)
    empty = ComponentProgramBuilder(Signature([]), name="__fq_controlled_0")
    record = replace(
        record, functions=(*record.functions, *empty.finalize({}).functions)
    )
    result = control_component_hierarchy(record, {"h": add_h, "p": add_phase})
    assert [f.name for f in result.functions[len(record.functions) :]] == [
        "__fq_controlled_1",
        "__fq_controlled_2",
    ]
    compare_layers(result, source, {"ch", "cp"})


@pytest.mark.parametrize(
    "failure",
    [
        "root_mapping",
        "missing_atom",
        "invalid_unused",
        "unknown",
        "callback",
        "directional",
    ],
)
def test_hierarchy_rejects_invalid_modules_and_missing_specializations_without_mutation(
    failure: str,
) -> None:
    record, _ = hierarchy(2, 1)
    adders = {"h": add_h, "p": add_phase}
    entry = "absent" if failure == "unknown" else "main"
    if failure == "root_mapping":
        adders["main"] = add_h
    elif failure == "missing_atom":
        adders.pop("h")
    elif failure == "invalid_unused":
        record = replace(record, functions=(*record.functions, record.functions[-1]))
    elif failure == "callback":

        def broken(
            builder: ComponentProgramBuilder,
            control: BindingCollection,
            values: dict[str, BindingCollection],
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            raise RuntimeError("controlled child failed")

        adders["h"] = broken
    elif failure == "directional":
        from flagquantum.compiler.registers.layout.component_workspace import (
            allocate_word_function,
        )

        alloc = allocate_word_function("alloc", quantum_bit_type(), False)
        native = ComponentProgramBuilder(
            Signature([Register("q", quantum_bit_type(), side=RegisterSide.RIGHT)]),
            [alloc],
        )
        record = native.finalize({"q": native.call_dict("alloc", {})["reg"]})
        adders["alloc"] = add_h
    before = repr(record)
    with pytest.raises(RuntimeError if failure == "callback" else ValueError):
        control_component_hierarchy(record, adders, entry)
    assert repr(record) == before


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_empty_shared_children_keep_control_and_shaped_ports(
    width: int, shape: tuple[int, ...]
) -> None:
    from qualtran import Controlled, CtrlSpec

    signature = Signature([Register("words", RegisterType("uint", width), shape)])
    empty = ComponentProgramBuilder(signature, name="empty")
    source = _source(signature)
    child = source.builder.finalize(**source.inputs)
    native = ComponentProgramBuilder(signature, empty.finalize(empty.inputs).functions)
    upstream = _source(signature)
    values, expected = native.inputs, upstream.inputs
    for _ in range(2):
        values = native.call_dict("empty", values)
        expected = upstream.builder.add_d(child, **expected)
    record = native.finalize(values)
    source_control = Controlled(upstream.builder.finalize(**expected), CtrlSpec())
    result = control_component_hierarchy(record, {})
    if 0 in shape:
        with pytest.raises(ValueError):
            compare_layers(result, source_control, set())
    else:
        compare_layers(result, source_control, set())
    assert len(result.functions) == 3
    count = int(np.prod(shape))
    inputs = (
        (0,),
        *(tuple(range(1 + i * width, 1 + (i + 1) * width)) for i in range(count)),
    )
    assert bind_register_program(result, inputs).circuit.instructions == ()


@pytest.mark.parametrize("hierarchical", [False, True])
def test_adders_retain_new_split_join_and_lifetime_helper_definitions(
    hierarchical: bool,
) -> None:
    from flagquantum.core.ir.program.register_bindings import RegisterBinding

    record, _ = hierarchy(2 if hierarchical else 1, 1)

    def rebind_h(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        q = values["q"]
        assert isinstance(q, RegisterBinding)
        bits = builder.split(q)
        q = builder.join(bits, dtype=q.register.dtype)
        work = builder.allocate(dtype=quantum_bit_type())
        builder.free(work)
        return add_h(builder, control, {"q": q})

    transform = (
        control_component_hierarchy if hierarchical else control_component_program
    )
    result = transform(record, {"h": rebind_h, "p": add_phase})
    expected = transform(record, {"h": add_h, "p": add_phase})
    assert len(result.functions) == len(expected.functions) + 4
    assert verify_program_record(result) == ()
    assert (
        bind_register_program(result, ((0,), (1,), (2,)), clean=(3,)).circuit
        == bind_register_program(expected, ((0,), (1,), (2,)), clean=(3,)).circuit
    )


def test_shared_child_work_and_rebinding_helpers_are_generated_once() -> None:
    from flagquantum.core.ir.program.register_bindings import RegisterBinding

    record, _ = hierarchy(3, 2)
    seen = []

    def rebind_h(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        seen.append("h")
        q = values["q"]
        assert isinstance(q, RegisterBinding)
        q = builder.join(builder.split(q), dtype=q.register.dtype)
        return add_h(builder, control, {"q": q})

    result = control_component_hierarchy(record, {"h": rebind_h, "p": add_phase})
    assert seen == ["h"]
    assert len(result.functions) == len(record.functions) + 4
    assert (
        Counter(
            g.name
            for g in bind_register_program(
                result, ((0,), (1,), (2,)), clean=(3,)
            ).circuit.instructions
        )["ch"]
        == 4
    )


def test_nondefault_entry_preserves_block_names_and_clears_payload() -> None:
    record, _ = hierarchy(2, 1)
    function = record.functions[0]
    block = function.blocks[0]
    values = {
        v: replace(v, scope="body")
        for v in (*block.arguments, *(v for op in block.operations for v in op.results))
    }
    renamed = replace(
        block,
        name="body",
        arguments=tuple(values[v] for v in block.arguments),
        operations=tuple(
            replace(
                op,
                operands=tuple(values[v] for v in op.operands),
                results=tuple(values[v] for v in op.results),
            )
            for op in block.operations
        ),
        return_values=tuple(values[v] for v in block.return_values or ()),
    )
    record = replace(
        record,
        canonical_payload={"forward": "only"},
        functions=(
            replace(function, name="run", entry="body", blocks=(renamed,)),
            *record.functions[1:],
        ),
    )
    result = control_component_hierarchy(record, {"h": add_h, "p": add_phase}, "run")
    assert result.canonical_payload is None and record.canonical_payload == {
        "forward": "only"
    }
    assert result.functions[0].entry == result.functions[0].blocks[0].name == "body"
    assert verify_program_record(result) == ()
    assert bind_register_program(
        result, ((0,), (1,), (2,)), clean=(3,), entry="run"
    ).outputs == ((0,), (1,), (2,))


def test_root_cannot_invalidate_a_retained_caller() -> None:
    record, _ = hierarchy(2, 1)
    root = record.functions[0]
    native = ComponentProgramBuilder(
        Signature(root.registers or ()), record.functions, name="caller"
    )
    record = native.finalize(native.call_dict("main", native.inputs))
    with pytest.raises(ValueError, match="invalid constructed register Program"):
        control_component_hierarchy(record, {"h": add_h, "p": add_phase})


def test_diamond_and_direct_calls_share_one_controlled_child_across_depths() -> None:
    from qualtran import Controlled, CtrlSpec

    base, source = fixture_program(1)
    signature = Signature(base.functions[0].registers or ())
    definitions = (replace(base.functions[0], name="shared"), *base.functions[1:])
    sources = {"shared": source.subbloq}
    for name, children in (
        ("left", ("shared",)),
        ("right", ("shared", "shared")),
        ("main", ("left", "right", "shared")),
    ):
        builder = ComponentProgramBuilder(signature, definitions, name=name)
        upstream = _source(signature)
        values, expected = builder.inputs, upstream.inputs
        for child in children:
            values = builder.call_dict(child, values)
            expected = upstream.builder.add_d(sources[child], **expected)
        module = builder.finalize(values)
        definitions = module.functions
        sources[name] = upstream.builder.finalize(**expected)
    record = ProgramRecord(definitions)
    result = control_component_hierarchy(record, {"h": add_h, "p": add_phase})
    compare_layers(result, Controlled(sources["main"], CtrlSpec()), {"ch", "cp"})
    assert len(result.functions) == len(record.functions) + 3
    assert program_component_call_graph(
        result, keep={"ch", "cp"}.__contains__
    ).leaf_counts == {"ch": 4, "cp": 4}
    direct, _ = fixture_program(4)
    expected_record = control_component_program(direct, {"h": add_h, "p": add_phase})
    assert (
        bind_register_program(result, ((0,), (1,), (2,)), clean=(3,)).circuit
        == bind_register_program(
            expected_record, ((0,), (1,), (2,)), clean=(3,)
        ).circuit
    )
