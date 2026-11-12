"""Classical edge labels use producer SSA values from the locked source chain."""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.multi_control import multi_and_stages
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.classical_component_graph import (
    export_classical_component_graph_dot,
)
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_graph import register_connections
from flagquantum.simulation.classical_and import and_classical_values
from flagquantum.simulation.component_classical_state import ComponentClassicalState

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def multi_fixture(
    values: tuple[int, ...],
) -> tuple[ProgramRecord, dict[str, tuple[int, int]]]:
    bit = RegisterType("bit", 1)
    patterns = dict.fromkeys(cvs for _, _, _, cvs in multi_and_stages(values))
    names = {"and" + "".join(map(str, cvs)): cvs for cvs in patterns}
    functions = tuple(
        f
        for name, cvs in names.items()
        for f in control_predicate_program(
            (Register("ctrl", bit, (2,)),), (ControlRegister(1, cvs),), name=name
        ).functions
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("ctrl", bit, (4,)),
                Register("junk", bit, (2,), RegisterSide.RIGHT),
                Register("target", bit, side=RegisterSide.RIGHT),
            )
        ),
        functions,
    )
    wires = dict(enumerate(np.asarray(builder.inputs["ctrl"])))
    for left, right, target, cvs in multi_and_stages(values):
        result = builder.call_dict(
            "and" + "".join(map(str, cvs)),
            {"ctrl": np.array([wires[left], wires[right]], dtype=object)},
        )
        ctrl = np.asarray(result["ctrl"])
        wires[left], wires[right], wires[target] = ctrl[0], ctrl[1], result["target"]
    return (
        builder.finalize(
            {
                "ctrl": np.array([wires[i] for i in range(4)], dtype=object),
                "junk": np.array([wires[4], wires[5]], dtype=object),
                "target": wires[6],
            }
        ),
        names,
    )


def state_for(
    record: ProgramRecord,
    inputs: Mapping[str, object],
    action: Callable[[FunctionRecord, dict[str, object]], object],
) -> ComponentClassicalState:
    state = ComponentClassicalState(
        record, tuple(i for i, _, _ in iter_component_calls(record)), inputs, action
    )
    state.simulate()
    return state


def parsed(dot: str) -> Any:
    import pydot

    graphs = pydot.graph_from_dot_data(dot)
    assert graphs is not None and len(graphs) == 1
    return graphs[0]


def labels(dot: str) -> list[str]:
    return [json.loads(e.get_label()) for e in parsed(dot).get_edges()]


@pytest.mark.parametrize("cvs", [(1, 1, 1, 1), (1, 1, 0, 1)])
def test_full_multi_and_all_inputs_edges_styles_and_original_fixture(
    cvs: tuple[int, ...],
) -> None:
    from qualtran import BloqInstance
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.drawing.classical_sim_graph import ClassicalSimGraphDrawer
    from qualtran.drawing.musical_score import _cbloq_musical_score

    record, patterns = multi_fixture(cvs)
    score = component_register_score(record)
    source = MultiAnd(cvs).decompose_bloq()
    _, positions, _ = _cbloq_musical_score(source.signature, source._binst_graph)
    lanes = {line.lane: line.register for line in score.lines}

    def action(function: FunctionRecord, values: dict[str, object]) -> object:
        controls = np.asarray(values["ctrl"])
        return and_classical_values(controls, patterns[function.name])

    for bits in product((0, 1), repeat=4):
        inputs = {"ctrl": np.array(bits)}
        state = state_for(record, inputs, action)
        actual = export_classical_component_graph_dot(record, state.assignments)
        reference = ClassicalSimGraphDrawer(source, inputs)
        expected = Counter(
            (
                lanes[positions[c.left].y],
                c.left.binst.i if isinstance(c.left.binst, BloqInstance) else None,
                c.right.binst.i if isinstance(c.right.binst, BloqInstance) else None,
                reference.cxn_label(c),
            )
            for c in source.connections
        )
        observed = Counter(
            (e.register, e.producer, e.consumer, label)
            for e, label in zip(
                register_connections(score), labels(actual), strict=True
            )
        )
        assert observed == expected
        graph = parsed(actual)
        assert len(graph.get_edges()) == 13
        for edge in graph.get_edges():
            attrs = edge.get_attributes()
            source_edge = reference.cxn_edge("left", "right", source.connections[0])
            expected_attrs = source_edge.get_attributes()

            def normalize(value: object) -> str:
                text = str(value)
                return text.strip('"').lower()

            assert {k: normalize(v) for k, v in attrs.items() if k != "label"} == {
                k: normalize(v) for k, v in expected_attrs.items() if k != "label"
            }
        assert record.functions[0].name == "main"
        assert inputs["ctrl"].tolist() == list(bits)


def twice_x() -> ProgramRecord:
    bit = RegisterType("bit", 1)
    signature = Signature((Register("q", bit),))
    leaf = bit_gate_function(
        "x", tuple(signature), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    current = builder.inputs
    for _ in range(2):
        current = builder.call_dict("x", current)
    return builder.finalize(current)


def test_through_edges_use_producer_not_overwritten_consumer_and_no_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.drawing.classical_sim_graph import ClassicalSimGraphDrawer

    from flagquantum.simulation.classical_pauli_values import x_classical_value

    record = twice_x()

    def action(_: FunctionRecord, values: dict[str, object]) -> object:
        q = values["q"]
        assert isinstance(q, (int, np.integer))
        return {"q": x_classical_value(q)}

    state = state_for(record, {"q": 0}, action)
    source = BloqBuilder()
    q = source.add_register("q", 1)
    for _ in range(2):
        q = source.add(XGate(), q=q)
    reference = ClassicalSimGraphDrawer(source.finalize(q=q), {"q": 0})
    assert Counter(
        reference.cxn_label(c) for c in reference._cbloq.connections
    ) == Counter(["0", "1", "0"])
    snapshot = dict(state.assignments)
    original = hash(record)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("drawing must not evaluate")

    monkeypatch.setattr(ComponentClassicalState, "simulate", forbidden)
    assert labels(export_classical_component_graph_dot(record, state.assignments)) == [
        "0",
        "1",
        "0",
    ]
    assert state.assignments == snapshot and hash(record) == original


def test_missing_assignments_and_failed_stringification_propagate() -> None:
    record = twice_x()
    with pytest.raises(KeyError):
        export_classical_component_graph_dot(record, {})

    class BadValue:
        def __str__(self) -> str:
            raise RuntimeError("value formatting failed")

    values = {v: BadValue() for v in (record.functions[0].blocks[0].arguments)}
    with pytest.raises(RuntimeError, match="value formatting failed"):
        export_classical_component_graph_dot(record, values)


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_idle_scalar_array_empty_and_escaped_labels(shape: tuple[int, ...]) -> None:
    builder = ComponentProgramBuilder(
        Signature((Register("x<&>", RegisterType("uint", 2), shape),))
    )
    record = builder.finalize(builder.inputs)
    block = record.functions[0].blocks[0]
    text = 'quote" newline\n backslash\\'
    assignments = dict.fromkeys(block.arguments, text)
    dot = export_classical_component_graph_dot(record, assignments)
    assert labels(dot) == [text] * len(block.arguments)
    assert ("&lt;&amp;&gt;" in dot) == bool(block.arguments)


def test_unknown_entry_and_unexecuted_assignment_failure() -> None:
    record = twice_x()
    with pytest.raises(ValueError, match="unknown"):
        export_classical_component_graph_dot(record, {}, "absent")
    state = ComponentClassicalState(record, (0, 1), {"q": 0}, lambda _, v: v)
    with pytest.raises(KeyError):
        export_classical_component_graph_dot(record, state.assignments)


def test_closed_state_child_effect_lifetimes_and_real_render(tmp_path: Path) -> None:
    from xml.etree import ElementTree

    from flagquantum.compiler.registers.component_basis_vectors import (
        integer_basis_function_data,
    )
    from flagquantum.compiler.registers.component_classical_vectors import (
        flank_component_with_classical_vectors,
    )
    from flagquantum.drawer.call_graph import render_call_graph_svg

    record = flank_component_with_classical_vectors(twice_x(), {"q": 0}, {"q": 0})

    def action(function: FunctionRecord, values: dict[str, object]) -> object:
        basis = integer_basis_function_data(function)
        if basis is not None:
            return {"val": basis[0]} if basis[1] else {}
        assert function.name == "main"
        return values

    state = ComponentClassicalState(
        record, (0, 1, 2), {}, action, entry="classical_vectors"
    )
    assert state.simulate() == {}
    dot = export_classical_component_graph_dot(
        record, state.assignments, "classical_vectors"
    )
    assert labels(dot) == ["0", "0"]
    svg = render_call_graph_svg(dot)
    root = ElementTree.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert len(root.findall(".//svg:g[@class='edge']", ns)) == 2
    (tmp_path / "classical.svg").write_bytes(svg)


def test_original_single_test_assertions_with_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from types import SimpleNamespace

    original = importlib.import_module("qualtran.drawing.classical_sim_graph_test")
    record, patterns = multi_fixture((1, 1, 1, 1))

    def drawer(bloq: Any, vals: dict[str, object]) -> Any:
        assert len(bloq.bloq_instances) == 3
        assert np.asarray(vals["ctrl"]).tolist() == [1, 1, 0, 1]

        def action(function: FunctionRecord, values: dict[str, object]) -> object:
            return and_classical_values(
                np.asarray(values["ctrl"]), patterns[function.name]
            )

        state = state_for(record, vals, action)
        graph = parsed(export_classical_component_graph_dot(record, state.assignments))
        # The original regex expects unquoted numeric labels. Pydot serializes
        # those equivalently; normalize only DOT quoting at this test boundary.
        for edge in graph.get_edges():
            edge.set_label(json.loads(edge.get_label()))
        return SimpleNamespace(get_graph=lambda: graph)

    with monkeypatch.context() as patch:
        patch.setattr(original, "ClassicalSimGraphDrawer", drawer)
        original.test_classical_sim_graph()
