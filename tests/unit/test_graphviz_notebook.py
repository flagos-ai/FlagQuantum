"""Whole locked graphviz notebook and standalone native graph workflow."""

import builtins
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pytest

from examples.algorithms.register_graphs import composite_dot, parallel_program
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.register_graph import register_connections
from flagquantum.drawer.signature_graph import signature_register_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


def execute(notebook: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"graph-cell-{i}", "exec"), namespace)
    return namespace


def test_all_original_code_cells_and_native_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import IPython.display
    import qualtran.drawing
    from qualtran.drawing.graphviz import PrettyGraphDrawer

    raw = (
        ROOT.parent / "ftqc-reference-env/qualtran/drawing/graphviz.ipynb"
    ).read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "977242ab7fed3bbe5ef11a2befd807007ab69a02"
    )
    source = json.loads(raw)
    renders: list[bytes] = []

    def show(bloq: Any) -> None:
        renders.append(PrettyGraphDrawer(bloq).get_svg_bytes())

    with monkeypatch.context() as patch:
        patch.setattr(qualtran.drawing, "show_bloq", show)
        original = execute(source)
    assert len(renders) == 7
    for i, svg in enumerate(renders):
        ElementTree.fromstring(svg)
        (tmp_path / f"source-{i}.svg").write_bytes(svg)
    notebook = json.loads(
        (ROOT / "examples/notebooks/register_graphs.ipynb").read_text()
    )
    assert len(notebook["cells"]) == len(source["cells"]) == 21
    assert sum(c["cell_type"] == "code" for c in notebook["cells"]) == 13
    assert len({c["id"] for c in notebook["cells"]}) == 21
    assert all(
        not c["metadata"] and not c.get("outputs") and not c.get("attachments")
        for c in notebook["cells"]
    )
    imports = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "quimb", "cirq"}:
            raise AssertionError(f"Native notebook imported {name}")
        return imports(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        patch.setattr(IPython.display, "display", lambda *args: None)
        native = execute(notebook)
    assert len(native["renders"]) == 10
    for i, svg in enumerate(native["renders"]):
        ElementTree.fromstring(svg)
        (tmp_path / f"native-{i}.svg").write_bytes(svg)
    assert [
        (r.name, int(r.bitsize), r.shape, r.side.value)
        for r in original["bloq"].signature
    ] == [(r.name, r.bitsize, r.shape, r.side.value) for r in native["registers"]]
    assert all(not c.get("outputs") for c in source["cells"])
    assert "a[0,0]" in native["signature_dot"](native["registers"])
    assert "a[0]" in native["signature_dot"](native["registers"])


@pytest.mark.parametrize("case", range(4))
def test_all_large_signature_cases_match_source_connection_widths_and_ports(
    case: int,
) -> None:
    from qualtran import Bloq, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.drawing.graphviz import GraphDrawer, PrettyGraphDrawer

    from flagquantum.drawer.register_graph_table import signature_instance_table

    source_registers = [SourceRegister("x", QAny(100))]
    registers = [Register("x", RegisterType("opaque", 100))]
    if case:
        shape: tuple[int, ...] = (2,) if case == 1 else (2, 2)
        source_registers.append(SourceRegister("a", QAny(2), shape, Side.LEFT))
        registers.append(
            Register("a", RegisterType("opaque", 2), shape, RegisterSide.LEFT)
        )
    if case >= 2:
        shape = () if case == 2 else (2,)
        width = 8 if case == 2 else 4
        source_registers.append(SourceRegister("a", QAny(width), shape, Side.RIGHT))
        registers.append(
            Register("a", RegisterType("opaque", width), shape, RegisterSide.RIGHT)
        )
    source_registers.append(SourceRegister("y", QAny(200)))
    registers.append(Register("y", RegisterType("opaque", 200)))
    source: Any = type(
        "SignatureOnly",
        (Bloq,),
        {"signature": property(lambda self: SourceSignature(source_registers))},
    )()
    drawer = GraphDrawer(source)
    expected = sorted(int(c.left.reg.bitsize) for c in drawer._cbloq.connections)
    score = signature_register_score(Signature(registers), "Signature")
    assert sorted(e.width for e in register_connections(score)) == expected
    for pretty, drawer_type in ((False, GraphDrawer), (True, PrettyGraphDrawer)):
        styled = drawer_type(source)
        graph = styled.get_graph()
        instance = next(iter(styled._binsts))
        table = signature_instance_table(
            Signature(registers),
            replace(score.columns[0].step, name=str(source)),
            {line.register: f"p{i}" for i, line in enumerate(score.lines)},
            pretty=pretty,
        )
        assert normalized_table("<" + table + ">") == normalized_table(
            graph.get_node(styled.ids[instance])[0].get_label()
        )
    assert len(score.columns) == 1
    assert sum(e.producer is None for e in register_connections(score)) == sum(
        1 for r in registers if r.side != RegisterSide.RIGHT for _ in r.all_idxs()
    )
    assert sum(e.consumer is None for e in register_connections(score)) == sum(
        1 for r in registers if r.side != RegisterSide.LEFT for _ in r.all_idxs()
    )


@pytest.mark.parametrize("right_first", [False, True])
def test_same_name_opposite_direction_never_aliases_even_in_reversed_order(
    right_first: bool,
) -> None:
    left = Register("a", RegisterType("opaque", 2), side=RegisterSide.LEFT)
    right = Register("a", RegisterType("opaque", 4), side=RegisterSide.RIGHT)
    score = signature_register_score(
        Signature((right, left) if right_first else (left, right)), "replace"
    )
    assert score.columns[0].step.inputs == ("a",)
    assert score.columns[0].step.outputs == ("out:a",)
    assert sorted(e.width for e in register_connections(score)) == [2, 4]


@pytest.mark.parametrize("width", [0, -1])
def test_invalid_signature_width_refused(width: int) -> None:
    with pytest.raises(ValueError):
        signature_register_score(
            Signature((Register("q", RegisterType("opaque", width)),)), "bad"
        )


def test_parallel_true_body_complete_complex_matrix_work_and_append() -> None:
    import torch
    from qualtran.bloqs.for_testing import TestParallelCombo
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch

    record = parallel_program()
    before = bind_register_program(record, ((0, 1, 2),))
    dot = composite_dot()
    after = bind_register_program(record, ((0, 1, 2),))
    assert before == after
    assert [g.name for g in before.circuit.instructions] == ["x"] * 3
    assert [g.wires for g in before.circuit.instructions] == [(0,), (1,), (2,)]
    expected = np.asarray(TestParallelCombo().tensor_contract())
    actual = np.column_stack(
        [
            forced_branch(
                torch.eye(16, dtype=torch.complex128)[:, i],
                before.circuit.instructions,
                (),
            ).numpy()
            for i in range(16)
        ]
    )
    # One unused spectator remains zero; compare full complex isometry columns.
    np.testing.assert_allclose(actual[::2, ::2], expected.reshape(8, 8), atol=1e-12)
    assert np.count_nonzero(actual[1::2, ::2]) == 0
    state = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128) * (1 + 2j)
    state /= torch.linalg.vector_norm(state)
    reference = torch.as_tensor(actual) @ state
    torch.testing.assert_close(
        forced_branch(state, before.circuit.instructions, ()), reference
    )
    circuit = DynamicCircuit(4, dtype=torch.complex128)
    circuit.append(FieldOperation(4, before.circuit.instructions))
    result = run_dynamic(circuit, shots=4, seed=19, strategy="trajectory")
    target = torch.zeros(16, dtype=torch.complex128)
    target[14] = 1
    torch.testing.assert_close(result.final_states.reshape(4, 16), target.expand(4, 16))
    assert get_cost_value(TestParallelCombo(), QECGatesCost()).asdict() == {"t": 300}
    assert "Split" in dot and "Join" in dot


def test_composite_complete_source_connections_and_parallel_generations() -> None:
    from collections import Counter

    from qualtran import BloqInstance
    from qualtran.bloqs.for_testing import TestParallelCombo

    from flagquantum.drawer.component_score import component_register_score

    source = TestParallelCombo().decompose_bloq()
    score = component_register_score(parallel_program())
    expected = Counter(
        (
            c.left.binst.i if isinstance(c.left.binst, BloqInstance) else None,
            c.right.binst.i if isinstance(c.right.binst, BloqInstance) else None,
            int(c.left.reg.bitsize),
        )
        for c in source.connections
    )
    assert (
        Counter((c.producer, c.consumer, c.width) for c in register_connections(score))
        == expected
    )
    assert len(score.columns) == 5
    assert len({c.generation for c in score.columns[1:4]}) == 1


def test_classical_symbolic_refusal_and_empty_signature_without_execution() -> None:
    import sympy

    from flagquantum.core.ir.program.register_types import classical_bit_type

    for dtype in (
        classical_bit_type(),
        RegisterType("opaque", sympy.Symbol("n", integer=True, positive=True)),
    ):
        with pytest.raises(ValueError, match="positive concrete"):
            signature_register_score(Signature((Register("q", dtype),)), "invalid")
    score = signature_register_score(Signature(()), "empty")
    assert len(score.columns) == 1 and not score.lines


def normalized_table(label: str) -> tuple[Any, ...]:
    """Compare HTML semantics independently of Graphviz IDs and source casing."""
    root = ElementTree.fromstring(label.strip()[1:-1])

    def node(element: ElementTree.Element) -> tuple[Any, ...]:
        attributes = tuple(
            sorted(
                (key.lower(), "port" if key.lower() == "port" else value)
                for key, value in element.attrib.items()
            )
        )
        return (
            element.tag.lower(),
            attributes,
            (element.text or "").strip(),
            tuple(node(child) for child in element),
        )

    return node(root)


@pytest.mark.parametrize("pretty", [False, True])
@pytest.mark.parametrize(
    "counts", [(0, 3), (3, 0), (1, 4), (4, 1), (2, 4), (4, 2), (2, 2), (0, 0)]
)
def test_signature_group_tables_match_unmodified_source(
    pretty: bool, counts: tuple[int, int]
) -> None:
    import pydot
    from qualtran import Bloq, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.drawing.graphviz import GraphDrawer, PrettyGraphDrawer

    from flagquantum.drawer.register_graph_table import signature_instance_table

    left, right = counts
    native = [Register("x", RegisterType("opaque", 100))]
    original = [SourceRegister("x", QAny(100))]
    for count, side, source_side in (
        (left, RegisterSide.LEFT, Side.LEFT),
        (right, RegisterSide.RIGHT, Side.RIGHT),
    ):
        if count:
            native.append(Register("a", RegisterType("opaque", 2), (count,), side))
            original.append(SourceRegister("a", QAny(2), (count,), source_side))
    native.append(Register("y", RegisterType("opaque", 200)))
    original.append(SourceRegister("y", QAny(200)))
    source: Any = type(
        "SignatureOnly",
        (Bloq,),
        {
            "signature": property(lambda self: SourceSignature(original)),
            "__str__": lambda self: "Signature<&>",
        },
    )()
    drawer = (PrettyGraphDrawer if pretty else GraphDrawer)(source)
    instance = next(iter(drawer._binsts))
    graph = drawer.get_graph()
    expected = graph.get_node(drawer.ids[instance])[0].get_label()
    signature = Signature(native)
    score = signature_register_score(signature, "Signature<&>")
    table = signature_instance_table(
        signature,
        score.columns[0].step,
        {line.register: f"p{i}" for i, line in enumerate(score.lines)},
        pretty=pretty,
    )
    assert normalized_table("<" + table + ">") == normalized_table(expected)
    # XML compares every row, local label, port, colspan and rowspan.
    assert len(register_connections(score)) == len(drawer._cbloq.connections)
    assert len(pydot.graph_from_dot_data(composite_dot(pretty=pretty))) == 1


@pytest.mark.parametrize("pretty", [False, True])
def test_composite_styles_preserve_ports_edges_and_match_source_attributes(
    pretty: bool,
) -> None:
    import pydot
    from qualtran.bloqs.for_testing import TestParallelCombo
    from qualtran.drawing.graphviz import GraphDrawer, PrettyGraphDrawer

    drawer = (PrettyGraphDrawer if pretty else GraphDrawer)(
        TestParallelCombo().decompose_bloq()
    )
    expected = drawer.get_graph()
    actual = pydot.graph_from_dot_data(composite_dot(pretty=pretty))[0]
    for instance in drawer._binsts:
        source_label = expected.get_node(drawer.ids[instance])[0].get_label()
        native_label = actual.get_node(f"n{instance.i}")[0].get_label()
        # The example uses its native Atom name for the source testing fixture.
        source_label = source_label.replace("TestAtom", "Atom")
        assert normalized_table(native_label) == normalized_table(source_label)
    assert sorted(
        tuple(sorted((k, str(v).strip(chr(34))) for k, v in e.get_attributes().items()))
        for e in actual.get_edges()
    ) == sorted(
        tuple(sorted((k, str(v).strip(chr(34))) for k, v in e.get_attributes().items()))
        for e in expected.get_edges()
    )
    other = pydot.graph_from_dot_data(composite_dot(pretty=not pretty))[0]
    assert [(e.get_source(), e.get_destination()) for e in actual.get_edges()] == [
        (e.get_source(), e.get_destination()) for e in other.get_edges()
    ]
    for node in actual.get_nodes():
        other_node = other.get_node(node.get_name())[0]

        def ports(label: str) -> list[str]:
            root = ElementTree.fromstring(label.strip()[1:-1])
            return [e.attrib["PORT"] for e in root.iter() if "PORT" in e.attrib]

        assert ports(node.get_label()) == ports(other_node.get_label())
