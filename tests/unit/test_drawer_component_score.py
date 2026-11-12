"""Core component scores preserve the locked notebook's geometry and glyphs."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.classical_tensor_scores import style, symbols
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    SymbolClass,
    export_register_score_json,
    register_score_data,
)
from flagquantum.drawer.register_score_html import export_register_score_html
from flagquantum.drawer.register_score_mpl import draw_register_score
from tests.unit.classical_vector_fixtures import (
    KINDS,
    evaluate,
    example,
    original_inputs,
)
from tests.unit.component_vector_checks import tracked_flank
from tests.unit.test_component_tensor_nodes import fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def assert_source_geometry(actual: MusicalScoreData, expected: Any) -> None:
    original = {
        "max_x": expected.max_x,
        "max_y": expected.max_y,
        "soqs": [p.json_dict() for p in expected.soqs],
        "hlines": [p.json_dict() for p in expected.hlines],
        "vlines": [p.json_dict() for p in expected.vlines],
    }
    assert (actual["max_x"], actual["max_y"]) == (original["max_x"], original["max_y"])
    for key in ("hlines", "vlines"):
        assert sorted(
            map(lambda x: json.dumps(x, sort_keys=True), actual[key])
        ) == sorted(map(lambda x: json.dumps(x, sort_keys=True), original[key]))

    def ports(items: Any) -> list[str]:
        result = []
        for port in items:
            port = dict(port)
            identity = port.pop("ident")
            if identity.startswith("dang,"):
                # Native boundary labels include array indices and owned dtype spelling.
                port["symb_attributes"] = {"text": "boundary", "fontsize": 8}
            result.append(json.dumps(port, sort_keys=True))
        return sorted(result)

    assert ports(actual["soqs"]) == ports(original["soqs"])


@pytest.mark.parametrize("kind", ["xgate", "cnot", "toffoli"])
def test_atomic_shapes_and_through_lanes_match_original(kind: str) -> None:
    _, source, emitted = example(kind)
    from qualtran.drawing import get_musical_score_data

    from tests.unit.test_classical_action_tensor import signature_from_source

    signature = signature_from_source(source.signature)
    leaf = bit_gate_function("primitive", tuple(signature), emitted)
    builder = ComponentProgramBuilder(signature, (leaf,))
    record = builder.finalize(builder.call_dict("primitive", builder.inputs))
    gate = {"xgate": "x", "cnot": "cx", "toffoli": "ccx"}[kind]
    score = component_register_score(
        record, style=lambda _, step: replace(step, gate=gate)
    )
    assert_source_geometry(register_score_data(score), get_musical_score_data(source))
    assert all(line.start == -1 and line.end == 1 for line in score.lines)


@pytest.mark.parametrize("kind", KINDS)
def test_original_notebook_closed_graphs_match_all_port_labels_and_geometry(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from qualtran.drawing import get_musical_score_data
    from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors

    record, source, _ = example(kind)
    inputs = original_inputs(kind)
    closed, vectors = tracked_flank(
        monkeypatch, record, inputs, evaluate=lambda v: evaluate(kind, v)
    )
    score = component_register_score(closed, "classical_vectors")
    actual = register_score_data(score, wire_symbol=symbols(closed, vectors, kind))
    expected = get_musical_score_data(flank_with_classical_vectors(source, inputs))
    assert_source_geometry(actual, expected)
    assert all(
        line.start >= 0 and line.end < len(score.columns) for line in score.lines
    )


@pytest.mark.parametrize("width", [1, 2, 3, 5, 32])
def test_expanded_add_array_rebindings_work_allocation_and_release_match_original(
    width: int,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.drawing import get_musical_score_data

    record = addition_component_program(width)
    score = component_register_score(record, style=style)
    actual = register_score_data(score, wire_symbol=symbols(record, {}))
    expected = get_musical_score_data(Add(QUInt(width)).decompose_bloq())
    assert_source_geometry(actual, expected)
    assert sum(line.width for line in score.lines if line.start == -1) == 2 * width
    assert (
        sum(line.width for line in score.lines if line.end == len(score.columns))
        == 2 * width
    )


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_shaped_idle_lines_and_core_graph_are_preserved(shape: tuple[int, ...]) -> None:
    record, _ = fixture(2, shape)
    before = hash(record)
    score = component_register_score(record)
    assert hash(record) == before
    assert len(score.columns) == 3
    idle = [line for line in score.lines if line.register.startswith("idle")]
    assert len(idle) == (1 if not shape else int(__import__("numpy").prod(shape)))
    assert all(line.start == -1 and line.end == 3 and line.width == 2 for line in idle)


def test_style_and_wire_symbol_failures_propagate_without_changing_record() -> None:
    record = addition_component_program(2)
    with pytest.raises(ValueError, match="connectivity"):
        component_register_score(
            record, style=lambda _, step: replace(step, outputs=())
        )
    score = component_register_score(record)

    def invalid_caption(
        *args: object,
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        return "TextBox", {"text": "bad"}

    with pytest.raises(ValueError, match="connector"):
        register_score_data(score, wire_symbol=invalid_caption)

    def fail(*args: object) -> Any:
        raise RuntimeError("supplier failed")

    with pytest.raises(RuntimeError, match="supplier failed"):
        register_score_data(score, wire_symbol=fail)


def test_renderers_use_core_projection_without_quantum_or_tensor_execution(
    tmp_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    record = addition_component_program(5)
    score = component_register_score(record, style=style)
    data = register_score_data(score, wire_symbol=symbols(record, {}))
    assert json.loads(export_register_score_json(data)) == data
    html = export_register_score_html(data)
    assert "createElementNS" in html and "<script src=" not in html
    figure, axes = draw_register_score(data, max_width=20, max_height=100)
    try:
        figure.savefig(tmp_path / "addition.svg")
        assert (tmp_path / "addition.svg").stat().st_size > 1000
        assert sum(
            len(collection.get_segments())
            for collection in axes.collections
            if isinstance(collection, LineCollection)
        ) == len(data["hlines"]) + len(data["vlines"])
    finally:
        plt.close(figure)


@pytest.mark.parametrize("kind", ["zero", "classical", "unknown", "physical"])
def test_outside_display_profile_fails_before_partial_rendering(kind: str) -> None:
    if kind == "zero":
        record, _ = fixture(0, ())
    elif kind == "classical":
        from flagquantum.core.ir.program.register_types import classical_bit_type
        from flagquantum.core.ir.program.registers import Register
        from tests.unit.test_program_classical_registers import identity

        signature = Signature((Register("c", classical_bit_type()),))
        leaf = identity("identity", tuple(signature))
        builder = ComponentProgramBuilder(signature, (leaf,))
        record = builder.finalize(builder.call_dict("identity", builder.inputs))
    else:
        record = addition_component_program(2)
    if kind == "physical":
        from flagquantum.core.ir import CircuitIR, Instruction
        from flagquantum.core.ir.program import circuit_to_program

        converted = circuit_to_program(CircuitIR(1, (Instruction("x", (0,)),)))
        assert converted.record is not None
        record = converted.record
    with pytest.raises(ValueError):
        component_register_score(record, "missing" if kind == "unknown" else "main")


def test_fresh_display_labels_do_not_collide_with_user_register_names() -> None:
    from flagquantum.compiler.registers.component_basis_vectors import (
        integer_basis_function,
    )
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )

    bit = RegisterType("bit", 1)
    signature = Signature(
        (Register("call0:val", bit), Register("made", bit, side=RegisterSide.RIGHT))
    )
    builder = ComponentProgramBuilder(
        signature, (integer_basis_function("make", bit, 0),)
    )
    value = builder.call_dict("make", {})["val"]
    record = builder.finalize({"call0:val": builder.inputs["call0:val"], "made": value})
    score = component_register_score(record)
    assert {line.register for line in score.lines} == {"call0:val", "_call0:val"}
    assert {line.lane for line in score.lines} == {0, 1}
