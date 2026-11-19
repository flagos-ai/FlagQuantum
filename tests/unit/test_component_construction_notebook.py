"""Execute the complete frozen construction tutorial and its native mapping."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_register_wiring_notebook import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")
pytest.importorskip("IPython")
pytest.importorskip("graphviz")

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/_infra/composite_bloq.ipynb"
NATIVE = ROOT / "examples/notebooks/component_construction.ipynb"


@pytest.fixture(scope="module")
def original() -> dict[int, dict[str, Any]]:
    raw = SOURCE.read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "1fe7ddcab53ddd9a909417c8994d643d8f9d8ff4"
    )
    namespace: dict[str, Any] = {}
    stages = {}
    for index, cell in enumerate(json.loads(raw)["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"source-{index}", "exec"), namespace)
            stages[index] = namespace.copy()
    return stages


@pytest.fixture(scope="module")
def native() -> dict[str, Any]:
    return execute(NATIVE)


def normalized(trace: list[Any]) -> list[Any]:
    labels = {row[0]: str(i) for i, row in enumerate(trace)}

    def collection(value: Any) -> Any:
        shape, bindings = value
        return shape, tuple(
            (labels.get(owner, owner), register, index)
            for owner, register, index in bindings
        )

    return [
        (
            str(i),
            tuple((name, collection(value)) for name, value in inputs),
            tuple(collection(value) for value in outputs),
        )
        for i, (_, inputs, outputs) in enumerate(trace)
    ]


def test_complete_cell_mapping_and_native_import_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = json.loads(SOURCE.read_text())
    actual = json.loads(NATIVE.read_text())
    assert len(actual["cells"]) == len(source["cells"]) == 42
    assert [cell["cell_type"] for cell in actual["cells"]] == [
        cell["cell_type"] for cell in source["cells"]
    ]
    assert [cell["metadata"]["frozen_source_cell"] for cell in actual["cells"]] == list(
        range(42)
    )
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("The native tutorial must use native owners")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        assert execute(NATIVE)["primitive_counts"] == {"x": 9}


@pytest.mark.parametrize(
    "cell,source_name,native_name",
    [
        (3, "cbloq", "straight"),
        (4, "cbloq", "crossed"),
        (19, "cbloq", "parallel"),
        (25, "copy", "manual_copy"),
        (27, "bloq", "nested_pair"),
        (28, "bloq", "inline_second"),
        (29, "bloq", "inline_first"),
        (30, "bloq", "inline_middle"),
        (32, "three_p", "three_wrapped"),
        (33, "flat_three_p", "flat_three"),
        (35, "decompose_children", "nested_decompositions"),
    ],
)
def test_each_construction_stage_exact_connections_and_output_semantics(
    original: dict[int, dict[str, Any]],
    native: dict[str, Any],
    cell: int,
    source_name: str,
    native_name: str,
) -> None:
    source = original[cell][source_name]
    record = native[native_name]
    assert normalized(_native_trace(record)) == normalized(
        _source_trace(SourceCalls, source)
    )
    assert verify_program_record(record) == ()
    assert_tensor_and_complex_state(source, record)
    bound = bind_register_program(
        record, ((0,), (1,)) if cell in (3, 4) else ((0, 1, 2),)
    )
    expected_count = (
        2
        if cell in (3, 4)
        else 3 if cell in (19, 25) else 6 if cell in (27, 28, 29) else 9
    )
    assert len(bound.circuit.instructions) == expected_count
    assert {instruction.name for instruction in bound.circuit.instructions} == (
        {"cx"} if cell in (3, 4) else {"x"}
    )


def assert_tensor_and_complex_state(source: Any, record: ProgramRecord) -> None:
    registers = record.functions[0].registers or ()
    offset = 0
    inputs = []
    for register in registers:
        inputs.append(tuple(range(offset, offset + register.bitsize)))
        offset += register.bitsize
    bound = bind_register_program(record, tuple(inputs))
    assert bound.events == ()
    actual = get_unitary(bound.circuit, dtype=torch.complex128)
    # Returned register order can differ from the physical output-wire order.
    order = [wire for word in bound.outputs for wire in word]
    actual = (
        actual.reshape([2] * (2 * offset))
        .permute(*order, *range(offset, 2 * offset))
        .reshape(1 << offset, 1 << offset)
    )
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    values = torch.arange(1 << (offset + 1), dtype=torch.float64)
    state = torch.complex(values + 1, values.flip(0) + 3)
    state /= state.norm()
    torch.testing.assert_close(
        actual @ state.reshape(1 << offset, 2),
        expected @ state.reshape(1 << offset, 2),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("stage", ["expanded_children", "selective", "fully_flat"])
def test_expansion_stages_and_separate_synthetic_resource_model(
    original: dict[int, dict[str, Any]], native: dict[str, Any], stage: str
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = original[41]
    if stage == "expanded_children":
        expected = source["flat_three_p"].flatten_once()
    elif stage == "selective":
        expected = source["flat_three_p"].flatten_once(lambda instance: instance.i == 1)
    else:
        expected = source["three_p"].flatten()
    assert normalized(_native_trace(native[stage])) == normalized(
        _source_trace(SourceCalls, expected)
    )
    assert_tensor_and_complex_state(expected, native[stage])
    assert (
        get_cost_value(expected, QECGatesCost()).t
        == native["source_model_estimate"]["t"]
        == 900
    )
    assert native["primitive_counts"] == {"x": 9}


def test_copy_presentation_labels_and_diagrams(native: dict[str, Any]) -> None:
    from examples.algorithms.component_construction_tutorial import labelled_connections

    assert labelled_connections(native["parallel"]) == labelled_connections(
        native["copied"]
    )
    assert len(native["original_connections"]) == len(native["copy_connections"]) == 8
    assert native["original_connections"][0] == ("input[0]", "0.reg[0]")
    assert native["copy_connections"][0] == ("input[0]", "100.reg[0]")
    for name in ("crossed", "parallel", "inline_middle", "fully_flat"):
        svg = native["show_program"](native[name]).pipe(format="svg").decode()
        assert "<svg" in svg and "<polygon" in svg and "<text" in svg
