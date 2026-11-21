"""Complete Basic Gates UI instance mapping through native domain owners."""

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.algorithms import basic_gate_ui_export as basic
from examples.algorithms import register_gate_ui_export as native
from examples.algorithms.control_predicate_catalog import PredicateExample
from examples.algorithms.controlled_via_and import ControlledHadamardExample
from tests.unit.test_component_ui_export import source_metadata
from tests.unit.test_development_tools import REFERENCE
from tools.development.component_ui_export import ComponentUIEncoder

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXAMPLES = {
    example.name: example.make()
    for _, _, examples in (*basic.DOCUMENTS, *native.DOCUMENTS)
    for example in examples
}


@pytest.fixture
def originals(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    repository = importlib.import_module("qualtran_dev_tools.git_tools")
    monkeypatch.setattr(repository, "get_git_root", lambda: REFERENCE)
    monkeypatch.delitem(sys.modules, "qualtran_dev_tools.notebook_specs", raising=False)
    source = importlib.import_module("qualtran_dev_tools.notebook_specs")
    specs = source.NB_BY_SECTION[0][1]
    return specs, {
        example.name: example.make()
        for spec in specs
        for documentation in spec.bloq_specs
        for example in documentation.examples
    }


def test_entire_catalog_order_groups_factories_ports_and_parameters(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    specs, examples = originals
    assert examples.keys() == EXAMPLES.keys() and len(EXAMPLES) == 51
    actual = native.basic_gate_sections()[0][1]
    assert len(actual) == len(specs) == 15
    assert [spec.title for spec in actual] == [spec.title for spec in specs]
    assert sum(len(spec.component_specs) for spec in actual) == 35
    metadata = source_metadata()
    for target, source in zip(actual, specs, strict=True):
        assert [native.documentation_name(doc) for doc in target.component_specs] == [
            doc.bloq_cls.__name__ for doc in source.bloq_specs
        ]
        assert [
            [example.name for example in doc.examples] for doc in target.component_specs
        ] == [[example.name for example in doc.examples] for doc in source.bloq_specs]
    for identifier, call in EXAMPLES.items():
        source = examples[identifier]
        assert native.name(call) == type(source).__name__
        assert [
            (p.name, p.bitsize, p.shape, p.side.value) for p in native.signature(call)
        ] == [(p.name, p.bitsize, p.shape, p.side.value) for p in source.signature]
        if isinstance(call, (PredicateExample, ControlledHadamardExample)):
            # Native control metadata retains values and signed typed shapes.
            groups = native.attributes(call)["controls"]
            assert isinstance(groups, tuple)
            for group, values, dtype in zip(
                groups, source.ctrl_spec.cvs, source.ctrl_spec.qdtypes, strict=True
            ):
                assert group["bitsize"] == dtype.num_qubits
                assert group["shape"] == values.shape
                if group["values"] is None:
                    assert type(values).__name__ == "Shaped"
                else:
                    assert group["values"] == tuple(values.flat)
                    assert group["encoding"] == (
                        "signed"
                        if type(dtype).__name__ == "QInt"
                        else "uint" if type(dtype).__name__ == "QUInt" else "bit"
                    )
            continue
        expected = json.dumps(
            metadata["bloq_attrs"](source), cls=metadata["BloqEncoder"], sort_keys=True
        )
        assert (
            json.dumps(native.attributes(call), cls=ComponentUIEncoder, sort_keys=True)
            == expected
        )
        assert native.filename(call) == metadata["bloq_filename"](source)


def test_control_shape_collision_is_retained_as_source_evidence(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    metadata = source_metadata()
    _, examples = originals
    assert metadata["bloq_filename"](examples["ctrl_on_bits"]) == metadata[
        "bloq_filename"
    ](examples["ctrl_on_nd_bits"])
    assert native.filename(EXAMPLES["ctrl_on_bits"]) != native.filename(
        EXAMPLES["ctrl_on_nd_bits"]
    )
    # Changes in actual activation values also contribute to native identities.
    assert (
        native.attributes(EXAMPLES["ctrl_on_bits"])["controls"]
        != native.attributes(EXAMPLES["ctrl_on_nd_bits"])["controls"]
    )


def test_full_basic_gate_export_has_all_fresh_interfaces(tmp_path: Path) -> None:
    native.export_example(tmp_path)
    nav = json.loads((tmp_path / "navigation.json").read_text())["Basic Gates"]
    assert len(nav) == 15
    assert sum(len(docs) for docs in nav.values()) == 35
    assert sum(len(doc["examples"]) for docs in nav.values() for doc in docs) == 51
    assert len(list(tmp_path.rglob("docs.txt"))) == 35
    assert len(list(tmp_path.rglob("*.json"))) == 52
    for component in EXAMPLES.values():
        path = tmp_path / native.name(component) / native.filename(component)
        record = json.loads(path.read_text())
        assert record["msd"] == native.score(component)
        assert record["msd"] is not None and record["callees"] == []
        assert (
            path.with_name("docs.txt")
            .read_text()
            .startswith("# " + native.name(component) + "\n")
        )
    # Large/symbolic examples only build logical typed interface graphs here.
    assert native.signature(EXAMPLES["swap_large"]).n_qubits() == 128


def test_unregistered_instance_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native instance"):
        native.name(object())


DECOMPOSITION_EXAMPLES = (
    "approx_cswap_small",
    "int_state",
    "int_effect",
    "multi_and",
    "swap_small",
    "cswap_small",
    "ctrl_on_int",
    "ctrl_on_bits",
    "ctrl_on_nd_bits",
    "ctrl_on_multiple_values",
    "ctrl_on_symbolic_cv",
    "ctrl_on_symbolic_cv_multi",
)


def _canonical_score(score: dict[str, Any]) -> dict[str, Any]:
    # Native SSA and boundary dtype spellings deliberately retain native identity.
    result = dict(score)
    ports = []
    for original in score["soqs"]:
        port = dict(original)
        identity = port.pop("ident")
        if identity.startswith("dang,"):
            port["symb_attributes"] = {"text": "boundary", "fontsize": 8}
        ports.append(port)
    result["soqs"] = ports
    return result


@pytest.mark.parametrize("identifier", DECOMPOSITION_EXAMPLES)
def test_fresh_source_decomposition_geometry_glyphs_and_order(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    from examples.algorithms.basic_gate_decomposition import decomposition_program

    source = get_musical_score_data(originals[1][identifier].decompose_bloq())
    source_data = {
        "max_x": source.max_x,
        "max_y": source.max_y,
        **{
            key: [item.json_dict() for item in getattr(source, key)]
            for key in ("soqs", "hlines", "vlines")
        },
    }

    # The source UI encoder loses NumPy bools as NOT_SERIALIZABLE. Compare the
    # actual glyph values here and retain the serializer defect separately.
    def source_scalar(value: object) -> object:
        assert isinstance(value, np.generic)
        return value.item()

    expected = json.loads(json.dumps(source_data, default=source_scalar))
    actual = native.score(EXAMPLES[identifier])
    assert isinstance(actual, dict)
    assert _canonical_score(actual) == _canonical_score(expected)
    program = decomposition_program(EXAMPLES[identifier])
    assert program is not None
    operations = program.functions[0].blocks[0].operations
    assert len(operations) == sum(
        originals[1][identifier].decompose_bloq().bloq_counts().values()
    )
    assert all(operation.opcode == "call" for operation in operations)


def test_control_archive_collision_uses_fresh_instance_decompositions(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    metadata = source_metadata()
    source = originals[1]["ctrl_on_int"].decompose_bloq()
    from qualtran.drawing import get_musical_score_data

    encoded = json.dumps(get_musical_score_data(source), cls=metadata["BloqEncoder"])
    assert "NOT_SERIALIZABLE" in encoded
    assert "NOT_SERIALIZABLE" not in json.dumps(native.score(EXAMPLES["ctrl_on_int"]))
    assert native.score(EXAMPLES["ctrl_on_bits"]) != native.score(
        EXAMPLES["ctrl_on_nd_bits"]
    )


@pytest.mark.parametrize(
    "identifier,columns",
    (("swap_large", 68), ("cswap_large", 68), ("approx_cswap_large", 1543)),
)
def test_source_large_swap_rendering_failure_and_native_graph(
    originals: tuple[Any, dict[str, Any]], identifier: str, columns: int
) -> None:
    from qualtran.drawing import get_musical_score_data

    with pytest.raises(IndexError, match="index out of range"):
        get_musical_score_data(originals[1][identifier].decompose_bloq())
    score = native.score(EXAMPLES[identifier])
    assert isinstance(score, dict) and score["max_x"] == columns
