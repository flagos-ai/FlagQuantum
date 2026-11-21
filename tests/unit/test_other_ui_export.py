"""Complete miscellaneous UI catalog compared with immutable source factories."""

import importlib
import json
import sys
import types
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.algorithms import other_ui_export as native
from tests.unit.test_block_encoding_ui_export import _assert_parameters
from tests.unit.test_component_ui_export import source_metadata
from tests.unit.test_development_tools import REFERENCE
from tools.development.component_ui_export import ComponentUIEncoder

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def originals() -> Iterator[tuple[Any, dict[str, Any]]]:
    with pytest.MonkeyPatch.context() as patch:
        package = types.ModuleType("qualtran_dev_tools")
        package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
        patch.setitem(sys.modules, package.__name__, package)
        repository = importlib.import_module("qualtran_dev_tools.git_tools")
        patch.setattr(repository, "get_git_root", lambda: REFERENCE)
        patch.delitem(sys.modules, "qualtran_dev_tools.notebook_specs", raising=False)
        source = importlib.import_module("qualtran_dev_tools.notebook_specs")
        sections = tuple(
            (name, specs) for name, specs in source.NB_BY_SECTION if name == "Other"
        )
        yield sections, {
            example.name: example.make()
            for _, specs in sections
            for spec in specs
            for doc in spec.bloq_specs
            for example in doc.examples
        }


def test_complete_notebooks_documentation_and_factory_order(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    sections, examples = originals
    actual = native.sections()
    assert examples.keys() == native.INSTANCES.keys() and len(examples) == 38
    assert len(actual) == len(sections) == 1
    assert len(actual[0][1]) == 14
    assert sum(len(spec.component_specs) for _, specs in actual for spec in specs) == 21
    for (label, specs), (original_label, source_specs) in zip(
        actual, sections, strict=True
    ):
        assert label == original_label
        for target, source in zip(specs, source_specs, strict=True):
            assert target.title == source.title
            assert [
                native.documentation_name(doc) for doc in target.component_specs
            ] == [doc.bloq_cls.__name__ for doc in source.bloq_specs]
            assert [
                [e.name for e in doc.examples] for doc in target.component_specs
            ] == [[e.name for e in doc.examples] for doc in source.bloq_specs]


def _port_record(port: Any, *, original: bool) -> tuple[object, ...]:
    if original:
        name = type(port.dtype).__name__
        encoding = {
            "QBit": "bit",
            "QAny": "opaque",
            "BQUInt": "uint",
            "QFxp": "fixed_point",
        }[name]
        params = (
            {"bound": port.dtype.iteration_length}
            if name == "BQUInt"
            else (
                {"fractional_width": port.dtype.num_frac, "signed": port.dtype.signed}
                if name == "QFxp"
                else {}
            )
        )
    else:
        encoding, params = port.dtype.encoding, dict(port.dtype.parameters)
    return port.name, port.bitsize, port.shape, port.side.value, encoding, params


@pytest.mark.parametrize("identifier", tuple(native.INSTANCES))
def test_all_actual_nominal_interfaces_parameters_and_filenames(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    source = originals[1][identifier]
    actual_ports = [
        _port_record(p, original=False) for p in native.signature(identifier)
    ]
    source_ports = [_port_record(p, original=True) for p in source.signature]
    if "hamsim" in identifier or identifier == "hubbard_time_evolution_by_gqsp":
        # Frozen QubitizationWalkOperator constructs a set union. Retain native order.
        assert sorted(actual_ports) == sorted(source_ports)
        assert [p[0] for p in actual_ports] == [
            "signal",
            "U",
            "V",
            "p_x",
            "p_y",
            "alpha",
            "q_x",
            "q_y",
            "beta",
            "target",
        ]
    else:
        assert actual_ports == source_ports
    metadata = source_metadata()
    expected = metadata["bloq_attrs"](source)
    actual = native.attributes(identifier)
    if identifier == "apply_z_to_odd":
        callback = actual["nth_gate"]
        assert callable(callback)
        assert [callback(i) for i in range(4)] == ["i", "z", "i", "z"]
        expected = {**expected, "nth_gate": actual["nth_gate"]}
    elif identifier == "apply_lth_bloq":
        assert [type(g).__name__ for g in source.ops] == [
            "TGate",
            "Hadamard",
            "ZGate",
            "XGate",
        ]
        np.testing.assert_array_equal(actual["ops"], ["t", "h", "z", "x"])
        expected = {**expected, "ops": actual["ops"]}
    elif identifier == "select_pauli_lcu":
        rows = tuple(
            {
                "pauli": "".join(str(p) for p in gate),
                "coefficient": complex(gate.coefficient),
            }
            for gate in source.select_unitaries
        )
        assert actual["select_unitaries"] == rows
        assert native.filename(identifier) != metadata["bloq_filename"](source)
        expected = {**expected, "select_unitaries": rows}
    expected_json = json.dumps(expected, cls=metadata["BloqEncoder"], sort_keys=True)
    actual_json = json.dumps(actual, cls=ComponentUIEncoder, sort_keys=True)
    assert actual_json == expected_json

    def compare(left: object, right: object) -> None:
        if isinstance(right, Mapping):
            assert isinstance(left, Mapping) and left.keys() == right.keys()
            for key in right:
                if key == "side":
                    assert left[key].value == right[key].value
                else:
                    compare(left[key], right[key])
        elif isinstance(right, (tuple, list)):
            assert isinstance(left, (tuple, list)) and len(left) == len(right)
            for lvalue, rvalue in zip(left, right, strict=True):
                compare(lvalue, rvalue)
        else:
            _assert_parameters(left, right)

    compare(actual, expected)
    if identifier != "select_pauli_lcu":
        assert native.filename(identifier) == metadata["bloq_filename"](source)


def test_fresh_interfaces_duplicate_base_docs_and_sparse_factory_alias(
    tmp_path: Path,
) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())
    assert len(navigation["Other"]) == 14
    assert len(list(tmp_path.rglob("docs.txt"))) == 19
    assert len(list(tmp_path.rglob("*.json"))) == 38
    for identifier in native.INSTANCES:
        data = json.loads(
            (
                tmp_path / native.name(identifier) / native.filename(identifier)
            ).read_text()
        )
        assert data["msd"] is not None
        assert data["msd"] == native.score(identifier)
        assert data["callees"] == []
    assert native.filename("sparse_state_prep_alias") == native.filename(
        "sparse_state_prep_alias_from_list"
    )
    base_docs = [
        doc
        for _, specs in native.sections()
        for spec in specs
        for doc in spec.component_specs
        if not doc.examples
    ]
    assert [native.documentation_name(doc) for doc in base_docs] == ["QROMBase"] * 3


def test_unknown_identifier_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native example"):
        native.score("unknown")
