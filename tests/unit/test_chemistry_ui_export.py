"""Complete chemistry UI group and original actual factory metadata."""

import importlib
import json
import sys
import types
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.algorithms import chemistry_ui_export as native
from tests.unit.test_component_ui_export import source_metadata
from tests.unit.test_development_tools import REFERENCE
from tests.unit.test_other_ui_export import _port_record as _shared_port_record
from tools.development.component_ui_export import ComponentUIEncoder

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def originals() -> Iterator[tuple[Any, dict[str, Any]]]:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(REFERENCE / "openfermion-reference"))
        package = types.ModuleType("qualtran_dev_tools")
        package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
        patch.setitem(sys.modules, package.__name__, package)
        repository = importlib.import_module("qualtran_dev_tools.git_tools")
        patch.setattr(repository, "get_git_root", lambda: REFERENCE)
        patch.delitem(sys.modules, "qualtran_dev_tools.notebook_specs", raising=False)
        source = importlib.import_module("qualtran_dev_tools.notebook_specs")
        sections = tuple(
            (name, specs) for name, specs in source.NB_BY_SECTION if name == "Chemistry"
        )
        yield sections, {
            ("projectile." if "WithProj" in doc.bloq_cls.__name__ else "")
            + example.name: example.make()
            for _, specs in sections
            for spec in specs
            for doc in spec.bloq_specs
            for example in doc.examples
        }


def test_complete_original_group_and_factory_order(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    sections, examples = originals
    actual = native.sections()
    assert examples.keys() == native.INSTANCES.keys() and len(examples) == 40
    assert len(actual) == len(sections) == 1
    assert len(actual[0][1]) == 13
    assert sum(len(spec.component_specs) for _, specs in actual for spec in specs) == 39
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
    if original and type(port.dtype).__name__ == "QUInt":
        return port.name, port.bitsize, port.shape, port.side.value, "uint", {}
    return _shared_port_record(port, original=original)


def _compare(actual: object, expected: object) -> None:
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and actual.keys() == expected.keys()
        for key in expected:
            if key == "side":
                assert actual[key].value == expected[key].value
            else:
                _compare(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _compare(left, right)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("identifier", tuple(native.INSTANCES))
def test_all_actual_factories_types_parameters_and_encodings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    source = originals[1][identifier]
    actual_ports = [
        _port_record(p, original=False) for p in native.signature(identifier)
    ]
    source_ports = [_port_record(p, original=True) for p in source.signature]
    assert actual_ports == source_ports
    metadata = source_metadata()
    expected = metadata["bloq_attrs"](source)
    actual = native.attributes(identifier)
    _compare(actual, expected)
    assert json.dumps(actual, cls=ComponentUIEncoder, sort_keys=True) == json.dumps(
        expected, cls=metadata["BloqEncoder"], sort_keys=True
    )
    assert native.filename(identifier) == metadata["bloq_filename"](source)


def test_fresh_complete_catalog_and_scoped_first_quantization(tmp_path: Path) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())
    assert len(navigation["Chemistry"]) == 13
    assert len(list(tmp_path.rglob("docs.txt"))) == 38
    for identifier in native.INSTANCES:
        data = json.loads(
            (
                tmp_path / native.name(identifier) / native.filename(identifier)
            ).read_text()
        )
        assert data["msd"] is not None
        assert data["msd"] == native.score(identifier)
        assert data["callees"] == []
    assert len(list(tmp_path.rglob("*.json"))) == 41
    for identifier in ("prep_first_quant", "sel_first_quant"):
        assert native.name(identifier) != native.name("projectile." + identifier)
        assert native.signature(identifier) != native.signature(
            "projectile." + identifier
        )


def test_unknown_identifier_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native example"):
        native.score("unknown")
