"""Complete bookkeeping UI group compared with locked source factories."""

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms import bookkeeping_ui_export as native
from tests.unit.test_component_ui_export import source_metadata
from tests.unit.test_development_tools import REFERENCE
from tools.development.component_ui_export import ComponentUIEncoder

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def originals(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    repository = importlib.import_module("qualtran_dev_tools.git_tools")
    monkeypatch.setattr(repository, "get_git_root", lambda: REFERENCE)
    monkeypatch.delitem(sys.modules, "qualtran_dev_tools.notebook_specs", raising=False)
    source = importlib.import_module("qualtran_dev_tools.notebook_specs")
    sections = tuple(
        (name, specs) for name, specs in source.NB_BY_SECTION if name == "Bookkeeping"
    )
    return sections, {
        example.name: example.make()
        for _, specs in sections
        for spec in specs
        for documentation in spec.bloq_specs
        for example in documentation.examples
    }


def test_complete_group_and_factory_order(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    sections, examples = originals
    actual = native.sections()
    assert len(actual) == len(sections) == 1
    assert sum(len(specs) for _, specs in actual) == 6
    assert sum(len(spec.component_specs) for _, specs in actual for spec in specs) == 10
    assert examples.keys() == native.INSTANCES.keys() and len(examples) == 11
    for (name, specs), (original_name, originals_specs) in zip(
        actual, sections, strict=True
    ):
        assert name == original_name
        for target, source in zip(specs, originals_specs, strict=True):
            assert target.title == source.title
            assert [
                native.documentation_name(doc) for doc in target.component_specs
            ] == [doc.bloq_cls.__name__ for doc in source.bloq_specs]
            assert [
                [example.name for example in doc.examples]
                for doc in target.component_specs
            ] == [
                [example.name for example in doc.examples] for doc in source.bloq_specs
            ]


@pytest.mark.parametrize("identifier", tuple(native.INSTANCES))
def test_actual_factory_interfaces_and_parameter_metadata(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    source = originals[1][identifier]
    ports = native.signature(identifier)
    assert [(p.name, p.bitsize, p.shape, p.side.value) for p in ports] == [
        (p.name, p.bitsize, p.shape, p.side.value) for p in source.signature
    ]
    encodings = {
        "QBit": "bit",
        "QAny": "opaque",
        "QUInt": "uint",
        "QInt": "signed",
        "QFxp": "fixed_point",
    }
    for port, original in zip(ports, source.signature, strict=True):
        assert port.dtype.encoding == encodings[type(original.dtype).__name__]
        if type(original.dtype).__name__ == "QFxp":
            assert dict(port.dtype.parameters) == {
                "fractional_width": original.dtype.num_frac,
                "signed": original.dtype.signed,
            }
        else:
            assert not port.dtype.parameters
    metadata = source_metadata()
    original_attributes = metadata["bloq_attrs"](source)
    if identifier.startswith("auto_partition"):
        attrs = native.attributes(identifier)
        assert attrs["child"] == {
            "operation": "Swap",
            "bitsize": 1,
            "control_values": (1,),
        }
        assert int(source.bloq.ctrl_spec.cvs[0]) == 1
        assert source.bloq.subbloq.bitsize == 1
        assert attrs["left_only"] is source.left_only is False
        groups = attrs["groups"]
        assert isinstance(groups, tuple)
        for (outer, segments), (register, mappings) in zip(
            groups, source.partitions, strict=True
        ):
            assert outer == register.name
            assert tuple(child for child, _ in segments) == tuple(
                part if isinstance(part, str) else None for part in mappings
            )
            assert sum(width for _, width in segments) == register.bitsize
        encoded = json.dumps(original_attributes, cls=metadata["BloqEncoder"])
        assert "NOT_SERIALIZABLE" in encoded
        assert "NOT_SERIALIZABLE" not in json.dumps(attrs, cls=ComponentUIEncoder)
        assert native.filename(identifier) != metadata["bloq_filename"](source)
    else:
        comparable_attributes = original_attributes
        if identifier == "partition":
            comparable_attributes = {
                **original_attributes,
                "regs": tuple(
                    {
                        **port,
                        "side": native.signature(identifier)[index + 1].side.__class__(
                            port["side"].value
                        ),
                    }
                    for index, port in enumerate(original_attributes["regs"])
                ),
            }
        assert dict(native.attributes(identifier)) == comparable_attributes
        assert json.dumps(
            native.attributes(identifier), cls=ComponentUIEncoder, sort_keys=True
        ) == json.dumps(
            original_attributes, cls=metadata["BloqEncoder"], sort_keys=True
        )
        assert native.filename(identifier) == metadata["bloq_filename"](source)


def test_complete_fresh_exports_preserve_directional_aliases(tmp_path: Path) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())
    assert list(navigation) == ["Bookkeeping"]
    assert len(navigation["Bookkeeping"]) == 6
    assert len(list(tmp_path.rglob("docs.txt"))) == 10
    assert len(list(tmp_path.rglob("*.json"))) == 12
    for identifier in native.INSTANCES:
        record = json.loads(
            (
                tmp_path / native.name(identifier) / native.filename(identifier)
            ).read_text()
        )
        assert record["msd"] is not None
        assert record["msd"] == native.score(identifier)
        assert record["callees"] == []
    split, join = native.signature("split"), native.signature("join")
    assert tuple(port.adjoint() for port in reversed(tuple(split))) == tuple(join)
    assert [port.name for port in split] == ["reg", "reg"]
    assert native.signature("partition")[0].bitsize == 49
    assert native.signature("auto_partition_unused").n_qubits() == 6


def test_unknown_identifier_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native example"):
        native.score("unknown")
