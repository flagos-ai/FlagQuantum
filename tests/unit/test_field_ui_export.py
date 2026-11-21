"""Complete field and polynomial UI groups compared with locked source factories."""

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms import field_ui_export as native
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
        (name, specs)
        for name, specs in source.NB_BY_SECTION
        if name in ("GF Arithmetic", "Polynomials over Galois Fields")
    )
    return sections, {
        example.name: example.make()
        for _, specs in sections
        for spec in specs
        for documentation in spec.bloq_specs
        for example in documentation.examples
    }


def test_both_complete_groups_and_factory_order(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    sections, examples = originals
    actual = native.sections()
    assert len(actual) == len(sections) == 2
    assert sum(len(specs) for _, specs in actual) == 8
    assert sum(len(spec.component_specs) for _, specs in actual for spec in specs) == 15
    assert examples.keys() == native.INSTANCES.keys() and len(examples) == 22
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


def _field_parameters(dtype: Any) -> dict[str, object]:
    result: dict[str, object] = {
        "characteristic": dtype.characteristic,
        "degree": dtype.degree,
        "order": dtype.characteristic**dtype.degree,
    }
    if dtype.irreducible_poly is not None:
        field = dtype.irreducible_poly.field
        result.update(
            polynomial=int(dtype.irreducible_poly),
            polynomial_field_characteristic=int(field.characteristic),
            polynomial_field_degree=int(field.degree),
            polynomial_field_primitive_element=int(field.primitive_element),
            polynomial_field_modulus=int(field.irreducible_poly),
        )
    return result


@pytest.mark.parametrize("identifier", tuple(native.INSTANCES))
def test_all_exact_original_parameters_and_nominal_field_ports(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    _, examples = originals
    source = examples[identifier]
    assert native.name(identifier) == type(source).__name__
    ports = native.signature(identifier)
    assert [(p.name, p.bitsize, p.shape, p.side.value) for p in ports] == [
        (p.name, p.bitsize, p.shape, p.side.value) for p in source.signature
    ]
    for port, original in zip(ports, source.signature, strict=True):
        dtype = original.dtype
        if type(dtype).__name__ == "QBit":
            assert port.dtype.encoding == "bit" and not port.dtype.parameters
        elif type(dtype).__name__ == "QGF":
            assert port.dtype.encoding == "finite_field"
            assert dict(port.dtype.parameters) == _field_parameters(dtype)
        else:
            assert type(dtype).__name__ == "QGFPoly"
            assert port.dtype.encoding == "finite_field_polynomial"
            assert dict(port.dtype.parameters) == {
                "polynomial_degree": dtype.degree,
                **_field_parameters(dtype.qgf),
            }
    metadata = source_metadata()
    assert dict(native.attributes(identifier)) == metadata["bloq_attrs"](source)
    assert json.dumps(
        native.attributes(identifier), cls=ComponentUIEncoder, sort_keys=True
    ) == json.dumps(
        metadata["bloq_attrs"](source), cls=metadata["BloqEncoder"], sort_keys=True
    )
    assert native.filename(identifier) == metadata["bloq_filename"](source)


def test_complete_fresh_outputs_and_original_duplicate_factories(
    tmp_path: Path,
) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())
    assert list(navigation) == ["GF Arithmetic", "Polynomials over Galois Fields"]
    assert sum(len(specs) for specs in navigation.values()) == 8
    assert (
        sum(len(docs) for specs in navigation.values() for docs in specs.values()) == 15
    )
    assert (
        sum(
            len(doc["examples"])
            for specs in navigation.values()
            for docs in specs.values()
            for doc in docs
        )
        == 22
    )
    assert len(list(tmp_path.rglob("docs.txt"))) == 15
    # Both original constant-multiplier factories describe the same instance.
    assert native.filename("gf2_multiply_by_constant") == native.filename(
        "gf2_poly_multiply_by_constant"
    )
    children = list((tmp_path / "ComponentFunction").glob("*.json"))
    assert len(children) == 333
    assert len(list(tmp_path.rglob("*.json"))) == 22 + len(children)
    for identifier in native.INSTANCES:
        record = json.loads(
            (
                tmp_path / native.name(identifier) / native.filename(identifier)
            ).read_text()
        )
        assert record["msd"] is not None
        assert record["msd"] == native.score(identifier)
        if identifier in {
            "gf16_addition",
            "gf16_add_k",
            "gf16_multiplication",
            "gf2_multiply_by_constant",
            "gf2_poly_multiply_by_constant",
            "multiplypolybyoneplusxk",
            "binarypolynomialmultiplication",
            "gf2shiftright",
            "gf2mulviakaratsuba",
            "gf2shiftleft",
            "gf16_square",
            "gf16_inverse",
            "gf2_poly_4_8_add_k",
            "gf2_poly_4_8_add",
        }:
            assert record["callees"]
            for child in record["callees"]:
                assert (tmp_path / child["name"] / child["filename"]).is_file()
        else:
            assert record["callees"] == []
    split, join = native.signature("gf_poly_split"), native.signature("gf_poly_join")
    assert tuple(port.adjoint() for port in reversed(tuple(split))) == tuple(join)
    assert [port.name for port in split] == ["reg", "reg"]
    assert split[0].bitsize == split[1].total_bits() == 15


def test_unregistered_identifier_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native example"):
        native.score("unknown")
