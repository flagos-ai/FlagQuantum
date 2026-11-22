"""Run the complete authoring guide against frozen examples and native owners."""

import ast
import hashlib
import importlib.util
import inspect
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

from flagquantum.testing.component_examples import ComponentDocumentation
from tools.development.docstrings import get_markdown_docstring, get_references
from tools.development.notebook_autogen import (
    NotebookSpec,
    example_source_lines,
    get_documentation_cells,
    get_example_cells,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "ftqc-reference-env"
GUIDE = ROOT / "docs/development/COMPONENT_DOCUMENTATION.md"


def guide_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", GUIDE.read_text(), re.S)


def test_complete_frozen_authoring_source_and_docstrings() -> None:
    raw = (REFERENCE / "qualtran/Autodoc.ipynb").read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "87a1784e4d68804069f3c4a5e8565ba4e12c5ec1"
    )
    cells = json.loads(raw)["cells"]
    assert len(cells) == 4
    assert all(cell["cell_type"] == "markdown" for cell in cells)
    rendered = []
    for index, block in enumerate(guide_blocks()[:3]):
        docstring = ast.literal_eval(block)
        assert isinstance(docstring, str)
        cls = type(f"Documentation{index}", (), {"__doc__": docstring})
        rendered.append("\n".join(get_markdown_docstring(cls)))
    assert "n" in rendered[0] and "cvs" in rendered[0]
    assert "Registers" in rendered[1] and "target [right]" in rendered[1]
    cls = type(
        "LookupDocumentation", (), {"__doc__": ast.literal_eval(guide_blocks()[2])}
    )
    references = get_references(cls)
    assert [getattr(reference, "url") for reference in references] == [
        "https://arxiv.org/abs/1805.03662",
        "https://arxiv.org/abs/2007.07391",
    ]
    assert "Figure 1" in references[0].text
    assert "Figure 3" in references[1].text


@pytest.fixture
def authored_module(tmp_path: Path) -> ModuleType:
    blocks = guide_blocks()
    path = tmp_path / "authored_component_examples.py"
    path.write_text("\n\n".join(blocks[3:]))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lazy_factories_metadata_and_generated_cells(
    authored_module: ModuleType,
) -> None:
    documentation = authored_module._MODEXP_DOC
    assert isinstance(documentation, ComponentDocumentation)
    assert [example.name for example in documentation.examples] == [
        "modexp_symb",
        "modexp_small",
        "modexp",
    ]
    symbolic, small, factoring = documentation.examples
    assert str(symbolic.make().base) == "g"
    with pytest.raises(ValueError, match="coprime"):
        small.make()
    with pytest.raises(ValueError, match="coprime"):
        factoring.make()
    assert authored_module.valid_factoring.modulus == 221
    assert authored_module.valid_factoring.base == 9
    assert authored_module.valid_factoring.exponent_width == 16
    for example in documentation.examples:
        lines = example_source_lines(example)
        assert lines[0].startswith(example.name + " = ") or lines[0].startswith("g, N")
        assert not any(line.startswith("return ") for line in lines)
    assert isinstance(authored_module._MODEXP_NOTEBOOK, NotebookSpec)
    assert authored_module._MODEXP_NOTEBOOK.component_specs == [documentation]
    doc_cells = get_documentation_cells(documentation, "authoring")
    example_cells = get_example_cells(documentation, "authoring")
    assert "ModularExponentiation" in doc_cells[0].source
    assert doc_cells[1].source == documentation.import_line
    assert len(example_cells) == 4
    assert "ModularExponentiation(3, 15, 3, 2048)" in example_cells[2].source


def test_original_small_factory_defect_and_valid_runtime_choice(
    authored_module: ModuleType,
) -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp

    runtime_raw = Path(inspect.getfile(ModExp)).read_bytes()
    assert (
        hashlib.sha1(f"blob {len(runtime_raw)}\0".encode() + runtime_raw).hexdigest()
        == "280bc11da4d7a5adef3382989b6e9bfed95a83ff"
    )

    raw = (REFERENCE / "qualtran/Autodoc.ipynb").read_text()
    cells = json.loads(raw)["cells"]
    block = re.findall(r"```python\n(.*?)\n```", "".join(cells[2]["source"]), re.S)[0]
    assert "ModExp(base=3, mod=15, exp_bitsize=3, x_bitsize=2048)" in block
    with pytest.raises(AssertionError):
        ModExp(base=3, mod=15, exp_bitsize=3, x_bitsize=2048)
    with pytest.raises(AssertionError):
        ModExp.make_for_shor(big_n=15 * 17, g=9)
    original_factoring = ModExp.make_for_shor(big_n=13 * 17, g=9)
    assert authored_module.valid_factoring.modulus == original_factoring.mod
    assert authored_module.valid_factoring.base == original_factoring.base
    assert (
        authored_module.valid_factoring.exponent_width == original_factoring.exp_bitsize
    )
    assert authored_module.valid_factoring.output_width == original_factoring.x_bitsize
    valid = ModExp(base=4, mod=15, exp_bitsize=3, x_bitsize=2048)
    native = authored_module.valid_small
    assert (
        native.base,
        native.modulus,
        native.exponent_width,
        native.output_width,
    ) == (valid.base, valid.mod, valid.exp_bitsize, valid.x_bitsize)
    for exponent in range(8):
        assert native.classical_values(exponent) == (
            exponent,
            valid.on_classical_vals(exponent=exponent)["x"],
        )
