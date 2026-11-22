"""Complete site configuration, native assets and original renderer consumers."""

import hashlib
import os
import re
import runpy
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    pytest.skip(
        "Documentation renderer dependencies are optional", allow_module_level=True
    )

pytest.importorskip("sphinx")
pytest.importorskip("myst_nb")
pytest.importorskip("pydata_sphinx_theme")
nbformat = pytest.importorskip("nbformat")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "ftqc-reference-env"


def test_complete_site_source_blobs() -> None:
    expected = {
        "conf.py": "0c40213a833739ac25870b4dffb93bccc6f3eb08",
        "index.rst": "3c5b9de7c4faadb5005049a5b3b134384d0ff69d",
        "_static/fixes.css": "f6c29fa560f7a8782d034e8278d37bae7db91136",
        "_static/qai-theme.css": "1264461f94f4cf4a3cfe68aeb8f1a2efbe3ea9e2",
        "_static/favicon.png": "a7e2ecf727b43f632af7d51b30cb72e629c70ca6",
        "_static/qualtran-logo-mode-sensitive.svg": (
            "8c949095cbae25b371677a83113e96af893b1eb6"
        ),
        "_static/qualtran-logo-sm.png": "734fe052caa3fe4a089201f25d984c84df8a55f6",
        "_static/quantum-ai.svg": "ba7961a0bece0bf686166f094cf22b58f5486243",
        "_static/quantum-ai-vertical.svg": "b41872775135a0459762e66b8874b6134f28bc2b",
        "_templates/qai_logo.html": "40b0c5366676f1c7e5211a8d7c64a93a50d1e58a",
        "_templates/qualtran-sidebar.html": "e94b247c4633baf833023fe0f1a49757a72d6cc5",
    }
    for name, blob in expected.items():
        raw = (REFERENCE / "docs" / name).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_all_configuration_assignments_have_native_owners() -> None:
    original = runpy.run_path(str(REFERENCE / "docs/conf.py"))
    native = runpy.run_path(str(ROOT / "docs/conf.py"))
    names = {name for name in original if not name.startswith("__")}
    assert names == {name for name in native if not name.startswith("__")}
    adapted = {
        "project",
        "copyright",
        "author",
        "html_theme_options",
        "html_sidebars",
        "html_css_files",
        "html_logo",
        "html_favicon",
    }
    for name in names - adapted:
        assert native[name] == original[name], name
    expected_theme = dict(original["html_theme_options"])
    expected_theme["footer_center"] = ["project-logo"]
    expected_theme["icon_links"] = [
        {
            "name": "GitHub repository",
            "url": "https://github.com/flagos-ai/FlagQuantum",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ]
    assert native["html_theme_options"] == expected_theme
    assert native["project"] == "FlagQuantum"
    assert native["author"] == "FlagQuantum Team"
    assert native["copyright"] == "2026, FlagQuantum Team"
    assert native["html_logo"] == "../assets/logo_flagquantum.png"
    assert native["html_favicon"] == "_static/favicon.png"
    assert native["html_css_files"] == ["layout.css", "theme.css"]
    assert native["html_sidebars"] == {
        "concepts/README": ["section-sidebar"],
        "reference/registers/index": ["section-sidebar"],
        "reference/README": ["section-sidebar"],
    }


def _css_rules(path: Path) -> list[str]:
    content = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
    content = content.replace("--fq-", "--qai-")
    return [
        re.sub(r"[\s\"']", "", match)
        for match in re.findall(r"[^{}]+\{[^{}]*\}", content)
    ]


def test_all_stylesheet_rules_retain_source_behavior() -> None:
    original = REFERENCE / "docs/_static"
    native = ROOT / "docs/_static"
    assert _css_rules(native / "theme.css") == _css_rules(original / "qai-theme.css")
    source_layout = _css_rules(original / "fixes.css")
    native_layout = _css_rules(native / "layout.css")
    assert native_layout[: len(source_layout)] == source_layout
    assert len(native_layout) == len(source_layout) + 2
    assert "brightness(0)invert(1)" in native_layout[-1]


@pytest.mark.parametrize("profile", ("source", "native"))
def test_complete_renderer_sidebar_math_assets_and_stored_notebook(
    tmp_path: Path, profile: str
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    origin = (REFERENCE if profile == "source" else ROOT) / "docs"
    shutil.copyfile(origin / "conf.py", source / "conf.py")
    for directory in ("_static", "_templates"):
        shutil.copytree(origin / directory, source / directory)
    pages = (
        ("bloq_infra", "bloqs/index", "reference/index")
        if profile == "source"
        else ("concepts/README", "reference/registers/index", "reference/README")
    )
    (source / "index.rst").write_text(
        "Site fixture\n============\n\n.. toctree::\n\n"
        + "".join(f"   {page}\n" for page in pages)
        + "   example\n"
    )
    for index, name in enumerate(pages):
        path = source / (name + ".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Section {index}\n\n## Heading anchor\n\nInline $x^2$.\n\n$$x+y$$\n\n"
            "Term\n: Definition text.\n\n"
            '<table class="fixed responsive"><tr><td>Parameter</td></tr></table>\n'
        )
    notebook = nbformat.v4.new_notebook(
        metadata={"language_info": {"name": "python"}},
        cells=[
            nbformat.v4.new_markdown_cell("# Stored notebook"),
            nbformat.v4.new_code_cell(
                "raise RuntimeError('execution must remain off')",
                execution_count=7,
                outputs=[
                    nbformat.v4.new_output(
                        "stream", name="stdout", text="Stored output survives\n"
                    )
                ],
            ),
        ],
    )
    nbformat.write(notebook, source / "example.ipynb")
    arguments = [sys.executable, "-m", "sphinx", "-W", "-b", "html"]
    if profile == "source":
        for key, name in (
            ("html_logo", "qualtran-logo-mode-sensitive.svg"),
            ("html_favicon", "favicon.png"),
        ):
            arguments.extend(["-D", f"{key}={source / '_static' / name}"])
    else:
        (tmp_path / "assets").mkdir()
        shutil.copyfile(
            ROOT / "assets/logo_flagquantum.png",
            tmp_path / "assets/logo_flagquantum.png",
        )
    output = tmp_path / "html"
    result = subprocess.run(
        [*arguments, str(source), str(output)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARNING" not in result.stderr
    for index, name in enumerate(pages):
        path = output / (name + ".html")
        page = BeautifulSoup(path.read_text(), "html.parser")
        sidebar = page.select_one(".bd-links__title")
        assert sidebar is not None and sidebar.get_text() == f"Section {index}"
        assert page.select_one("#heading-anchor") is not None
        assert page.select_one(".math") is not None
        assert page.select_one("dl") is not None
        assert "Definition text." in page.get_text()
        if profile == "native":
            footer = page.select_one(".project-footer-logo")
            assert footer is not None and footer.get("alt") == "FlagQuantum"
            image = footer.get("src")
            assert isinstance(image, str) and "://" not in image
            assert (path.parent / image).read_bytes() == (
                ROOT / "assets/logo_flagquantum.png"
            ).read_bytes()
    rendered = (output / "example.html").read_text()
    assert "execution must remain off" in rendered
    assert "Stored output survives" in rendered
    stylesheet = "fixes.css" if profile == "source" else "layout.css"
    assert (output / "_static" / stylesheet).read_bytes() == (
        origin / "_static" / stylesheet
    ).read_bytes()
    if profile == "native":
        assert (output / "_static/favicon.png").read_bytes() == (
            ROOT / "docs/_static/favicon.png"
        ).read_bytes()


def test_native_landing_and_register_portal_targets_exist() -> None:
    for path in (ROOT / "docs/index.rst", ROOT / "docs/reference/registers/index.rst"):
        for target in re.findall(
            r"^   (?:[^<>\n]+ <)?([\w./-]+)>?$", path.read_text(), re.M
        ):
            destination = path.parent / target
            assert (
                destination.with_suffix(".md").is_file()
                or destination.with_suffix(".rst").is_file()
            ), target
    document = (ROOT / "docs/index.rst").read_text()
    example = re.search(r"\.\. code-block:: python\n\n((?:   [^\n]*\n|\n)+)", document)
    assert example is not None
    exec(textwrap.dedent(example[1]), {})
