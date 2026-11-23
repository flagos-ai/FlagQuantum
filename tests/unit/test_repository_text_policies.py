"""Complete frozen text policies and native Git/editor/tool ownership."""

import configparser
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.unit.test_development_tools import REFERENCE
from tools.check_docs_links import violations

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


def test_five_complete_policy_blobs() -> None:
    blobs = {
        ".editorconfig": "27247697fffdf501f7701d89054270ac1e0cd4e4",
        ".gitattributes": "6b61e741eb3fc17b850369eff1fc3cdd73bdc45a",
        ".gitignore": "339809b29bd7be6f2973a6ec1625aa12a72d5e78",
        ".markdownlintrc": "bbf11d7d4f04eedab5920f66ea287a9b0a611b19",
        ".yamllint.yaml": "a85c84e188c4188e85c146ff2c2cb22c64fc9c80",
    }
    for name, blob in blobs.items():
        raw = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def _editor(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string("[root]\n" + path.read_text())
    return parser


def test_complete_editor_policy_has_explicit_native_overrides() -> None:
    original = _editor(REFERENCE / ".editorconfig")
    native = _editor(ROOT / ".editorconfig")
    for section in original.sections():
        for name, value in original[section].items():
            assert native[section][name] == value
    assert native["*.py"]["max_line_length"] == "88"
    assert native["*.py"]["indent_size"] == "4"
    for section in (
        "benchmarks/results/**",
        "flagquantum/drawer/tex/yquant*.{sty,tex}",
        "flagquantum/drawer/tex/licenses/yquant.txt",
    ):
        assert native[section]["trim_trailing_whitespace"] == "false"
        assert native[section]["insert_final_newline"] == "false"
    assert native["assets/readme/*.svg"]["trim_trailing_whitespace"] == "false"


def _git(root: Path, *arguments: str, data: str | None = None) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        input=data,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "core.autocrlf", "false")
    return tmp_path


@pytest.mark.parametrize("profile", ["source", "native"])
def test_actual_git_generated_attributes_and_normalized_index(
    checkout: Path, profile: str
) -> None:
    owner = REFERENCE if profile == "source" else ROOT
    shutil.copyfile(owner / ".gitattributes", checkout / ".gitattributes")
    names = ["schema_pb2.py", "nested/schema_pb2.pyi", "handwritten.py"]
    attributes = _git(
        checkout, "check-attr", "linguist-generated", "--", *names
    ).splitlines()
    assert attributes == [
        f"{names[0]}: linguist-generated: true",
        f"{names[1]}: linguist-generated: true",
        f"{names[2]}: linguist-generated: unspecified",
    ]
    for name in names:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"value = 1\r\n")
    _git(checkout, "add", "--", *names)
    expected = "value = 1\n"
    # Binary output avoids Python's universal-newline conversion masking CRLF.
    raw = subprocess.check_output(
        ["git", "show", f":{names[0]}"], cwd=checkout, timeout=20
    )
    assert raw == (expected.encode() if profile == "native" else b"value = 1\r\n")


def test_every_original_ignore_directive_is_consumed_by_git(checkout: Path) -> None:
    shutil.copyfile(REFERENCE / ".gitignore", checkout / ".gitignore")
    patterns = [
        line
        for line in (REFERENCE / ".gitignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    paths = []
    for pattern in patterns:
        path = pattern.lstrip("/").replace("[cod]", "c").replace("*", "sample")
        paths.append(path + "entry" if path.endswith("/") else path)
    ignored = _git(checkout, "check-ignore", "--stdin", data="\n".join(paths) + "\n")
    assert ignored.splitlines() == paths
    # Ignoring generated products does not suppress authored algorithms or locks.
    result = subprocess.run(
        ["git", "check-ignore", "--", "qualtran/algo.py", "Pipfile.lock"],
        cwd=checkout,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 1 and result.stdout == b""


def test_native_git_ignores_artifacts_but_retains_templates_and_evidence(
    checkout: Path,
) -> None:
    shutil.copyfile(ROOT / ".gitignore", checkout / ".gitignore")
    excluded = [
        "module/__pycache__/a.pyc",
        "build/lib/module.py",
        "dist/package.whl",
        ".venv/bin/python",
        "outputs/state.json",
        "test-results/suite/run.xml",
        "runtime.log",
        "model.pt",
        "paper/draft.tex",
        ".scratch/probe.py",
    ]
    retained = [
        ".env.example",
        ".env.prod.template",
        "benchmarks/results/a/run.log",
        "flagquantum/algorithms/example.py",
        "docs/development/guide.md",
        "tests/unit/data/circuit.json",
    ]
    ignored = _git(
        checkout,
        "check-ignore",
        "--stdin",
        data="\n".join(excluded + retained) + "\n",
    )
    assert ignored.splitlines() == excluded
    tracked = checkout / "runtime.log"
    tracked.write_text("immutable evidence\n")
    _git(checkout, "add", "--force", "runtime.log")
    assert _git(checkout, "ls-files", "runtime.log") == "runtime.log\n"


def test_complete_original_markdown_and_yaml_declarations() -> None:
    node = shutil.which("node")
    assert node is not None, "Locked JSONC reference consumer must be available"
    parser = REFERENCE / "repomix-deps/node_modules/json5"
    result = subprocess.run(
        [
            node,
            "-e",
            "const fs=require('fs'); const j=require(process.argv[1]); "
            "process.stdout.write(JSON.stringify(j.parse(fs.readFileSync(process.argv[2], 'utf8'))));",
            str(parser),
            str(REFERENCE / ".markdownlintrc"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    markdown: dict[str, Any] = json.loads(result.stdout)
    assert set(markdown) == {
        "$schema",
        "headings",
        "first-line-heading",
        "no-trailing-punctuation",
        "whitespace",
        "line-length",
        "list-marker-space",
        "ul-indent",
        "no-bare-urls",
        "html",
    }
    assert markdown["$schema"].startswith("https://")
    assert markdown["headings"] == {"style": "atx"}
    assert markdown["first-line-heading"] is False
    assert markdown["line-length"] == {
        "line_length": 100,
        "code_block_line_length": 100,
        "heading_line_length": 100,
        "code_blocks": False,
        "headings": False,
        "tables": False,
    }
    assert markdown["list-marker-space"] == {
        "ol_multi": 2,
        "ol_single": 2,
        "ul_multi": 3,
        "ul_single": 3,
    }
    assert markdown["whitespace"] == {"br_spaces": 0}
    assert markdown["ul-indent"] == {"indent": 4}
    assert markdown["no-bare-urls"] is False
    elements = markdown["html"]["allowed_elements"]
    assert len(elements) == len(set(elements)) == 56
    assert {"details", "summary", "picture", "table"} <= set(elements)
    assert not {"script", "object", "svg"} & set(elements)
    assert yaml.safe_load((REFERENCE / ".yamllint.yaml").read_text()) == {
        "rules": {
            "line-length": {
                "max": 100,
                "allow-non-breakable-words": True,
                "allow-non-breakable-inline-mappings": True,
            }
        }
    }


def test_native_yaml_and_markdown_functional_consumers(tmp_path: Path) -> None:
    hooks = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    assert "check-yaml" in {item["id"] for item in hooks["repos"][0]["hooks"]}
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("rules: [unterminated\n")
    document = tmp_path / "guide.md"
    document.write_text("# Guide!\n\n- One\n\n[Jump](#guide)\n")
    assert violations(document) == []
    document.write_text("# Guide\n\n[Missing](missing.md)\n[Bad](#absent)\n")
    errors = violations(document)
    assert len(errors) == 2
    assert "link target does not exist" in errors[0]
    assert "anchor does not resolve" in errors[1]
