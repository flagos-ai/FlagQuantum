"""Complete upstream reporting sources and native project-owned routes."""

import hashlib
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from tests.unit.test_development_tools import REFERENCE
from tools.check_docs_links import link_targets, violations

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


def test_complete_reporting_support_and_direct_conduct_dependency() -> None:
    for name, blob in {
        ".github/SECURITY.md": "2a81548cfe9528761d5f651d84d474d1e32a1f55",
        ".github/report-vulnerability-button.png": "bfaabdce374ef1b208e8deb26e8f5bebbc7c7909",
        "SUPPORT.md": "5be0f2ad615a5818e608d7fb0ffae28149440cec",
        "CODE_OF_CONDUCT.md": "34ac21b4bc6d0f9e343ad69be7a3ca40f9ced9bb",
    }.items():
        raw = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_complete_original_button_image_decodes() -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise AssertionError("Pillow reference decoder is required") from error
    with Image.open(REFERENCE / ".github/report-vulnerability-button.png") as image:
        assert image.format == "PNG"
        image.load()
        assert image.size == (1446, 292)
        assert len(image.tobytes()) == image.width * image.height * len(
            image.getbands()
        )


def test_native_support_routes_resolve_and_match_package_ownership() -> None:
    support = ROOT / "SUPPORT.md"
    assert violations(support) == []
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    targets = {target for _, target in link_targets(support.read_text())}
    assert package["urls"]["Issues"] in targets
    contact = package["authors"][0]["email"]
    assert "mailto:" + contact in targets
    security = (ROOT / "SECURITY.md").read_text()
    conduct = (ROOT / "CODE_OF_CONDUCT.md").read_text()
    assert contact in security and contact in conduct
    assert "SECURITY.md#reporting-a-vulnerability" in targets
    assert "CODE_OF_CONDUCT.md#enforcement" in targets
    assert "CONTRIBUTING.md" in targets and "GOVERNANCE.md" in targets
    assert "quantumlib/Qualtran" not in support.read_text()
    assert not re.search(r"@(google\.com|googlegroups\.com)", support.read_text())


def test_conduct_dependency_resolves_through_existing_native_policy() -> None:
    source = (REFERENCE / "SUPPORT.md").read_text()
    # The frozen blockquote wraps its link label across two physical lines.
    assert re.search(r"\[code of\s*>\s*conduct\]\(CODE_OF_CONDUCT\.md\)", source)
    assert violations(ROOT / "CODE_OF_CONDUCT.md") == []
    assert violations(ROOT / "SECURITY.md") == []
