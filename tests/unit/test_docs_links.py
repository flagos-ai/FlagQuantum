from pathlib import Path

import pytest

from tools.check_docs_links import heading_slugs, violations

pytestmark = pytest.mark.unit


def test_accepts_resolving_relative_link_and_anchor(tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text(
        "# Install the compiler plugin\n", encoding="utf-8"
    )
    page = tmp_path / "page.md"
    page.write_text(
        "[target](target.md)\n[anchor](target.md#install-the-compiler-plugin)\n",
        encoding="utf-8",
    )
    assert violations(page) == []


def test_reports_missing_link_target(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("[gone](missing.md)\n", encoding="utf-8")
    assert violations(page) == [f"{page}:1: link target does not exist: missing.md"]


def test_reports_unresolved_anchor(tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text("# Present\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text("[bad](target.md#absent)\n", encoding="utf-8")
    assert violations(page) == [f"{page}:1: anchor does not resolve: #absent"]


def test_reports_unresolved_same_page_anchor(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("# Present\n[bad](#absent)\n", encoding="utf-8")
    assert violations(page) == [f"{page}:2: anchor does not resolve: #absent"]


def test_ignores_external_and_trailing_hash_targets(tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text("# Present\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "[web](https://example.invalid/missing.md)\n"
        "[mail](mailto:maintainers@example.invalid)\n"
        "[no anchor](target.md#)\n",
        encoding="utf-8",
    )
    assert violations(page) == []


def test_ignores_links_inside_fenced_code(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("```markdown\n[example](missing.md)\n```\n", encoding="utf-8")
    assert violations(page) == []


def test_reports_link_inside_fenced_code_when_fence_is_closed(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("```text\nplain\n```\n[gone](missing.md)\n", encoding="utf-8")
    assert violations(page) == [f"{page}:4: link target does not exist: missing.md"]


def test_accepts_parent_directory_link(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "target.md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "page").mkdir()
    page = tmp_path / "page" / "page.md"
    page.write_text("[up](../docs/target.md#target)\n", encoding="utf-8")
    assert violations(page) == []


def test_decodes_percent_escaped_target(tmp_path: Path) -> None:
    (tmp_path / "a file.md").write_text("# Title\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text("[spaced](a%20file.md)\n", encoding="utf-8")
    assert violations(page) == []


def test_heading_slug_drops_punctuation_and_backticks() -> None:
    assert heading_slugs("## `flagquantum/qec`: the DEM plan!\n") == {
        "flagquantumqec-the-dem-plan"
    }


def test_heading_slug_disambiguates_duplicates() -> None:
    assert heading_slugs("# Direct\n## Direct\n### Direct\n") == {
        "direct",
        "direct-1",
        "direct-2",
    }


def test_heading_scan_ignores_commented_headings_in_code(tmp_path: Path) -> None:
    (tmp_path / "script.md").write_text(
        "# Real heading\n\n```python\n# not a heading\n```\n", encoding="utf-8"
    )
    page = tmp_path / "page.md"
    page.write_text("[code](script.md#not-a-heading)\n", encoding="utf-8")
    assert violations(page) == [f"{page}:1: anchor does not resolve: #not-a-heading"]
