from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_readme_links_architecture_map():
    root = Path(__file__).parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert (root / "ARCHITECTURE.md").is_file()
    assert "[ARCHITECTURE.md](ARCHITECTURE.md)" in readme
