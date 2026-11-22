"""Complete original ignore policy and native generated-output ownership."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit.test_development_tools import REFERENCE

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "docs/_build/html/index.html",
    "docs/reference/autotoc.rst.inc",
    "docs/components/topic/demo.ipynb",
    "docs/components/index.rst",
    "docs/reference/flagquantum.md",
    "docs/reference/flagquantum/circuit.md",
    "docs/reference/API.md",
    "docs/development/guide.md",
    "docs/concepts/guide.md",
    "docs/example.ipynb",
    "docs/index.rst",
    "examples/notebooks/demo.ipynb",
)


def test_complete_source_output_policy_blob() -> None:
    raw = (REFERENCE / "docs/.gitignore").read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "a82e2038fc266b5b92c91c29f7b1de80089ddd2a"
    )
    assert [
        line for line in raw.decode().splitlines() if line and not line.startswith("#")
    ] == ["_build/", "reference/autotoc.rst.inc", "**/*.md", "**/*.ipynb"]


@pytest.mark.parametrize("profile", ["source", "native"])
def test_actual_git_generated_selection_and_explicit_publication(
    profile: str, tmp_path: Path
) -> None:
    result = subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "init", "-q", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    for name in FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Fixture\n")
    origin = REFERENCE if profile == "source" else ROOT
    shutil.copyfile(origin / "docs/.gitignore", tmp_path / "docs/.gitignore")
    selected = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        input="\0".join(FILES) + "\0",
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert selected.returncode == 0, selected.stderr
    ignored = set(selected.stdout.rstrip("\0").split("\0"))
    if profile == "source":
        assert ignored == set(FILES) - {
            "docs/components/index.rst",
            "docs/index.rst",
            "examples/notebooks/demo.ipynb",
        }
    else:
        assert ignored == set(FILES[:6])
    visible = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert visible.returncode == 0, visible.stderr
    assert set(visible.stdout.rstrip("\0").split("\0")) == (
        set(FILES) - ignored | {"docs/.gitignore"}
    )
    published = "docs/reference/flagquantum/circuit.md"
    result = subprocess.run(
        ["git", "add", "--force", "--", published],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert staged.returncode == 0, staged.stderr
    assert staged.stdout.splitlines() == [published]
