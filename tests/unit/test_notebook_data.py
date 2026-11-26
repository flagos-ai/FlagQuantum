"""Qualify actual distribution RECORD paths for pip target and prefix installs."""

from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

from tests.unit import notebook_data

pytestmark = pytest.mark.unit


def distribution_record(root: Path, entries: list[str]) -> PathDistribution:
    metadata = root / "nbconvert-7.16.6.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Name: nbconvert\nVersion: 7.16.6\n")
    (metadata / "RECORD").write_text("".join(f"{name},,\n" for name in entries))
    return PathDistribution(metadata)


@pytest.mark.parametrize("prefix", ["share/jupyter", "../../../share/jupyter"])
def test_actual_record_resolves_target_and_prefix_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    location = tmp_path / "environment/lib/python3.12/site-packages"
    relative = prefix + "/nbconvert/templates/markdown/conf.json"
    package = distribution_record(location, [relative])
    marker = Path(str(package.locate_file(relative)))
    marker.parent.mkdir(parents=True)
    marker.write_text('{"base_template":"base"}')
    monkeypatch.setattr(notebook_data, "distribution", lambda name: package)
    assert (
        notebook_data.installed_jupyter_data().resolve() == marker.parents[3].resolve()
    )


def test_missing_declared_template_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = distribution_record(
        tmp_path, ["share/jupyter/nbconvert/templates/markdown/conf.json"]
    )
    monkeypatch.setattr(notebook_data, "distribution", lambda name: package)
    with pytest.raises(FileNotFoundError, match="template data is missing"):
        notebook_data.installed_jupyter_data()


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [
            "one/nbconvert/templates/markdown/conf.json",
            "two/nbconvert/templates/markdown/conf.json",
        ],
    ],
)
def test_absent_or_ambiguous_template_declarations_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: list[str]
) -> None:
    package = distribution_record(tmp_path, entries)
    monkeypatch.setattr(notebook_data, "distribution", lambda name: package)
    with pytest.raises(ValueError, match="exactly one"):
        notebook_data.installed_jupyter_data()
