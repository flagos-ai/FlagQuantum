"""Complete source discovery logic applied to native metadata and namespaces."""

import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tests.unit.test_development_tools import REFERENCE, load_source
from tools.development import component_discovery as discovery
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    package = types.ModuleType("_development_reference")
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(
        sys.modules, package.__name__ + ".git_tools", load_source("git_tools")
    )
    original = load_source("bloq_finder")
    # Substitute the existing native metadata owner at the source type boundary.
    monkeypatch.setattr(original, "BloqExample", ComponentExample)
    monkeypatch.setattr(original, "BloqDocSpec", ComponentDocumentation)
    return original


@pytest.fixture(autouse=True)
def bounded_processes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    original = subprocess.run

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        kwargs.setdefault("timeout", 5)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    yield


@pytest.fixture
def catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: types.ModuleType
) -> tuple[Path, types.ModuleType]:
    root = tmp_path / "qualtran" / "bloqs"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=5)
    for name in ("owned.py", "owned_test.py", "__init__.py"):
        (root / name).write_text("")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, timeout=5)
    (root / "local.py").write_text("")
    module = types.ModuleType("qualtran.bloqs.owned")
    own_class = type("Owned", (source.Bloq,), {"__module__": module.__name__})
    private_class = type("_Private", (source.Bloq,), {"__module__": module.__name__})
    foreign_class = type("Foreign", (source.Bloq,), {"__module__": "other_module"})
    setattr(module, "Owned", own_class)
    setattr(module, "_Private", private_class)
    setattr(module, "Foreign", foreign_class)
    setattr(module, "Other", type("Other", (), {"__module__": module.__name__}))
    example = ComponentExample(
        factory=lambda: object(), name="owned", component_type=own_class
    )
    setattr(module, "example", example)
    setattr(module, "imported_example", example)
    setattr(
        module,
        "docs",
        ComponentDocumentation(component_type=own_class, examples=(example,)),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        sys.modules, "qualtran.bloqs.local", types.ModuleType("qualtran.bloqs.local")
    )
    return root, module


@pytest.mark.parametrize("committed", [False, True])
def test_all_file_discovery_modes(
    source: types.ModuleType, catalog: tuple[Path, types.ModuleType], committed: bool
) -> None:
    root, _ = catalog
    assert discovery.get_component_module_paths(
        root, committed
    ) == source.get_bloq_module_paths(root, committed)
    assert discovery.get_component_test_module_paths(
        root, committed
    ) == source.get_bloq_test_module_paths(root, committed)
    assert discovery._get_paths(
        root, lambda path: True, committed
    ) == source._get_paths(root, lambda path: True, committed)


def test_native_test_filename_extension(catalog: tuple[Path, types.ModuleType]) -> None:
    root, _ = catalog
    (root / "test_native.py").write_text("")
    assert Path("test_native.py") not in discovery.get_component_module_paths(
        root, False
    )
    assert Path("test_native.py") in discovery.get_component_test_module_paths(
        root, False
    )


def test_all_class_example_and_documentation_consumers(
    source: types.ModuleType,
    catalog: tuple[Path, types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, module = catalog
    path = Path("owned.py")

    def predicate(cls: type) -> bool:
        return issubclass(cls, source.Bloq)

    assert (
        list(
            discovery.module_components(
                path, package="qualtran.bloqs", is_component=predicate
            )
        )
        == list(source.modpath_to_bloqs(path))
        == [module.Owned]
    )
    assert list(discovery.module_examples(path, package="qualtran.bloqs")) == list(
        source.modpath_to_bloq_exs(path)
    )
    assert list(discovery.module_documentation(path, package="qualtran.bloqs")) == list(
        source.modpath_to_bloqdocspecs(path)
    )
    assert discovery.get_component_classes(
        root, package="qualtran.bloqs", is_component=predicate
    ) == source.get_bloq_classes(root)
    assert discovery.get_component_examples(
        root, package="qualtran.bloqs"
    ) == source.get_bloq_examples(root)
    assert discovery.get_component_documentation(
        root, package="qualtran.bloqs"
    ) == source.get_bloqdocspecs(root)
    monkeypatch.setattr(discovery, "get_git_root", lambda: root.parents[1])
    native_root = root.parents[1] / "flagquantum" / "algorithms"
    native_root.mkdir(parents=True)
    (native_root / "owned.py").write_text("")
    subprocess.run(["git", "add", "."], cwd=root.parents[1], check=True, timeout=5)
    assert discovery.get_component_examples(package="qualtran.bloqs") == [
        module.example,
        module.example,
    ]
    assert discovery.get_component_documentation(package="qualtran.bloqs") == [
        module.docs
    ]
    assert discovery.get_component_classes(
        package="qualtran.bloqs", is_component=predicate
    ) == [module.Owned]


def test_native_catalog_uses_existing_lazy_metadata() -> None:
    from examples.algorithms.component_example_catalog import SWAP_DOCUMENTATION

    path = Path("component_example_catalog.py")
    examples = list(discovery.module_examples(path, package="examples.algorithms"))
    assert len(examples) == 1
    assert examples[0][2] is SWAP_DOCUMENTATION.examples[0]
    assert (
        list(discovery.module_documentation(path, package="examples.algorithms"))[0][2]
        is SWAP_DOCUMENTATION
    )
    # WordSwap is imported, so the owned-class rule does not claim it here.
    assert (
        list(
            discovery.module_components(
                path, package="examples.algorithms", is_component=lambda cls: True
            )
        )
        == []
    )


def test_import_failure_propagates(source: types.ModuleType) -> None:
    for action in (
        lambda: list(
            discovery.module_examples(
                Path("does_not_exist.py"), package="missing_catalog"
            )
        ),
        lambda: list(source.modpath_to_bloq_exs(Path("does_not_exist.py"))),
    ):
        with pytest.raises(ModuleNotFoundError):
            action()


def test_explicit_domain_predicate_for_native_class(
    catalog: tuple[Path, types.ModuleType],
) -> None:
    _, module = catalog
    assert list(
        discovery.module_components(
            Path("owned.py"),
            package=module.__package__ or "qualtran.bloqs",
            is_component=lambda cls: cls.__name__ == "Other",
        )
    ) == [module.Other]


def test_module_name_mapping(source: types.ModuleType) -> None:
    path = Path("nested/owned.py")
    assert discovery._module_name(
        path, "qualtran.bloqs"
    ) == source._bloq_modpath_to_modname(path)
    assert (
        discovery._module_name(path, "flagquantum.algorithms")
        == "flagquantum.algorithms.nested.owned"
    )


def test_frozen_class_corpus(source: types.ModuleType) -> None:
    root = REFERENCE / "qualtran" / "bloqs"
    expected = source.get_bloq_classes(root)
    actual = discovery.get_component_classes(
        root,
        package="qualtran.bloqs",
        is_component=lambda cls: issubclass(cls, source.Bloq),
    )
    assert actual == expected
    assert len(actual) == 340


def test_frozen_discovery_blob() -> None:
    import csv
    import hashlib

    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0057")
    data = (REFERENCE / row["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )
