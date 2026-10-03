from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from flagquantum.kernels.catalog import (
    EVIDENCE,
    IMPLEMENTATIONS,
    SEMANTICS,
    validate_catalog,
)

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _module_tree(module: str) -> ast.Module:
    source = _REPOSITORY_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    return ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def _literal_all(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            return {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    return set()


def test_current_catalog_is_valid_and_has_expected_inventory() -> None:
    validate_catalog()

    assert len(SEMANTICS) == 23
    assert len(IMPLEMENTATIONS) == 30
    assert len(EVIDENCE) == 30
    assert {semantic.domain for semantic in SEMANTICS} == {
        "gradient",
        "mps",
        "numerics",
        "statevector",
    }
    maturity_by_id = {
        implementation.implementation_id: implementation.maturity
        for implementation in IMPLEMENTATIONS
    }
    assert maturity_by_id["FQKI-TRITON-MPS-003-A"] == "provisional"
    assert maturity_by_id["FQKI-TRITON-MPS-004-A"] == "provisional"
    assert maturity_by_id["FQKI-TRITON-MPS-005-A"] == "provisional"
    assert maturity_by_id["FQKI-TRITON-MPS-006-A"] == "provisional"
    assert maturity_by_id["FQKI-TRITON-MPS-007-A"] == "provisional"
    assert maturity_by_id["FQKI-FLAGTREE-SV-001-A"] == "experimental"
    assert maturity_by_id["FQKI-FLAGTREE-SV-006-A"] == "experimental"
    assert maturity_by_id["FQKI-FLAGTREE-SV-007-A"] == "experimental"
    assert maturity_by_id["FQKI-FLAGTREE-SV-008-A"] == "experimental"
    assert maturity_by_id["FQKI-FLAGTREE-GR-003-A"] == "experimental"
    assert set(maturity_by_id.values()) == {"experimental", "provisional"}


def test_cataloged_implementation_symbols_exist_without_importing_providers() -> None:
    trees: dict[str, ast.Module] = {}
    for implementation in IMPLEMENTATIONS:
        tree = trees.setdefault(
            implementation.module, _module_tree(implementation.module)
        )
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert implementation.symbol in functions
        assert implementation.symbol in _literal_all(tree)


def test_catalog_covers_every_triton_wrapper() -> None:
    triton_directory = _REPOSITORY_ROOT / "flagquantum" / "kernels" / "triton"
    cataloged = {
        (implementation.module, implementation.symbol)
        for implementation in IMPLEMENTATIONS
        if implementation.provider == "triton"
    }
    exported: set[tuple[str, str]] = set()
    for source in sorted(triton_directory.glob("*.py")):
        if source.name in {"__init__.py", "_jit.py"}:
            continue
        module = f"flagquantum.kernels.triton.{source.stem}"
        exported.update(
            (module, symbol) for symbol in _literal_all(_module_tree(module))
        )

    assert cataloged == exported


def test_evidence_references_existing_tests_and_artifacts() -> None:
    for record in EVIDENCE:
        references = (
            *record.correctness_tests,
            *record.gradient_tests,
            *record.capability_tests,
        )
        for reference in references:
            relative_path, function_name = reference.split("::", maxsplit=1)
            path = _REPOSITORY_ROOT / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            assert function_name in functions
        for relative_path in record.benchmark_artifacts:
            assert (_REPOSITORY_ROOT / relative_path).is_file()


def test_catalog_import_is_accelerator_provider_free() -> None:
    command = (
        "import sys; "
        "import flagquantum.kernels.catalog; "
        "assert 'triton' not in sys.modules; "
        "assert not any(name.startswith('flagquantum.kernels.triton') "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_validation_rejects_provider_metadata_in_semantic_id() -> None:
    invalid = replace(
        SEMANTICS[0],
        semantic_id="statevector.apply.matrix_1q.triton",
    )

    with pytest.raises(ValueError, match="provider or policy metadata"):
        validate_catalog((invalid, *SEMANTICS[1:]), IMPLEMENTATIONS)


def test_validation_rejects_unknown_implementation_semantic() -> None:
    invalid = replace(
        IMPLEMENTATIONS[0],
        semantic_id="statevector.apply.unknown.local",
    )

    with pytest.raises(ValueError, match="unknown semantic ID"):
        validate_catalog(SEMANTICS, (invalid, *IMPLEMENTATIONS[1:]))


def test_validation_rejects_implementation_without_evidence() -> None:
    with pytest.raises(ValueError, match="implementations without evidence"):
        validate_catalog(SEMANTICS, IMPLEMENTATIONS, EVIDENCE[1:])


def test_validation_requires_evidence_for_declared_derivatives() -> None:
    invalid = replace(EVIDENCE[1], gradient_tests=())

    with pytest.raises(ValueError, match="missing gradient evidence"):
        validate_catalog(
            SEMANTICS, IMPLEMENTATIONS, (EVIDENCE[0], invalid, *EVIDENCE[2:])
        )


def test_validation_requires_evidence_for_internal_fallbacks() -> None:
    invalid = replace(EVIDENCE[6], capability_tests=())

    with pytest.raises(ValueError, match="missing fallback evidence"):
        validate_catalog(
            SEMANTICS,
            IMPLEMENTATIONS,
            (*EVIDENCE[:6], invalid, *EVIDENCE[7:]),
        )
