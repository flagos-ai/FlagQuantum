from pathlib import Path

import pytest

from tools.write_audit_requirements import audit_requirement, main

pytestmark = pytest.mark.unit


def test_audit_requirement_removes_only_the_local_build_suffix() -> None:
    assert audit_requirement("torch", "2.13.0+cpu") == "torch==2.13.0"
    assert audit_requirement("Example_Package", "1.2.3rc1") == (
        "example-package==1.2.3rc1"
    )


def test_audit_snapshot_is_pinned_and_excludes_the_editable_project(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit-requirements.txt"

    assert main([str(output)]) == 0

    requirements = output.read_text(encoding="utf-8").splitlines()
    assert requirements == sorted(set(requirements))
    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert all(
        not requirement.startswith("flagquantum==") for requirement in requirements
    )
