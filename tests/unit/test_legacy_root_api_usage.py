from __future__ import annotations

import pytest

from tools.check_legacy_root_api_usage import validate, violations_for_text

pytestmark = pytest.mark.unit


def test_user_facing_files_do_not_use_approved_legacy_root_names() -> None:
    assert validate() == ()


def test_checker_reports_only_legacy_root_access() -> None:
    text = (
        "fq.run(circuit); fq.run_native(circuit); fq.create_deployment_package(circuit)"
    )

    assert violations_for_text(
        text,
        names=("create_deployment_package", "run_native"),
    ) == ("fq.run_native", "fq.create_deployment_package")
