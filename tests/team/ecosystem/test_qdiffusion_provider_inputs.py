from __future__ import annotations

from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.provider_inputs import normalize_provider_identifier

pytestmark = pytest.mark.unit


def test_provider_identifier_is_trimmed_once() -> None:
    assert (
        normalize_provider_identifier("  CPQC-project  ", label="project_no")
        == "CPQC-project"
    )


@pytest.mark.parametrize(
    "value",
    (None, 7, "", "   ", "task\nname", "task\tname", "task\u200bname"),
)
def test_provider_identifier_rejects_unsafe_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="task_prefix"):
        normalize_provider_identifier(value, label="task_prefix")


@pytest.mark.parametrize(
    "entrypoint",
    (
        "qboson_live_smoke",
        "qdiffusion_system_live",
        "qdiffusion_protein_training_live",
        "qdiffusion_portability_replay_live",
    ),
)
def test_live_cli_normalizes_provider_identifiers_before_credentials(
    entrypoint: str,
) -> None:
    source = (
        Path(__file__).parents[3] / "examples" / "qdiffusion_kaiwu" / f"{entrypoint}.py"
    ).read_text(encoding="utf-8")
    main_source = source[source.index("def main() -> None:") :]

    assert main_source.count("normalize_provider_identifier(") == 2
    assert main_source.rindex("normalize_provider_identifier(") < main_source.index(
        "resolve_kaiwu_credentials()"
    )
