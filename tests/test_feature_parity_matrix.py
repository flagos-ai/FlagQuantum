"""Feature parity matrix coverage guard."""

from pathlib import Path


def test_feature_parity_matrix_indexes_authoritative_capability_sources():
    root = Path(__file__).resolve().parents[1]
    matrix = (root / "docs" / "FEATURE_PARITY_MATRIX.md").read_text(encoding="utf-8")

    for source in (
        "capability-maturity.toml",
        "public_api_v1.json",
        "operator_manifest.json",
        "runtime_contracts.schema.json",
        "benchmarks/results/",
        "pyproject.toml",
    ):
        assert source in matrix

    assert "not a second capability database" in matrix
    assert "generated/CAPABILITIES.md" in matrix
    assert "future intent" in matrix
