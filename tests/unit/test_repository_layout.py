from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

import pytest

from tools.check_repository_hygiene import layout_violations

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_capability_contracts_are_grouped_and_parseable() -> None:
    assert list(ROOT.glob("*-contract.toml")) == []

    contracts = sorted((ROOT / "contracts").glob("*-contract.toml"))
    assert contracts
    for contract in contracts:
        payload = tomllib.loads(contract.read_text(encoding="utf-8"))
        assert payload["schema"].startswith("flagquantum_")


def test_repository_layout_does_not_regress() -> None:
    assert layout_violations(ROOT) == ()
