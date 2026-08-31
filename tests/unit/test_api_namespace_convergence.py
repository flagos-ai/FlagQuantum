from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "public-api-v1-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("section_name", ("stable_extensions", "experimental"))
def test_every_migration_destination_is_importable(section_name: str) -> None:
    candidate = _candidate()
    sections = candidate[section_name]
    assert isinstance(sections, list)

    for section in sections:
        assert isinstance(section, dict)
        namespace = section["namespace"]
        symbols = section["symbols"]
        assert isinstance(namespace, str)
        assert isinstance(symbols, list)
        module = importlib.import_module(namespace)
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        assert not missing, f"{namespace} is missing candidate exports: {missing}"


def test_backend_facade_preserves_implementation_identity() -> None:
    backends = importlib.import_module("flagquantum.backends")
    execution = importlib.import_module("flagquantum.runtime.execution")
    mps_execution = importlib.import_module("flagquantum.simulation.mps_execution")

    assert backends.run_native is execution.run_native
    assert backends.run_mps is mps_execution.run_mps


def test_compiler_facade_has_one_canonical_short_name() -> None:
    compiler = importlib.import_module("flagquantum.compiler")

    assert compiler.compile is compiler.compile_for_backend
