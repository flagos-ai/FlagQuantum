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


def test_every_stable_migration_destination_is_importable() -> None:
    candidate = _candidate()
    sections = candidate["stable_extensions"]
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


def test_historical_experimental_inventory_is_superseded_without_rewriting_it() -> None:
    candidate = _candidate()
    surface = json.loads(
        (ROOT / "contracts" / "experimental-surface-v2-candidate.json").read_text(
            encoding="utf-8"
        )
    )

    assert candidate["experimental"]
    assert surface["proposal"].endswith(
        "API_CHANGE_PROPOSAL_011_EXPERIMENTAL_SURFACE.md"
    )
    assert surface["rules"]["stable_core_changed"] is False
    assert surface["transition"]["internal_routes_removed"] is True


def test_expert_interfaces_preserve_implementation_identity() -> None:
    runtime = importlib.import_module("flagquantum.runtime")
    execution = importlib.import_module("flagquantum.runtime.execution")
    mps = importlib.import_module("flagquantum.simulation.mps")
    mps_entrypoints = importlib.import_module("flagquantum.simulation.mps.entrypoints")
    tensor_network = importlib.import_module("flagquantum.simulation.tensor_network")
    tensor_entrypoints = importlib.import_module(
        "flagquantum.simulation.tensor_network.entrypoints"
    )

    assert runtime.run_native is execution.run_native
    assert mps.run_mps is mps_entrypoints.run_mps
    assert tensor_network.run_tensor_network is tensor_entrypoints.run_tensor_network
    assert "run_advanced" not in execution.__all__


def test_compiler_facade_has_one_canonical_short_name() -> None:
    compiler = importlib.import_module("flagquantum.compiler")

    assert callable(compiler.compile)
    assert callable(compiler.optimize)
    assert not hasattr(compiler, "compile_for_backend")
    assert not hasattr(compiler, "simple_compile")
    assert not hasattr(compiler, "remove_identity_gates")
    assert not hasattr(compiler, "merge_self_inverse")
    assert not hasattr(compiler, "merge_adjacent_rotations")
    assert not hasattr(compiler, "channel_instruction")


def test_historical_api_aggregators_are_not_shipped() -> None:
    assert importlib.util.find_spec("flagquantum.api") is None
    assert importlib.util.find_spec("flagquantum.agent") is None
    assert importlib.util.find_spec("flagquantum.agent_services") is None
    assert importlib.util.find_spec("flagquantum.backends") is None
    assert importlib.util.find_spec("flagquantum.services") is not None
    assert importlib.util.find_spec("flagquantum.runtime.compatibility") is None


def test_runtime_contracts_is_a_reexport_only_facade() -> None:
    runtime_contracts = importlib.import_module("flagquantum.runtime.contracts")

    locally_defined = {
        name
        for name in runtime_contracts.__all__
        if getattr(getattr(runtime_contracts, name), "__module__", None)
        == runtime_contracts.__name__
    }

    assert locally_defined == set()
