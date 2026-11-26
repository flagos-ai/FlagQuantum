"""Public values and offline workers survive implementation relocation."""

from __future__ import annotations

import importlib
import json
import os
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import get_type_hints

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import HamiltonianTerm, VQEResult
from flagquantum.compute.local import registry
from flagquantum.compute.local.contracts import MemorySnapshot
from flagquantum.compute.remote.packages import CloudBackendProfile, DeploymentPackage
from flagquantum.core.operators.registry import OperatorRegistry
from flagquantum.runtime.results.counts import PauliMeasurementPlan
from flagquantum.runtime.results.trajectories import MPSMonteCarloResult

pytestmark = pytest.mark.unit


def test_relocated_registry_and_public_entry_share_provider_state(monkeypatch) -> None:
    historical = importlib.import_module("flagquantum.compute.registry")
    provider = object()
    monkeypatch.setattr(historical, "_PLATFORMS", {"test": provider})
    assert historical is registry
    assert registry.get_platform_runtime("test") is provider
    assert fq.compute.get_platform_runtime("test") is provider


def test_observable_pickle_keeps_its_public_identity_and_executes() -> None:
    observable = fq.X(0)
    payload = pickle.dumps(observable, protocol=4)
    assert b"flagquantum.observables" in payload
    assert b"flagquantum.core.operators" not in payload
    restored = pickle.loads(payload)
    assert type(restored) is fq.Observable
    circuit = fq.Circuit(1).h(0)
    result = fq.run(circuit, outputs=fq.expectation(restored))
    expectation = result.expectation()
    torch.testing.assert_close(expectation, torch.ones_like(expectation))


def test_platform_value_pickle_reads_through_the_original_module() -> None:
    value = MemorySnapshot(allocated_bytes=12, reserved_bytes=32)
    restored = pickle.loads(pickle.dumps(value, protocol=4))
    assert type(restored) is MemorySnapshot
    assert restored == value
    assert MemorySnapshot.__module__ == "flagquantum.compute.contracts"


@pytest.mark.parametrize(
    "cls",
    (
        fq.Observable,
        fq.OutputRequest,
        HamiltonianTerm,
        VQEResult,
        OperatorRegistry,
        CloudBackendProfile,
        DeploymentPackage,
        PauliMeasurementPlan,
        MPSMonteCarloResult,
    ),
)
def test_public_class_annotations_resolve_after_mixed_namespace_split(cls) -> None:
    assert get_type_hints(cls)


@pytest.mark.parametrize(
    "module",
    ("flagquantum.compute.realtime", "flagquantum.simulation.kernels.catalog"),
)
def test_metadata_and_protocol_imports_do_not_activate_compute(module: str) -> None:
    source = (
        "import importlib, sys; "
        f"importlib.import_module({module!r}); "
        "assert 'torch' not in sys.modules; "
        "assert 'triton' not in sys.modules; "
        "assert 'flagquantum.runtime' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", source], check=True, timeout=30)


def test_source_snapshot_runs_both_worker_entrypoints(tmp_path: Path) -> None:
    from flagquantum.compute.remote.compute._managed_program import _source_archive

    _, compressed = _source_archive()
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(compressed)
    unpacked = tmp_path / "unpacked"
    with tarfile.open(archive) as bundle:
        bundle.extractall(unpacked, filter="data")
    script = tmp_path / "entry.py"
    script.write_text(
        "def main():\n"
        "    import flagquantum\n"
        "    return {'value': 7, 'package': flagquantum.__file__}\n",
        encoding="utf-8",
    )
    environment = dict(os.environ, PYTHONPATH=str(unpacked))
    for index, module in enumerate(
        (
            "flagquantum.remote.compute._worker",
            "flagquantum.compute.remote.compute._worker",
        )
    ):
        output = tmp_path / f"result-{index}.json"
        subprocess.run(
            [sys.executable, "-m", module, str(script), str(output), "compatibility"],
            cwd=tmp_path,
            env=environment,
            check=True,
            timeout=60,
        )
        result = json.loads(output.read_text())
        assert result["run_id"] == "compatibility"
        assert result["value"]["value"] == 7
        assert result["worker"] == {"requested_gpus": 0}
        assert Path(result["value"]["package"]).is_relative_to(unpacked)


def test_relocated_environment_still_installs_the_repository_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    recipe = root / "config/environments/environment.yml"
    editable = next(
        line.strip().removeprefix("- -e ")
        for line in recipe.read_text().splitlines()
        if line.strip().startswith("- -e ")
    )
    checkout, extras = editable.split("[", 1)
    assert (recipe.parent / checkout).resolve() == root
    assert extras == "dev,jax]"
    assert (root / "pyproject.toml").is_file()


@pytest.mark.parametrize(
    "domain", ("compute/realtime", "core", "compiler", "runtime", "simulation")
)
def test_relocated_realtime_keeps_its_dependency_and_consumer_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str
) -> None:
    from tools.architecture import check_architecture as checker

    package = tmp_path / "flagquantum"
    path = package / domain / "consumer.py"
    path.parent.mkdir(parents=True)
    source = (
        "from flagquantum.runtime.results import ExecutionResult\n"
        if domain == "compute/realtime"
        else "from flagquantum.compute.realtime.protocol import RPCHeader\n"
    )
    path.write_text(source)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "PACKAGE", package)
    assert checker._realtime_dependency_errors(path)


def test_relocated_kernels_cannot_import_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.architecture import check_architecture as checker

    original = checker._imports
    target = checker.PACKAGE / "simulation/kernels/catalog/__init__.py"
    assert target.is_file()

    def with_forbidden_import(path: Path) -> tuple[tuple[str, str], ...]:
        imports = original(path)
        if path == target:
            return imports + (("flagquantum.runtime.results", "ExecutionResult"),)
        return imports

    monkeypatch.setattr(checker, "_imports", with_forbidden_import)
    assert any(
        "kernel imports forbidden layer" in error
        for error in checker.architecture_errors()
    )


@pytest.mark.parametrize(
    "historical,canonical",
    (
        (
            "flagquantum.remote.compute.jiuding",
            "flagquantum.compute.remote.compute.jiuding",
        ),
        (
            "flagquantum.kernels.catalog.schema",
            "flagquantum.simulation.kernels.catalog.schema",
        ),
        ("flagquantum.noise.channels", "flagquantum.core.operators.channels"),
        (
            "flagquantum.operators.superoperator",
            "flagquantum.core.operators.superoperator",
        ),
        ("flagquantum.core.algebra.pauli", "flagquantum.core.operators.pauli"),
        (
            "flagquantum.compiler._torch.module_builder",
            "flagquantum.compiler.frontends.torch._module_builder",
        ),
        (
            "flagquantum.compiler._hybrid.specialize",
            "flagquantum.compiler.frontends._structured.specialize",
        ),
        (
            "flagquantum.simulation.jax.primitives",
            "flagquantum.simulation.numerics.jax",
        ),
        (
            "flagquantum.simulation.jax.mps.kernels",
            "flagquantum.simulation.mps.jax.kernels",
        ),
    ),
)
def test_standard_legacy_import_after_canonical_import_shares_identity(
    historical: str, canonical: str
) -> None:
    source = (
        "import importlib; "
        f"canonical = importlib.import_module({canonical!r}); "
        f"import {historical} as historical; "
        "assert historical is canonical"
    )
    subprocess.run([sys.executable, "-c", source], check=True, timeout=30)
