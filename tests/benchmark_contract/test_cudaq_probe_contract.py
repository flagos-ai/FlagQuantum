"""Contract for the two CUDA-Q baseline probes named by the parity contract.

Commit 836bf8b added ``benchmarks/cudaq_backend_compare.py`` as a 615-line probe
that called ``fq.Hamiltonian``, ``fq.pauli_term``, and
``fq.compile_quantum_kernel``. None of those names exist on the ``flagquantum``
package, so the probe raised ``AttributeError`` on its first Hamiltonian build.
Nothing caught it: no test, tool, or CI lane imported the file. These tests are
the check that was missing, plus the provenance requirements a probe needs
before its payload can be landed as evidence.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

import flagquantum

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).parents[2]
PROBES = (
    ROOT / "benchmarks" / "cudaq_backend_compare.py",
    ROOT / "benchmarks" / "cudaq_gradient_capability_probe.py",
)
PARITY_CONTRACT = ROOT / "contracts" / "cudaq-parity-matrix.toml"


def _flagquantum_attributes(path: Path) -> set[str]:
    """Return every attribute name the module reads from the ``fq`` alias."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = {"fq"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "flagquantum"
            )
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    }


@pytest.mark.parametrize("probe", PROBES, ids=lambda path: path.name)
def test_every_flagquantum_attribute_a_probe_uses_exists(probe: Path) -> None:
    missing = sorted(
        name
        for name in _flagquantum_attributes(probe)
        if not hasattr(flagquantum, name)
    )
    assert not missing, (
        f"{probe.name} reads {missing} from the flagquantum package, and the "
        "package does not export them; the probe cannot run"
    )


def _parity_contract() -> dict[str, object]:
    return tomllib.loads(PARITY_CONTRACT.read_text(encoding="utf-8"))


def test_every_probe_the_parity_contract_names_exists() -> None:
    baseline = _parity_contract()["baseline"]
    assert isinstance(baseline, dict)
    named = baseline["probes"]
    assert isinstance(named, list)
    assert named, "the parity contract must name the probes behind its baseline"
    for entry in named:
        assert (ROOT / entry["path"]).is_file(), entry["path"]


@pytest.mark.parametrize("probe", PROBES, ids=lambda path: path.name)
def test_every_probe_declares_a_schema_identifier(probe: Path) -> None:
    tree = ast.parse(probe.read_text(encoding="utf-8"), filename=str(probe))
    identifiers = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert any(
        value.startswith("flagquantum.") and value.endswith(".v1")
        for value in identifiers
    ), f"{probe.name} declares no versioned payload schema identifier"


@pytest.mark.parametrize("probe", PROBES, ids=lambda path: path.name)
def test_every_probe_records_source_identity(probe: Path) -> None:
    text = probe.read_text(encoding="utf-8")
    assert "source_identity" in text, (
        f"{probe.name} records no source identity, so its payload cannot be "
        "audited against a revision"
    )


def test_backend_compare_emits_a_non_release_payload_with_provenance() -> None:
    probe = ROOT / "benchmarks" / "cudaq_backend_compare.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--device",
            "cpu",
            "--n-wires",
            "2",
            "--layers",
            "1",
            "--observable",
            "ising",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--container-digest",
            "sha256:contract-test",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)

    assert payload["schema"] == "flagquantum.external.cudaq_backend_compare.v1"
    assert payload["benchmark"] == "cudaq_backend_compare"
    assert payload["benchmark_evidence_class"] == "comparison"
    assert payload["non_release_evidence"] is True
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    identity = payload["source_identity"]
    assert identity["container_digest"] == "sha256:contract-test"
    assert identity["commit"]
    assert identity["workload_sha256"]
    # The probe records the comparison axes a reader needs before any ratio is
    # meaningful, and it publishes no claim sentence of its own.
    assert payload["gradient_method"] in {
        "parameter-shift",
        "central-difference",
        "forward-difference",
        "finite-difference",
    }
    assert set(payload["dtype"]) == {"flagquantum_jax", "cudaq"}
    assert payload["warmup"] == 0
    assert payload["iters"] == 1
    assert "recommended_claim" not in payload.get("conclusion", {})
