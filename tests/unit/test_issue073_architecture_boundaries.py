import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_architecture import architecture_errors

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_checked_architecture_boundaries_pass():
    assert architecture_errors() == ()


def test_stable_root_import_is_lazy_and_does_not_initialize_runtimes():
    code = """
import sys
import flagquantum
forbidden = ('flagquantum.runtime_stack', 'flagquantum.deployment', 'flagquantum.drawer', 'jax', 'jaxlib')
assert not any(name.startswith(forbidden) for name in sys.modules), sorted(sys.modules)
# The v1 public surface is snapshot-governed in public_api_v1.json. Keep this
# guard as a coarse emergency ceiling without contradicting that frozen API.
assert len(flagquantum.__all__) <= 64
assert flagquantum.COMPATIBILITY_EXPORT_OWNER
assert flagquantum.COMPATIBILITY_EXPORT_REMOVAL_VERSION == '0.3.0'
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_stable_api_and_lazy_compatibility_symbols_preserve_behavior():
    import flagquantum as fq

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    assert circuit.state().shape[-1] == 4
    assert fq.DistributedQuantumDevice is not None


def test_dependency_graph_and_compatibility_registry_are_checked_in():
    graph = (ROOT / "docs" / "ARCHITECTURE_DEPENDENCIES.md").read_text(encoding="utf-8")
    policy = (ROOT / "architecture.toml").read_text(encoding="utf-8")

    assert "flowchart TD" in graph
    assert "Executors emit backend-neutral" in graph
    assert "legacy_exceptions" in policy
    assert "owner =" in policy
    assert "removal_version =" in policy
