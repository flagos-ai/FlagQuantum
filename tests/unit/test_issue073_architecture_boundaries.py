import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_architecture import CONFIG, architecture_errors

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_checked_architecture_boundaries_pass():
    assert architecture_errors() == ()


def test_top_level_package_layout_is_explicitly_frozen():
    allowed = set(CONFIG["package_layout"]["allowed_top_level_directories"])
    assert {"core", "compiler", "runtime", "simulation", "providers"} <= allowed
    assert {"ops", "numerics", "compilation", "_compiler"}.isdisjoint(allowed)


def test_stable_root_import_is_lazy_and_does_not_initialize_runtimes():
    code = """
import sys
import flagquantum
forbidden = ('flagquantum.runtime_stack', 'flagquantum.deployment', 'flagquantum.drawer', 'jax', 'jaxlib')
assert not any(name.startswith(forbidden) for name in sys.modules), sorted(sys.modules)
# The public surface is snapshot-governed in public_api_v1.json. Keep this
# guard as a coarse emergency ceiling without contradicting that snapshot.
assert len(flagquantum.__all__) <= 64
assert 'flagquantum.api' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_stable_api_preserves_behavior_without_v01_device_surface():
    import flagquantum as fq

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    assert circuit.state().shape[-1] == 4
    assert not hasattr(fq, "DistributedQuantumDevice")


def test_dependency_graph_and_architecture_policy_are_checked_in():
    graph = (ROOT / "docs" / "architecture" / "ARCHITECTURE_DEPENDENCIES.md").read_text(
        encoding="utf-8"
    )
    policy = (ROOT / "architecture.toml").read_text(encoding="utf-8")

    assert "flowchart TD" in graph
    assert "Executors emit backend-neutral" in graph
    assert "legacy_exceptions" not in policy
    assert "[legacy_subsystems.v01_dtensor_device]" not in policy
