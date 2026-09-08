import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_architecture import (
    CONFIG,
    _imports_jax_executor,
    architecture_errors,
    eager_import_cycles,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_checked_architecture_boundaries_pass():
    assert architecture_errors() == ()


def test_eager_import_cycle_is_rejected_but_delayed_references_are_allowed(tmp_path):
    package = tmp_path / "flagquantum"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "first.py").write_text(
        "from flagquantum import second\n", encoding="utf-8"
    )
    (package / "second.py").write_text(
        "from flagquantum import first\n", encoding="utf-8"
    )
    (package / "delayed.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from flagquantum import first\n"
        "def load():\n"
        "    from flagquantum import first\n",
        encoding="utf-8",
    )

    assert eager_import_cycles(package) == (
        ("flagquantum.first", "flagquantum.second"),
    )


@pytest.mark.parametrize(
    "source",
    (
        "from ..jax import plan\n",
        "from flagquantum.runtime.executors.jax import plan\n",
        "from flagquantum.runtime.executors import jax\n",
    ),
)
def test_non_jax_backend_import_check_resolves_relative_and_absolute_imports(
    monkeypatch, tmp_path, source
):
    path = tmp_path / "flagquantum/runtime/executors/mps/example.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")

    import tools.check_architecture as check_architecture

    monkeypatch.setattr(check_architecture, "ROOT", tmp_path)
    assert _imports_jax_executor(path)


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
