from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.extensions as fqx

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "examples/extensions/reference_extensions.py"


def _load_reference_extensions():
    spec = importlib.util.spec_from_file_location(
        "external_flagquantum_reference_extensions", REFERENCE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_third_party_backend_uses_only_the_public_extension_namespace() -> None:
    tree = ast.parse(REFERENCE_PATH.read_text(encoding="utf-8"))
    flagquantum_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("flagquantum")
    }

    assert flagquantum_imports == {"flagquantum.extensions"}

    reference = _load_reference_extensions()
    backend = reference.ReferenceTorchBackend()
    assert isinstance(backend, fqx.ExecutionBackendExtension)
    report = fqx.run_backend_conformance(backend)
    assert report.extension == "reference_torch"
    assert {"dtype_device", "gradients", "cleanup"} <= set(report.checks)


def test_extension_protocol_is_layered_and_does_not_expand_root_api() -> None:
    from flagquantum.errors import CapabilityError, ExecutionError, FlagQuantumError

    assert fqx.SDK_API_VERSION == "1.0"
    assert set(fqx.__all__).isdisjoint(fq.__all__)
    assert issubclass(fqx.ExtensionError, FlagQuantumError)
    assert issubclass(fqx.ExtensionCompatibilityError, CapabilityError)
    assert issubclass(fqx.ExtensionLifecycleError, ExecutionError)


def test_execution_plan_and_deployment_package_have_distinct_owners() -> None:
    from flagquantum.deployment import DeploymentPackage

    plan = fq.plan(fq.Circuit(1).h(0))

    assert not isinstance(plan, DeploymentPackage)
    assert not hasattr(plan, "submit")
    assert not hasattr(plan, "provider")
    assert not hasattr(DeploymentPackage, "from_json")
