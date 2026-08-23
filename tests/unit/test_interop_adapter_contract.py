from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.interop import (
    DEFAULT_INTEROP_REGISTRY,
    INTEROP_API_VERSION,
    InteropAdapter,
    InteropAdapterSpec,
    InteropConformanceResult,
    InteropConversionError,
    InteropConversionIssue,
    InteropConversionReport,
    InteropDependencyError,
    InteropExportResult,
    InteropImportResult,
    InteropRegistry,
    InteropRegistryError,
    InteropRoundTripCase,
    available_adapters,
    get_adapter,
    run_adapter_conformance,
    semantic_fingerprint,
)
from flagquantum.interop.qiskit import (
    QISKIT_ADAPTER,
    QiskitConversionError,
    QiskitConversionIssue,
    QiskitConversionReport,
    QiskitDependencyError,
    QiskitExportResult,
    QiskitImportResult,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _dependency_policy() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    return tomllib.loads((ROOT / "dependency-policy.toml").read_text(encoding="utf-8"))


def test_framework_neutral_report_is_machine_readable_and_fail_closed() -> None:
    blocker = InteropConversionIssue("unsupported", "cannot convert", "error", 2, "x")
    report = InteropConversionReport("example", "from_example", "1.2", (blocker,))

    assert not report.lossless
    assert report.blockers == (blocker,)
    assert report.to_dict() == {
        "schema": "flagquantum_interop_conversion_report_v1",
        "adapter": "example",
        "direction": "from_example",
        "framework_version": "1.2",
        "lossless": False,
        "issues": [blocker.to_dict()],
    }

    with pytest.raises(ValueError, match="severity"):
        InteropConversionIssue("invalid", "invalid", "fatal")  # type: ignore[arg-type]


def test_default_registry_is_immutable_lazy_and_qiskit_only() -> None:
    assert available_adapters() == ("qiskit",)
    assert DEFAULT_INTEROP_REGISTRY.names == ("qiskit",)
    with pytest.raises(TypeError):
        DEFAULT_INTEROP_REGISTRY.specs["other"] = DEFAULT_INTEROP_REGISTRY.spec(
            "qiskit"
        )

    adapter = get_adapter("qiskit")
    assert adapter is QISKIT_ADAPTER
    assert isinstance(adapter, InteropAdapter)
    assert (adapter.name, adapter.api_version, adapter.dependency_extra) == (
        "qiskit",
        INTEROP_API_VERSION,
        "qiskit",
    )
    payload = DEFAULT_INTEROP_REGISTRY.to_dict()
    assert payload["schema"] == "flagquantum_interop_registry_v1"
    assert payload["adapters"] == [DEFAULT_INTEROP_REGISTRY.spec("qiskit").to_dict()]


def test_registered_adapter_extras_are_dependency_governed() -> None:
    policy = _dependency_policy()
    interop_extras = set(policy["classes"]["interop"])

    for spec in DEFAULT_INTEROP_REGISTRY.specs.values():
        assert spec.dependency_extra in policy["extras"]
        assert spec.dependency_extra in interop_extras
        assert spec.module.startswith(f"flagquantum.interop.{spec.name}")


def test_registry_additions_return_new_registry_and_reject_ambiguity() -> None:
    spec = InteropAdapterSpec(
        name="example",
        module="example.interop",
        attribute="ADAPTER",
        dependency_extra="example",
    )
    expanded = InteropRegistry().with_spec(spec)

    assert expanded.names == ("example",)
    assert InteropRegistry().names == ()
    with pytest.raises(InteropRegistryError, match="already registered"):
        expanded.with_spec(spec)
    with pytest.raises(InteropRegistryError, match="unknown interop adapter"):
        expanded.load("missing")

    incompatible = InteropAdapterSpec(
        name="future",
        module="future.interop",
        attribute="ADAPTER",
        dependency_extra="future",
        api_version="2.0",
    )
    with pytest.raises(InteropRegistryError, match="targets interop API 2.0"):
        InteropRegistry({"future": incompatible}).load("future")

    missing = InteropAdapterSpec(
        name="missing",
        module="flagquantum.interop.missing_adapter",
        attribute="ADAPTER",
        dependency_extra="missing",
    )
    with pytest.raises(InteropRegistryError, match="implementation is unavailable"):
        InteropRegistry({"missing": missing}).load("missing")


def test_importing_and_resolving_adapter_does_not_import_qiskit() -> None:
    code = """
import sys
from flagquantum.interop import available_adapters, get_adapter
assert available_adapters() == ('qiskit',)
assert get_adapter('qiskit').name == 'qiskit'
loaded = [
    name for name in sys.modules
    if name == 'qiskit' or name.startswith('qiskit.')
    or name == 'qiskit_aer' or name.startswith('qiskit_aer.')
]
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_qiskit_types_are_framework_neutral_contracts_with_legacy_payload() -> None:
    issue = QiskitConversionIssue("unsupported", "cannot convert", "error")
    report = QiskitConversionReport("from_qiskit", "2.5", (issue,))
    ir = fq.Circuit(1).x(0).to_ir()
    imported = QiskitImportResult(ir, report)
    exported = QiskitExportResult("circuit", report)

    assert isinstance(issue, InteropConversionIssue)
    assert isinstance(report, InteropConversionReport)
    assert isinstance(imported, InteropImportResult)
    assert isinstance(exported, InteropExportResult)
    assert exported.circuit == exported.artifact == "circuit"
    assert report.to_dict()["schema"] == "flagquantum_qiskit_conversion_report_v1"
    assert issubclass(QiskitDependencyError, InteropDependencyError)
    assert issubclass(QiskitConversionError, InteropConversionError)


def test_generic_result_and_error_preserve_typed_diagnostics() -> None:
    report = InteropConversionReport("example", "to_example", None)
    ir = fq.Circuit(1).to_ir()

    assert InteropImportResult(ir, report).ir is ir
    assert InteropExportResult("artifact", report).artifact == "artifact"
    error = InteropConversionError("blocked", report)
    assert error.report is report


def test_conformance_kit_is_available_without_external_frameworks() -> None:
    assert InteropConformanceResult.__module__ == "flagquantum.interop.conformance"
    assert InteropRoundTripCase.__module__ == "flagquantum.interop.conformance"
    assert callable(run_adapter_conformance)
    assert callable(semantic_fingerprint)
