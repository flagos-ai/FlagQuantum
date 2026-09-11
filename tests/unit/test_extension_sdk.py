"""Extension SDK contracts."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.ecosystem.extensions.sdk as extension_sdk
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    ExtensionCompatibilityError,
    ExtensionConfig,
    ExtensionLifecycleError,
    ExtensionManifest,
    ExtensionRegistry,
    compile_with_extension,
    discover_extensions,
    extension_scope,
)
from flagquantum.ecosystem.extensions.conformance import (
    run_backend_conformance,
    run_compiler_conformance,
    run_provider_conformance,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "examples/extensions/reference_extensions.py"
SPEC = importlib.util.spec_from_file_location("reference_extensions", REFERENCE_PATH)
assert SPEC and SPEC.loader
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)

COMPILER_REFERENCE_PATH = ROOT / "examples/extensions/reference_compiler_extension.py"
COMPILER_SPEC = importlib.util.spec_from_file_location(
    "reference_compiler_extension", COMPILER_REFERENCE_PATH
)
assert COMPILER_SPEC and COMPILER_SPEC.loader
COMPILER_REFERENCE = importlib.util.module_from_spec(COMPILER_SPEC)
COMPILER_SPEC.loader.exec_module(COMPILER_REFERENCE)


def test_reference_backend_and_provider_pass_conformance_without_root_mutation():
    root_before = set(fq.__all__)
    backend_report = run_backend_conformance(REFERENCE.ReferenceTorchBackend())
    provider_report = run_provider_conformance(REFERENCE.ReferenceProvider())
    assert "gradients" in backend_report.checks
    assert "cleanup" in provider_report.checks
    assert set(fq.__all__) == root_before
    assert "ReferenceTorchBackend" not in dir(fq)


def test_reference_compiler_preserves_ir_ownership_and_determinism():
    report = run_compiler_conformance(COMPILER_REFERENCE.ReferenceCircuitCompiler())

    assert report.checks == (
        "manifest_serialization",
        "circuit_ir",
        "input_immutable",
        "deterministic",
        "cleanup",
    )


def test_compiler_discovery_loads_only_requested_kind(monkeypatch):
    class EntryPoint:
        def __init__(self, name, factory):
            self.name = name
            self._factory = factory

        def load(self):
            return self._factory

    def entry_points(*, group):
        assert group == "flagquantum.extensions"
        return (
            EntryPoint("provider.must_not_load", lambda: 1 / 0),
            EntryPoint(
                "compiler.reference_compiler",
                COMPILER_REFERENCE.ReferenceCircuitCompiler,
            ),
        )

    monkeypatch.setattr(extension_sdk.metadata, "entry_points", entry_points)

    registry = discover_extensions("compiler")

    assert tuple(registry.entries) == (("compiler", "reference_compiler"),)


def test_compile_with_extension_is_an_explicit_complete_user_journey(monkeypatch):
    compiler = COMPILER_REFERENCE.ReferenceCircuitCompiler()

    class EntryPoint:
        name = "compiler.reference_compiler"

        @staticmethod
        def load():
            return lambda: compiler

    monkeypatch.setattr(
        extension_sdk.metadata,
        "entry_points",
        lambda *, group: (EntryPoint(),),
    )
    source = fq.Circuit(2).h(0).cx(0, 1).to_ir()
    target = {"basis_gates": ("h", "cx")}

    compiled = compile_with_extension(
        source,
        extension="reference_compiler",
        target=target,
    )

    assert compiled.metadata["reference_compiler_target"] == target
    assert compiled.instructions == source.instructions
    assert not compiler.active


def test_discovery_rejects_entry_point_manifest_mismatch(monkeypatch):
    class EntryPoint:
        name = "compiler.wrong_name"

        @staticmethod
        def load():
            return COMPILER_REFERENCE.ReferenceCircuitCompiler

    monkeypatch.setattr(
        extension_sdk.metadata,
        "entry_points",
        lambda *, group: (EntryPoint(),),
    )

    with pytest.raises(ExtensionCompatibilityError, match="does not match manifest"):
        discover_extensions("compiler")


def test_discovery_contains_entry_point_load_failure(monkeypatch):
    class EntryPoint:
        name = "compiler.broken"

        @staticmethod
        def load():
            raise ImportError("missing compiler dependency")

    monkeypatch.setattr(
        extension_sdk.metadata,
        "entry_points",
        lambda *, group: (EntryPoint(),),
    )

    with pytest.raises(
        ExtensionCompatibilityError,
        match="compiler.broken.*ImportError.*missing compiler dependency",
    ):
        discover_extensions("compiler")


def test_registration_is_immutable_task_local_and_serializable():
    backend = REFERENCE.ReferenceTorchBackend()
    empty = ExtensionRegistry()
    with extension_scope([backend]) as scoped:
        assert not empty.entries
        assert ("backend", "reference_torch") in scoped.entries
        json.dumps(backend.manifest.to_dict(), sort_keys=True)
    assert not empty.entries


def test_version_and_capability_mismatches_fail_early_with_diagnostics():
    backend = REFERENCE.ReferenceTorchBackend()
    backend.manifest = ExtensionManifest(
        "future_backend", "1.0", "backend", api_version="99.0"
    )
    with pytest.raises(ExtensionCompatibilityError, match="99.0.*1.0.*upgrade"):
        ExtensionRegistry().with_extension(backend)

    backend = REFERENCE.ReferenceTorchBackend()
    registry = ExtensionRegistry().with_extension(backend)
    with pytest.raises(ExtensionCompatibilityError, match="missing capabilities"):
        registry.negotiate(
            "backend",
            "reference_torch",
            CapabilityRequest(required=frozenset({"distributed_sharding"})),
        )


@pytest.mark.parametrize("key", ["token", "api_key", "db-password", "credential"])
def test_credentials_are_rejected_from_serializable_config(key):
    with pytest.raises(ValueError, match="host-owned credential resolver"):
        ExtensionConfig({key: "must-not-leak"})


def test_failed_extension_is_contained_and_cleanup_is_attempted():
    class BrokenBackend(REFERENCE.ReferenceTorchBackend):
        manifest = ExtensionManifest(
            "broken", "1.0", "backend", capabilities=frozenset({"cpu"})
        )

        def execute(self, program, parameters=None):
            raise ValueError("extension-local failure")

    broken = BrokenBackend()
    handle = (
        ExtensionRegistry()
        .with_extension(broken)
        .negotiate("backend", "broken", CapabilityRequest(required=frozenset({"cpu"})))
    )
    handle.start(ExtensionConfig())
    with pytest.raises(ExtensionLifecycleError, match="ValueError"):
        handle.invoke("execute", None, None)
    handle.close()
    assert not broken.active


def test_all_protocol_kinds_are_declared_without_import_side_effects():
    before = set(sys.modules)
    import flagquantum.ecosystem.extensions as extensions

    expected = {
        "ExecutionBackendExtension",
        "KernelExtension",
        "OperatorExtension",
        "CompilerPassExtension",
        "CompilerExtension",
        "DeviceExtension",
        "ProviderExtension",
        "MeasurementCollectorExtension",
        "PlannerExtension",
    }
    assert expected <= set(extensions.__all__)
    assert not ({"jax", "jaxlib"} - before) & set(sys.modules)


def test_negotiation_exceptions_are_contained():
    class BadNegotiator(REFERENCE.ReferenceProvider):
        manifest = ExtensionManifest("bad_negotiate", "1", "provider")

        def negotiate(self, request):
            raise RuntimeError("bad capability probe")

    registry = ExtensionRegistry().with_extension(BadNegotiator())
    with pytest.raises(ExtensionCompatibilityError, match="negotiation failed"):
        registry.negotiate("provider", "bad_negotiate", CapabilityRequest())
