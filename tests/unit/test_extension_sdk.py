"""Extension SDK contracts."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.ecosystem.extensions.sdk as extension_sdk
from flagquantum.compiler.pass_manager import OPTIMIZATION_PIPELINE
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
    run_compiler_pass_conformance,
    run_provider_conformance,
)
from flagquantum.ecosystem.extensions.pass_admission import (
    PassAdmissionError,
    check_pass_admission,
)

pytestmark = pytest.mark.unit

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

PASS_REFERENCE_PATH = ROOT / "examples/extensions/reference_pass_extension.py"
PASS_SPEC = importlib.util.spec_from_file_location(
    "reference_pass_extension", PASS_REFERENCE_PATH
)
assert PASS_SPEC and PASS_SPEC.loader
PASS_REFERENCE = importlib.util.module_from_spec(PASS_SPEC)
PASS_SPEC.loader.exec_module(PASS_REFERENCE)


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


def test_reference_pass_certifies_the_whole_pass_contract():
    root_before = set(fq.__all__)
    extension = PASS_REFERENCE.ReferenceSwapPass()

    report = run_compiler_pass_conformance(extension)

    assert report.extension == "reference_swap_pass"
    assert report.checks == (
        "manifest_serialization",
        "admissible",
        "host_named",
        "circuit_ir",
        "input_immutable",
        "deterministic",
        "cleanup",
    )
    assert not extension.active
    assert set(fq.__all__) == root_before


def test_the_reference_pass_rewrites_the_swap_it_names_and_carries_the_rest():
    source = fq.CircuitIR(
        2,
        (fq.Instruction("h", (0,)), fq.Instruction("swap", (0, 1))),
        metadata={"origin": "example"},
    )
    extension = PASS_REFERENCE.ReferenceSwapPass()
    extension.start(ExtensionConfig())

    transformed = extension.transform(source)
    extension.close()

    assert [item.name for item in transformed.instructions] == ["h", "cx", "cx", "cx"]
    assert [item.wires for item in transformed.instructions[1:]] == [
        (0, 1),
        (1, 0),
        (0, 1),
    ]
    assert transformed.instructions[0] is source.instructions[0]
    assert transformed.metadata == {"origin": "example"}
    assert [item.name for item in source.instructions] == ["h", "swap"]


def test_the_pass_runner_registers_the_host_name_rather_than_the_declared_one():
    extension = PASS_REFERENCE.ReferenceSwapPass()

    registered = check_pass_admission(extension)

    assert registered == "extension.reference_swap_pass.decompose_swap"
    assert extension.pass_name == "decompose_swap"


def _pass_whose_transform(transform):
    """The reference pass with only its transform replaced, so it stays admissible."""

    class _Scripted(PASS_REFERENCE.ReferenceSwapPass):
        def transform(self, program):
            return transform(program)

    return _Scripted()


def test_the_pass_runner_fails_when_the_crossing_stops_prefixing(monkeypatch):
    # The runner is a regression detector for the crossing, so the regression it
    # exists to catch is simulated rather than waited for.
    monkeypatch.setattr(
        "flagquantum.ecosystem.extensions.pass_admission.check_pass_admission",
        lambda extension: "remove_identity_gates",
    )

    with pytest.raises(AssertionError, match="outside the host prefix"):
        run_compiler_pass_conformance(PASS_REFERENCE.ReferenceSwapPass())


def test_the_pass_runner_refuses_a_declaration_admission_would_refuse():
    class _NoIR(PASS_REFERENCE.ReferenceSwapPass):
        manifest = ExtensionManifest(
            name="no_ir_pass",
            version="1.0.0",
            kind="compiler_pass",
            capabilities=frozenset(),
        )

    with pytest.raises(PassAdmissionError, match="circuit_ir"):
        run_compiler_pass_conformance(_NoIR())


def test_a_pass_that_mutates_the_program_it_was_given_is_refused():
    def mutating(program):
        object.__setattr__(program, "instructions", ())
        return program

    with pytest.raises(AssertionError, match="mutated its input"):
        run_compiler_pass_conformance(_pass_whose_transform(mutating))


def test_a_pass_that_disagrees_with_itself_is_refused():
    applications = []

    def drifting(program):
        applications.append(1)
        return fq.CircuitIR(2, (fq.Instruction("h", (0,)),) * len(applications))

    with pytest.raises(AssertionError, match="not deterministic"):
        run_compiler_pass_conformance(_pass_whose_transform(drifting))


def test_a_pass_that_returns_something_other_than_ir_is_refused():
    def not_ir(program):
        return [instruction.name for instruction in program.instructions]

    with pytest.raises(AssertionError, match="must return CircuitIR"):
        run_compiler_pass_conformance(_pass_whose_transform(not_ir))


def test_a_pass_that_asks_for_a_built_in_name_still_cannot_take_it():
    class _Rival(PASS_REFERENCE.ReferenceSwapPass):
        pass_name = "remove_identity_gates"

    rival = _Rival()

    assert check_pass_admission(rival) == (
        "extension.reference_swap_pass.remove_identity_gates"
    )
    # The bare name the pass asked for is still the pipeline's own, which is why
    # admission has nothing to refuse here: the prefix makes the request harmless.
    assert "remove_identity_gates" in OPTIMIZATION_PIPELINE
    assert "cleanup" in run_compiler_pass_conformance(rival).checks


def test_a_pass_that_leaks_its_lifecycle_state_is_refused():
    class _Leaking(PASS_REFERENCE.ReferenceSwapPass):
        def close(self):
            pass

    with pytest.raises(AssertionError, match="leaked active lifecycle state"):
        run_compiler_pass_conformance(_Leaking())


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
