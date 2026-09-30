"""Backend execution admission: one crossing from the extension SDK to Runtime.

These tests demonstrate the boundary rather than the mechanism. A third-party
backend is admitted, `fq.run` reaches it without any FlagQuantum source change,
and every failure mode is rejected before a kernel starts.
"""

import types
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

import flagquantum as fq
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionCompatibilityError,
    ExtensionConfig,
    ExtensionManifest,
)
from flagquantum.ecosystem.extensions.admission import (
    BackendAdmissionError,
    admit_backend_extension,
    check_backend_admission,
    withdraw_backend,
)
from flagquantum.errors import CapabilityError, ExecutionError, FlagQuantumError
from flagquantum.runtime.backend_registry import (
    BackendCapabilities,
    get_backend_capabilities,
    list_backends,
    refresh_backend_registry,
    register_backend,
    resolve_backend_executor,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "examples/extensions/reference_backend_extension.py"
# Compiled from source rather than imported through the bytecode cache: a cached
# .pyc survives a same-size edit made within the same second, which would let
# this module assert against code that is no longer in the tree.
REFERENCE = types.ModuleType("reference_backend_extension")
exec(
    compile(REFERENCE_PATH.read_text(encoding="utf-8"), str(REFERENCE_PATH), "exec"),
    REFERENCE.__dict__,
)


class _ScriptedBackend:
    """A minimal admitted backend whose result is chosen by the test."""

    manifest = ExtensionManifest(
        name="scripted_backend",
        version="1.0.0",
        kind="backend",
        capabilities=frozenset({"complex64", "cpu"}),
    )
    backend_capabilities: ClassVar[dict[str, Any]] = {
        "tensor_backend": "scripted",
        "devices": ("cpu",),
        "dtypes": ("complex64",),
        "supports_autograd": True,
        "supports_distributed": False,
        "supports_statevector": True,
        "supports_density_matrix": False,
        "supports_mps": False,
    }

    def __init__(self, result=None, error=None):
        self.active = False
        self.result = result
        self.error = error
        self.options = None

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        return CapabilityResponse(True, self.manifest.capabilities, ())

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def close(self) -> None:
        self.active = False

    def execute(self, program, *, options):
        self.options = options
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return torch.zeros((1, 2**program.n_wires), dtype=torch.complex64)


@pytest.fixture
def admitted():
    """Admit a backend for one test and restore the registry exactly."""

    admitted_names = []

    def _admit(extension, **kwargs):
        handle, capabilities = admit_backend_extension(extension, **kwargs)
        admitted_names.append(capabilities.name)
        return handle, capabilities

    yield _admit
    for name in admitted_names:
        withdraw_backend(name)
    refresh_backend_registry()


def _bell() -> "fq.Circuit":
    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _variant(
    extension_name: str, kind: str = "backend", **declaration: Any
) -> _ScriptedBackend:
    """A ``_ScriptedBackend`` that differs only in the declaration under test.

    ``extension_name`` is deliberately not called ``name``: ``name`` is one of
    the host-owned keys a test has to be able to pass through ``declaration``.
    """

    return type(
        extension_name,
        (_ScriptedBackend,),
        {
            "manifest": ExtensionManifest(
                name=extension_name,
                version="1.0.0",
                kind=kind,
                capabilities=frozenset({"cpu"}),
            ),
            "backend_capabilities": {
                **_ScriptedBackend.backend_capabilities,
                **declaration,
            },
        },
    )()


def _gate_circuit(name: str, wires: tuple[int, ...], **params: float) -> "fq.Circuit":
    circuit = fq.Circuit(3)
    circuit.h(0)
    getattr(circuit, name)(*wires, **params)
    circuit.ry(2, theta=0.3)
    return circuit


# --------------------------------------------------------------------------
# Scenario: a third-party backend executes a circuit without a source change.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "wires", "params"),
    [
        ("h", (1,), {}),
        ("x", (1,), {}),
        ("rz", (1,), {"theta": 0.7}),
        ("rx", (1,), {"theta": -1.1}),
        ("ry", (1,), {"theta": 2.2}),
        ("cx", (0, 1), {}),
        ("cx", (1, 2), {}),
        ("cx", (2, 0), {}),
    ],
)
def test_the_reference_backend_agrees_with_the_builtin_engine(
    admitted, name, wires, params
):
    """Every declared gate matches the built-in engine, bit order included.

    The controlled cases use three different control/target pairs, one of them
    with control above target: a wrong amplitude bit order still passes a single
    pairing, so the wire identity is the part that has to be pinned.
    """

    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    try:
        circuit = _gate_circuit(name, wires, **params)
        result = fq.run(
            circuit,
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )
        reference = fq.run(circuit)
        assert torch.allclose(result.state, reference.state, atol=1e-5)
    finally:
        handle.close()


def test_admitted_extension_backend_executes_a_circuit_through_fq_run(admitted):
    backend = REFERENCE.ReferenceStatevectorBackend()
    handle, capabilities = admitted(backend)
    try:
        result = fq.run(
            _bell(),
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )
        reference = fq.run(_bell())
        assert torch.allclose(result.state, reference.state, atol=1e-5)
        assert backend.executions == 1
    finally:
        handle.close()


def test_an_unsupported_gate_fails_closed_through_the_admitted_route(admitted):
    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    try:
        circuit = fq.Circuit(2)
        circuit.h(0)
        circuit.cz(0, 1)
        with pytest.raises(ExecutionError, match="unsupported gates: cz"):
            fq.run(
                circuit,
                options=fq.ExecutionOptions(
                    backend=capabilities.name, mode="statevector"
                ),
            )
    finally:
        handle.close()


def test_a_closed_extension_refuses_to_execute(admitted):
    """Closing the extension is observable at execution, never a silent success.

    The route stays registered because withdrawal is a separate, explicit act:
    a caller that only closes the handle gets a refusal rather than a result
    produced by a lifecycle that has already ended.
    """

    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    handle.close()
    assert resolve_backend_executor(capabilities.name) is not None
    with pytest.raises(ExecutionError, match="backend is closed"):
        fq.run(
            _bell(),
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )


def test_a_withdrawn_route_does_not_execute(admitted):
    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    handle.close()
    assert withdraw_backend(capabilities.name) is True
    with pytest.raises(FlagQuantumError, match=capabilities.name):
        fq.run(
            _bell(),
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )


def test_admitted_backend_result_names_its_own_route_and_semantics(admitted):
    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    try:
        result = fq.run(
            _bell(),
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )
        assert result.runtime["backend"] == capabilities.name
        assert result.runtime["execution_path"] == "admitted_extension_backend"
        assert result.runtime["distribution_semantics"] == "single_device_fast_path"
        assert result.runtime["scalability_claim_allowed"] is False
        # The host does not claim a third-party backend's output as its own local
        # engine, so the facts the local statevector path stamps are absent.
        assert "platform_provider" not in result.runtime
        assert "cpu_fallback_used" not in result.provenance
    finally:
        handle.close()


def test_admitted_backend_receives_the_planned_request(admitted):
    backend = _ScriptedBackend()
    handle, capabilities = admitted(backend)
    try:
        fq.run(
            _bell(),
            options=fq.ExecutionOptions(
                backend=capabilities.name, mode="statevector", batch_size=1
            ),
        )
        assert backend.options["backend"] == capabilities.name
        assert backend.options["mode"] == "statevector"
        assert backend.options["dtype"] == "complex64"
        assert backend.options["world_size"] == 1
    finally:
        handle.close()


def test_admitted_backend_executes_a_plan_that_was_already_returned(admitted):
    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    try:
        plan = fq.plan(
            _bell(),
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )
        assert fq.run(plan).state.shape == (1, 4)
    finally:
        handle.close()


# --------------------------------------------------------------------------
# The built-in fast path stays free of admission work.
# --------------------------------------------------------------------------


def test_builtin_backends_never_resolve_an_extension_route():
    for name in ("pytorch", "torch"):
        assert resolve_backend_executor(name) is None
        assert name in list_backends()


def test_builtin_and_auto_backends_execute_without_any_admission():
    refresh_backend_registry()
    for backend in ("auto", "pytorch"):
        result = fq.run(
            _bell(), options=fq.ExecutionOptions(backend=backend, mode="statevector")
        )
        assert result.runtime["execution_path"] == "local_statevector"


def test_the_builtin_path_consults_no_admitted_route_state(monkeypatch):
    """The built-in fast path pays nothing for the admission boundary.

    Only the two functions that decide whether an *extension* may execute are
    counted. Looking a record up is not admission work; asking whether a record
    has an admitted route, or whether it blocks this plan, is.
    """

    calls = []
    registry = fq.runtime.backend_registry
    for name in ("resolve_backend_executor",):
        original = getattr(registry, name)
        monkeypatch.setattr(
            registry,
            name,
            lambda *a, _n=name, _o=original, **k: (
                calls.append(_n),
                _o(*a, **k),
            )[1],
        )
    original_blockers = registry.BackendCapabilities.admission_blockers
    monkeypatch.setattr(
        registry.BackendCapabilities,
        "admission_blockers",
        lambda self, **k: (
            calls.append("admission_blockers"),
            original_blockers(self, **k),
        )[1],
    )

    refresh_backend_registry()
    result = fq.run(
        _bell(), options=fq.ExecutionOptions(backend="pytorch", mode="statevector")
    )
    assert result.runtime["execution_path"] == "local_statevector"
    assert calls == []


def test_a_builtin_name_cannot_be_taken_over_by_an_executor():
    with pytest.raises(ValueError, match="built in"):
        BackendCapabilities(
            name="pytorch",
            tensor_backend="torch",
            devices=("cpu",),
            dtypes=("complex64",),
            supports_autograd=True,
            supports_distributed=False,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_mps=False,
            executor=_ScriptedBackend(),
        )


def test_refreshing_the_registry_preserves_an_admitted_route(admitted):
    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    try:
        refresh_backend_registry()
        assert capabilities.name in list_backends()
        assert resolve_backend_executor(capabilities.name) is not None
        assert fq.run(
            _bell(),
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        ).state.shape == (1, 4)
    finally:
        handle.close()


# --------------------------------------------------------------------------
# Fail closed: declaration, capability, mode, gradients, distribution, failure.
# --------------------------------------------------------------------------


def test_a_registered_record_without_a_route_is_not_executable():
    register_backend(
        BackendCapabilities(
            name="declared_only",
            tensor_backend="torch",
            devices=("cpu",),
            dtypes=("complex64",),
            supports_autograd=True,
            supports_distributed=False,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_mps=False,
        )
    )
    try:
        with pytest.raises(CapabilityError, match="no execution route"):
            fq.plan(_bell(), options=fq.ExecutionOptions(backend="declared_only"))
    finally:
        refresh_backend_registry()


def test_an_unadmitted_backend_name_is_rejected_at_planning_time():
    with pytest.raises(CapabilityError, match="is not available"):
        fq.plan(_bell(), options=fq.ExecutionOptions(backend="absent_backend"))


def test_a_mode_the_backend_does_not_support_fails_before_any_kernel(admitted):
    backend = _ScriptedBackend()
    handle, capabilities = admitted(backend)
    try:
        with pytest.raises(CapabilityError, match="mode 'mps' is not supported"):
            fq.plan(
                _bell(),
                options=fq.ExecutionOptions(backend=capabilities.name, mode="mps"),
            )
        assert backend.options is None
    finally:
        handle.close()


def test_declared_blockers_are_reported_verbatim(admitted):
    extension = _variant("blocked_backend", blockers=("reason one", "reason two"))
    handle, capabilities = admitted(extension)
    try:
        with pytest.raises(CapabilityError) as excinfo:
            fq.plan(_bell(), options=fq.ExecutionOptions(backend=capabilities.name))
        assert "reason one" in str(excinfo.value)
        assert "reason two" in str(excinfo.value)
    finally:
        handle.close()


def test_a_backend_without_autograd_cannot_serve_a_gradient_request(admitted):
    extension = _variant("no_grad_backend", supports_autograd=False)
    handle, capabilities = admitted(extension)
    try:
        with pytest.raises(CapabilityError, match="autograd is not supported"):
            fq.plan(
                _bell(),
                options=fq.ExecutionOptions(
                    backend=capabilities.name, require_gradients=True
                ),
            )
    finally:
        handle.close()


def test_a_scalability_claim_requires_sharded_distribution_semantics():
    with pytest.raises(ValueError, match="sharded_across_ranks"):
        BackendCapabilities(
            name="unsupported_claim",
            tensor_backend="torch",
            devices=("cpu",),
            dtypes=("complex64",),
            supports_autograd=True,
            supports_distributed=True,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_mps=False,
            distribution_semantics="replicated_per_rank",
            scalability_claim_allowed=True,
        )


def test_an_unknown_distribution_semantics_is_rejected():
    with pytest.raises(ValueError, match="unknown distribution semantics"):
        BackendCapabilities(
            name="odd_semantics",
            tensor_backend="torch",
            devices=("cpu",),
            dtypes=("complex64",),
            supports_autograd=True,
            supports_distributed=False,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_mps=False,
            distribution_semantics="probably_sharded",
        )


def test_a_backend_failure_is_reported_and_never_substituted(admitted):
    backend = _ScriptedBackend(error=RuntimeError("device exploded"))
    handle, capabilities = admitted(backend)
    try:
        with pytest.raises(ExecutionError, match="device exploded"):
            fq.run(
                _bell(),
                options=fq.ExecutionOptions(
                    backend=capabilities.name, mode="statevector"
                ),
            )
    finally:
        handle.close()


def test_a_flagquantum_error_from_a_backend_is_not_rewrapped(admitted):
    backend = _ScriptedBackend(error=CapabilityError("static refusal"))
    handle, capabilities = admitted(backend)
    try:
        with pytest.raises(CapabilityError, match="static refusal"):
            fq.run(
                _bell(),
                options=fq.ExecutionOptions(
                    backend=capabilities.name, mode="statevector"
                ),
            )
    finally:
        handle.close()


def test_a_malformed_backend_result_is_contained(admitted):
    backend = _ScriptedBackend(result="not a statevector")
    handle, capabilities = admitted(backend)
    try:
        result = fq.run(
            _bell(),
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )
        # The adapter records what it could not interpret instead of pretending
        # the native object was a statevector.
        assert result.runtime["execution_path"] == "admitted_extension_backend"
    finally:
        handle.close()


def test_withdrawing_a_route_fails_closed_instead_of_falling_back(admitted):
    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    plan = fq.plan(
        _bell(),
        options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
    )
    handle.close()
    assert withdraw_backend(capabilities.name) is True
    with pytest.raises(FlagQuantumError, match=capabilities.name):
        fq.run(plan)
    assert capabilities.name not in list_backends()


def test_withdrawing_an_unknown_backend_reports_no_removal():
    assert withdraw_backend("never_admitted_backend") is False


def test_a_builtin_route_cannot_be_withdrawn():
    for name in ("pytorch", "torch"):
        with pytest.raises(ValueError, match="built in"):
            withdraw_backend(name)
    assert resolve_backend_executor("pytorch") is None
    assert "pytorch" in list_backends()


# --------------------------------------------------------------------------
# Declaration validation: host-owned keys, unknown keys, completeness, kind.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["name", "executor", "accelerators"])
def test_an_extension_may_not_declare_a_host_owned_key(key):
    extension = _variant(f"host_owned_{key}", **{key: "supplied by the extension"})
    with pytest.raises(BackendAdmissionError, match="host-owned"):
        check_backend_admission(extension)


def test_an_unknown_capability_field_is_rejected():
    extension = _variant("unknown_field_backend", supports_gradients=True)
    with pytest.raises(BackendAdmissionError, match="unknown capability fields"):
        check_backend_admission(extension)


def test_a_missing_declaration_is_rejected():
    class Silent:
        manifest = ExtensionManifest(
            name="silent_backend",
            version="1.0.0",
            kind="backend",
            capabilities=frozenset({"cpu"}),
        )

        def negotiate(self, request):
            return CapabilityResponse(True, frozenset({"cpu"}), ())

        def start(self, config):
            pass

        def close(self):
            pass

        def execute(self, program, *, options):
            raise AssertionError("must not execute")

    with pytest.raises(BackendAdmissionError, match="must declare"):
        check_backend_admission(Silent())


def test_an_incomplete_declaration_is_rejected():
    extension = _variant("incomplete_backend")
    extension.backend_capabilities = {"devices": ("cpu",)}
    with pytest.raises(BackendAdmissionError, match="incomplete"):
        check_backend_admission(extension)


def test_an_extension_cannot_take_a_builtin_backend_name():
    extension = _variant("torch")
    with pytest.raises(BackendAdmissionError, match="built-in"):
        check_backend_admission(extension)


def test_a_non_backend_kind_is_rejected():
    extension = _variant("wrong_kind", kind="provider")
    with pytest.raises(BackendAdmissionError, match="kind"):
        check_backend_admission(extension)


def test_negotiation_refusal_blocks_admission():
    class Refusing(_ScriptedBackend):
        manifest = ExtensionManifest(
            name="refusing_backend",
            version="1.0.0",
            kind="backend",
            capabilities=frozenset({"cpu"}),
        )

        def negotiate(self, request):
            return CapabilityResponse(False, frozenset(), ("device is busy",))

    with pytest.raises(BackendAdmissionError, match="device is busy"):
        admit_backend_extension(Refusing())
    assert "refusing_backend" not in list_backends()


def test_negotiation_failure_is_contained_as_an_admission_error():
    class Exploding(_ScriptedBackend):
        manifest = ExtensionManifest(
            name="exploding_backend",
            version="1.0.0",
            kind="backend",
            capabilities=frozenset({"cpu"}),
        )

        def negotiate(self, request):
            raise RuntimeError("probe crashed")

    with pytest.raises(BackendAdmissionError, match="probe crashed"):
        admit_backend_extension(Exploding())


def test_checking_admission_registers_nothing(admitted):
    backend = REFERENCE.ReferenceStatevectorBackend()
    capabilities = check_backend_admission(backend)
    assert capabilities.name == "reference_statevector"
    assert "reference_statevector" not in list_backends()
    assert resolve_backend_executor("reference_statevector") is None


def test_an_sdk_incompatible_extension_is_rejected_before_activation():
    extension = type(
        "FutureSdk",
        (_ScriptedBackend,),
        {
            "manifest": ExtensionManifest(
                name="future_sdk_backend",
                version="1.0.0",
                kind="backend",
                capabilities=frozenset({"cpu"}),
                api_version="2.0",
            ),
            "backend_capabilities": dict(_ScriptedBackend.backend_capabilities),
        },
    )()
    with pytest.raises(ExtensionCompatibilityError, match="SDK API 2.0"):
        admit_backend_extension(extension)
    assert "future_sdk_backend" not in list_backends()


def test_a_second_admission_of_a_live_name_is_rejected(admitted):
    name = "re_admitted_backend"
    manifest = ExtensionManifest(
        name=name, version="1.0.0", kind="backend", capabilities=frozenset({"cpu"})
    )
    first = type("First", (_ScriptedBackend,), {"manifest": manifest})()
    second = type("Second", (_ScriptedBackend,), {"manifest": manifest})()
    handle, capabilities = admitted(first)
    try:
        with pytest.raises(BackendAdmissionError, match="already has an admitted"):
            admit_backend_extension(second)
        assert resolve_backend_executor(capabilities.name) is first
    finally:
        handle.close()


def test_admission_errors_are_both_extension_and_capability_errors():
    assert issubclass(BackendAdmissionError, FlagQuantumError)
    assert issubclass(BackendAdmissionError, CapabilityError)


def test_an_admitted_capability_record_is_serializable_without_the_route(admitted):
    from flagquantum.services import preflight

    handle, capabilities = admitted(REFERENCE.ReferenceStatevectorBackend())
    try:
        summary = preflight.capabilities()
        record = summary["backends"][capabilities.name]
        assert record["distribution_semantics"] == "single_device_fast_path"
        assert record["scalability_claim_allowed"] is False
        assert "executor" not in record
        assert get_backend_capabilities(capabilities.name).executor is not None
    finally:
        handle.close()


def test_the_reference_backend_imports_only_the_extension_namespace():
    import ast

    tree = ast.parse(REFERENCE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    flagquantum_imports = {
        name for name in imported if name.split(".")[0] == "flagquantum"
    }
    assert flagquantum_imports == {"flagquantum.ecosystem.extensions"}
