"""Declaration, refusal, and evidence-shape tests for local target emulation.

The end-to-end compile, emit, and execute path lives in
``tests/integration/test_local_target_emulation.py``. This file pins the part a
reader has to be able to trust without running a circuit: which facts describe
the local machine, which describe the claimed target, and every refusal that
happens before a program is compiled.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import flagquantum as fq
from flagquantum.compiler.routing import CouplingMap
from flagquantum.core.target_capabilities import (
    CAPABILITY_NAMES,
    EvidenceLevel,
    FactExposure,
)
from flagquantum.deployment.cloud import CloudBackendProfile
from flagquantum.errors import CompilationError, ExecutionError
from flagquantum.remote import emulation
from flagquantum.remote.emulation import (
    EMULATED_TARGET_ADAPTER_VERSION,
    EMULATED_TARGET_CLASS,
    EMULATED_TARGET_DECLARATION_SOURCE_KIND,
    EMULATED_TARGET_PROVIDER,
    LOCAL_TARGET_EMULATION_SCHEMA,
    TargetEmulationError,
    emulate,
)

pytestmark = pytest.mark.unit


class _Profile:
    """One declared target, so a test states only the field it varies."""

    @staticmethod
    def make(**overrides: object) -> CloudBackendProfile:
        fields: dict[str, object] = {
            "provider": "acme",
            "name": "snake",
            "n_qubits": 5,
            "basis_gates": ("h", "x", "rz", "cx"),
        }
        fields.update(overrides)
        return CloudBackendProfile(**fields)  # type: ignore[arg-type]


def _bell():
    """A two-qubit program whose result depends on the compiled form."""

    return fq.Circuit(2, dtype="complex128").h(0).cx(0, 1)


def _snapshot(
    profile: CloudBackendProfile,
    *,
    precision: str = "complex128",
    emission_profile: str = "openqasm-3.0",
):
    return emulation._target_snapshot(
        profile,
        precision=precision,
        emission_profile=emission_profile,
        native_gates=emulation._declared_native_gates(profile),
        declared_maximum_program_operations=4096,
        declared_maximum_shots=100_000,
    )


def test_emulation_evidence_schema_is_versioned_and_stable() -> None:
    assert LOCAL_TARGET_EMULATION_SCHEMA == "flagquantum.local_target_emulation.v1"
    assert EMULATED_TARGET_ADAPTER_VERSION == "1.0"
    assert EMULATED_TARGET_CLASS == "emulated_target"
    assert EMULATED_TARGET_PROVIDER == "flagquantum.emulated_target"
    assert EMULATED_TARGET_DECLARATION_SOURCE_KIND == (
        "emulated_target_profile_observation"
    )


def test_emulation_entry_is_not_re_exported_from_the_remote_facade() -> None:
    # The facade stays the adapter roster; the emulation entry is reached by its
    # own module path, the same way the deployment dry run is.
    import flagquantum.remote as remote

    assert not hasattr(remote, "emulate")
    assert "emulate" not in remote.__all__
    assert emulation.emulate is emulate


def test_emulation_entry_cannot_submit_or_decode_a_provider_task() -> None:
    # Every name this module exposes is either an evidence value or the entry
    # itself. A caller that wants a submission uses the adapter roster.
    for name in emulation.__all__:
        assert not hasattr(CloudBackendProfile, name)
    assert not hasattr(emulation, "submit")
    assert not hasattr(emulation, "ProviderTaskHandle")
    assert not hasattr(emulation, "DeploymentResult")


def test_snapshot_separates_observed_local_facts_from_declared_target_facts() -> None:
    snapshot = _snapshot(_Profile.make())

    observed = {
        fact.name: fact
        for fact in snapshot.facts
        if fact.fact_exposure is FactExposure.OBSERVED
    }
    declared = {
        fact.name: fact
        for fact in snapshot.facts
        if fact.fact_exposure is FactExposure.DECLARED
    }

    # The local machine answers for the device and the precision.
    assert observed["device.kind"].value == "cpu"
    assert observed["precision.effective_dtype"].value == "complex128"
    # The profile answers for capacity, gate set, result formats, and limits.
    assert declared["qubits.logical_capacity"].value == 5
    assert declared["target.class"].value == EMULATED_TARGET_CLASS
    assert declared["gates.native"].value == ("cx", "h", "rz", "x")
    assert declared["limits.maximum_shots"].value == 100_000
    assert declared["measurements.results"].value == ("counts", "samples")
    assert declared["artifacts.profiles"].value == ("openqasm-3.0",)
    assert set(observed).isdisjoint(declared)


def test_snapshot_never_claims_the_device_that_supplied_the_profile() -> None:
    profile = _Profile.make()
    snapshot = _snapshot(profile)
    identity = snapshot.target_identity

    assert identity.provider == EMULATED_TARGET_PROVIDER
    assert identity.provider != profile.provider
    assert identity.target_class == EMULATED_TARGET_CLASS
    assert identity.target_class != "local_runtime"
    assert identity.target_id == "emulated-acme-snake"
    # The environment is the local one, because that is where the run happens.
    assert identity.environment_id


def test_snapshot_declaration_is_pinned_to_the_profile_payload() -> None:
    # The revision is a digest of the declared payload alone, so it is stable
    # across captures. The snapshot identity is not: the local probe timestamps
    # every capture, which is why a snapshot is evidence about one capture.
    first = _snapshot(_Profile.make())
    again = _snapshot(_Profile.make())
    widened = _snapshot(_Profile.make(n_qubits=6))
    reconnected = _snapshot(
        _Profile.make(coupling_map=CouplingMap(5, ((0, 1), (1, 2), (2, 3), (3, 4))))
    )
    regated = _snapshot(_Profile.make(basis_gates=("h", "x", "rz", "cx", "swap")))
    reencoded = _snapshot(_Profile.make(), emission_profile="qcis-1.0")

    assert first.target_identity.target_revision == (
        again.target_identity.target_revision
    )
    for other in (widened, reconnected, regated, reencoded):
        assert other.target_identity.target_revision != (
            first.target_identity.target_revision
        )
    assert first.captured_at == again.captured_at or (
        first.snapshot_id != again.snapshot_id
    )


def test_snapshot_declaration_evidence_resolves_at_the_observed_level() -> None:
    snapshot = _snapshot(_Profile.make())
    evidence = {item.evidence_id: item for item in snapshot.evidence_refs}

    declared_sources = {
        fact.source.ref
        for fact in snapshot.facts
        if fact.fact_exposure is FactExposure.DECLARED
    }
    assert len(declared_sources) == 1
    declaration = evidence[declared_sources.pop()]
    assert declaration.level is EvidenceLevel.OBSERVABLE
    assert declaration.scope == snapshot.scope
    assert len(declaration.sha256) == 64
    assert declaration.sha256 == snapshot.target_identity.target_revision


def test_snapshot_leaves_capabilities_the_profile_cannot_answer_absent() -> None:
    snapshot = _snapshot(_Profile.make())
    named = {fact.name for fact in snapshot.facts}

    # Nothing here invents an ancilla policy, a physical capacity, or a memory
    # budget: the profile does not state them, so no fact is asserted.
    assert "ancillas.policy" not in named
    assert "ancillas.maximum_compiler" not in named
    assert "qubits.physical_capacity" not in named
    assert "memory.available_bytes" in named
    assert named <= set(CAPABILITY_NAMES)


def test_declared_native_gates_are_canonicalised_and_sorted() -> None:
    gates = emulation._declared_native_gates(
        _Profile.make(basis_gates=("CX", "H", "RZ", "x"))
    )
    assert gates == ("cx", "h", "rz", "x")


def test_emulation_requires_a_declared_profile_not_a_provider_object() -> None:
    with pytest.raises(TypeError, match="target must be a CloudBackendProfile"):
        emulate(_bell(), target="not a profile")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"provider": ""}, "non-empty provider"),
        ({"name": "  "}, "non-empty name"),
    ],
)
def test_emulation_refuses_a_profile_without_an_identity(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(TargetEmulationError, match=message):
        emulate(_bell(), target=_Profile.make(**overrides))


def test_emulation_refuses_a_profile_without_a_native_gate_set() -> None:
    with pytest.raises(TargetEmulationError, match="does not declare basis gates"):
        emulate(_bell(), target=_Profile.make(basis_gates=()))


@pytest.mark.parametrize("declared", [("h", ""), ("h", 3)])
def test_emulation_refuses_a_basis_gate_that_is_not_a_name(
    declared: tuple[object, ...],
) -> None:
    # A profile is external input. A blank or non-string entry is refused by
    # name rather than reaching the canonicaliser.
    with pytest.raises(TargetEmulationError, match="non-empty gate names"):
        emulate(_bell(), target=_Profile.make(basis_gates=declared))


def test_emulation_refuses_a_gate_the_compiler_cannot_lower() -> None:
    with pytest.raises(TargetEmulationError) as error:
        emulate(_bell(), target=_Profile.make(basis_gates=("h", "not_a_gate")))

    message = str(error.value)
    assert "not_a_gate" in message
    assert "operator schemas the compiler can lower" in message


def test_emulation_refuses_a_target_with_no_text_emission_profile() -> None:
    target = _Profile.make(supports_openqasm=False, supports_qcis=False)
    with pytest.raises(TargetEmulationError, match="neither OpenQASM nor"):
        emulate(_bell(), target=target)


def test_emulation_refuses_an_unknown_emission_profile() -> None:
    with pytest.raises(TargetEmulationError) as error:
        emulate(_bell(), target=_Profile.make(), emission_profile="openqasm-9.9")

    assert "openqasm-9.9" in str(error.value)
    assert "openqasm-2.0" in str(error.value)
    assert "qir-2.0" in str(error.value)


def test_emulation_refuses_a_request_the_static_text_profile_cannot_describe() -> None:
    from dataclasses import replace

    from flagquantum.observables import expectation, lower_outputs

    partial = replace(
        _bell().to_ir(),
        measurements=tuple(
            lower_outputs(fq.counts(qubits=[0]), n_wires=2, shots=8, seed=1)
        ),
    )
    with pytest.raises(TargetEmulationError, match="terminal full-register"):
        emulate(partial, target=_Profile.make(), shots=8, seed=1)

    doubled = replace(
        _bell().to_ir(),
        measurements=tuple(lower_outputs(fq.counts(), n_wires=2, shots=8, seed=1))
        + tuple(lower_outputs(fq.counts(), n_wires=2, shots=8, seed=1)),
    )
    with pytest.raises(TargetEmulationError, match="exactly one terminal"):
        emulate(doubled, target=_Profile.make(), shots=8, seed=1)

    # An observable request is a different kind of result, and the static text
    # emission profile cannot describe it. It is refused on the kind rather than
    # on the wire set, so a full-register observable is refused too.
    observed = replace(
        _bell().to_ir(),
        measurements=tuple(
            lower_outputs(expectation(fq.Z(0)), n_wires=2, shots=None, seed=1)
        ),
    )
    with pytest.raises(TargetEmulationError, match="counts, samples requests"):
        emulate(observed, target=_Profile.make(), shots=8, seed=1)


def test_emulation_accepts_an_emission_profile_written_in_any_case() -> None:
    # The profile name is external input, so it is canonicalised like every
    # other declared name rather than silently missing the registry.
    result = emulate(
        _bell(),
        target=_Profile.make(),
        shots=8,
        seed=1,
        emission_profile="openqasm-3.0",
    )
    upper = emulate(
        _bell(),
        target=_Profile.make(),
        shots=8,
        seed=1,
        emission_profile="OpenQASM-3.0",
    )
    assert upper.emission_profile == result.emission_profile == "openqasm-3.0"


def test_emulation_refuses_a_shot_request_above_the_declared_target_cap() -> None:
    with pytest.raises(TargetEmulationError, match="above the declared"):
        emulate(
            _bell(),
            target=_Profile.make(),
            shots=4096,
            declared_maximum_shots=1024,
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        # Each pattern is anchored, because the point of this test is that the
        # entry point refuses before compiling. Several of these values would
        # also be refused further down -- the shots rule by the IR measurement
        # node, the coupling-map rule by the topology pass -- with a message that
        # contains the entry point's own wording, so an unanchored pattern would
        # pass with the entry-point guard deleted.
        ({"shots": 0}, ValueError, r"^shots must be >= 1$"),
        ({"shots": 1.5}, TypeError, r"^shots must be an integer$"),
        ({"seed": -1}, ValueError, r"^seed must be a non-negative integer or None$"),
        (
            {"allow_approximate": "yes"},
            TypeError,
            r"^allow_approximate must be a boolean$",
        ),
        (
            {"noise_model": "flip"},
            TypeError,
            r"^noise_model must be a NoiseModel or None$",
        ),
        (
            {"coupling_map": ((0, 1),)},
            TypeError,
            r"^coupling_map must be a CouplingMap, a DirectedCouplingMap or None$",
        ),
        (
            {"device_profile": "cal"},
            TypeError,
            r"^device_profile must be a DeviceNoiseProfile or None$",
        ),
        (
            {"declared_maximum_program_operations": 0},
            ValueError,
            r"^declared_maximum_program_operations must be >= 1$",
        ),
        (
            {"declared_maximum_program_operations": 1.0},
            TypeError,
            r"^declared_maximum_program_operations must be an integer$",
        ),
        (
            {"declared_maximum_shots": 1.5},
            TypeError,
            r"^declared_maximum_shots must be an integer$",
        ),
    ],
)
def test_emulation_argument_validation_fails_before_compiling(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        emulate(_bell(), target=_Profile.make(), **kwargs)  # type: ignore[arg-type]


def test_emulation_error_is_an_execution_error() -> None:
    assert issubclass(TargetEmulationError, ExecutionError)


def test_emulation_reports_the_compilers_own_refusal_for_an_unreachable_gate() -> None:
    # A basis with no z-rotation and no pi/2 pulse cannot carry Hadamard. The
    # caller sees the compiler's diagnostic, not a reworded one.
    with pytest.raises(CompilationError, match="no verified decomposition"):
        emulate(_bell(), target=_Profile.make(basis_gates=("u3", "cx")))


def test_evaluated_at_is_forwarded_to_the_capability_matcher() -> None:
    # A stale evaluation time is refused by the Core matcher, which proves this
    # entry point passes its snapshot through the real capability check instead
    # of skipping it.
    future = datetime(2100, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(CompilationError, match="stale"):
        emulate(_bell(), target=_Profile.make(), evaluated_at=future)
