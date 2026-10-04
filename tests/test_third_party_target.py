"""A provider-declared target compiles and executes, with no FlagQuantum change.

This is the W5-12 vertical path, expressed as one scenario a reviewer can follow:
a third-party device declaration becomes an authoritative capability snapshot and
a coupling map, a user's circuit is legalized against them, noise from the
declaration's own evidence is applied, and the result is executed both locally
and through an admitted third-party backend route.

The point of the scenario is that none of it required a change to the public API,
to ``flagquantum/_api.py``, or to any Compiler, Runtime, or Simulation module. The
provider supplies data; the existing authoritative engine does the rest.
"""

from __future__ import annotations

import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap
from flagquantum.compiler.target_legalization import (
    TargetLegalizationError,
    legalize_circuit_for_target,
)
from flagquantum.ecosystem.extensions.admission import (
    admit_backend_extension,
    withdraw_backend,
)
from flagquantum.ecosystem.extensions.conformance import run_device_conformance
from flagquantum.ecosystem.extensions.target_sdk import (
    TargetDescriptionError,
    check_target_description,
    target_capability_snapshot,
)
from flagquantum.noise import NoiseModel, depolarizing_channel
from flagquantum.observables import Z, expectation
from flagquantum.remote.emulation import TargetEmulationError, emulate_declared_target
from flagquantum.runtime.backend_registry import list_backends
from flagquantum.runtime.options import ExecutionOptions

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]

# Both files are the shipped reference material a provider copies, loaded from
# disk rather than imported, so this test fails if the published example breaks.
TARGET_REFERENCE_PATH = ROOT / "examples/extensions/reference_target_extension.py"
BACKEND_REFERENCE_PATH = ROOT / "examples/extensions/reference_backend_extension.py"


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TARGET_REFERENCE = _load("reference_target_extension", TARGET_REFERENCE_PATH)
BACKEND_REFERENCE = _load("reference_backend_extension", BACKEND_REFERENCE_PATH)

# The correlator the scenario measures, and the rotation whose expectation it is.
_ROTATION = 0.7
_CORRELATOR = expectation(Z(0) @ Z(3))
# One-qubit depolarizing channel per two-qubit gate, applied by the caller.
_GATE_ERROR = 0.05
# `depolarizing_channel` is a one-qubit Pauli channel and `NoiseModel` applies it
# to every wire of the gate it is attached to, so a two-qubit gate damps a
# two-wire correlator by this factor exactly once. The trailing margin against a
# wrong rule is large: attaching the channel to one wire only would give 0.9333.
_EXPECTED_PER_CX_DECAY = (1.0 - 4.0 * _GATE_ERROR / 3.0) ** 2


@pytest.fixture(scope="module")
def declared_target():
    """The provider's declaration, validated and converted exactly once."""

    device = TARGET_REFERENCE.AcmeAuroraTarget()
    (description,) = check_target_description(device)
    return device, description, target_capability_snapshot(description)


# Three variations on the same provider, each a declaration only a provider
# could write: one that offers a second device, one whose device returns nothing
# this local emulation produces, and one that states neither a result format nor
# a limit. They check that emulation refuses a declaration it cannot honour
# rather than resolving it by guessing, and that it does not read silence in a
# declaration as a refusal.
_FIRST_DECLARATION = TARGET_REFERENCE.AcmeAuroraTarget._declarations[0]
_SECOND_DECLARATION = {
    **_FIRST_DECLARATION,
    "target_id": "acme-aurora-8q",
    "device_id": "acme-aurora-8q",
    "n_qubits": 8,
    "coupling_edges": tuple((index, index + 1) for index in range(7)),
}
_EXPECTATION_ONLY_DECLARATION = {
    **_FIRST_DECLARATION,
    "declared_facts": {
        **_FIRST_DECLARATION["declared_facts"],
        "measurements.results": ("expectation",),
    },
}
_UNQUALIFIED_DECLARATION = {
    **_FIRST_DECLARATION,
    "target_id": "acme-aurora-4q-minimal",
    "device_id": "acme-aurora-4q-minimal",
    "declared_facts": {"ancillas.policy": "compiler_allocated"},
    "observed_facts": {
        fact: value
        for fact, value in _FIRST_DECLARATION["observed_facts"].items()
        if not fact.startswith("limits.")
    },
}
# Both limits are declared here rather than observed, and they are far apart, so
# reading the operation limit out of the shot limit cannot pass unnoticed.
_LIMITED_DECLARATION = {
    **_FIRST_DECLARATION,
    "target_id": "acme-aurora-4q-limited",
    "device_id": "acme-aurora-4q-limited",
    "declared_facts": {
        **_FIRST_DECLARATION["declared_facts"],
        "limits.maximum_shots": 16,
        "limits.maximum_program_operations": 2,
    },
    "observed_facts": {
        fact: value
        for fact, value in _FIRST_DECLARATION["observed_facts"].items()
        if not fact.startswith("limits.")
    },
}


class _TwoDeviceProvider(TARGET_REFERENCE.AcmeAuroraTarget):
    """A provider whose listing offers two devices behind one manifest."""

    _declarations = (_FIRST_DECLARATION, _SECOND_DECLARATION)


class _ExpectationOnlyProvider(TARGET_REFERENCE.AcmeAuroraTarget):
    """A provider whose device returns expectation values, not samples."""

    _declarations = (_EXPECTATION_ONLY_DECLARATION,)


class _UnqualifiedProvider(TARGET_REFERENCE.AcmeAuroraTarget):
    """A provider whose device states no result format and no limit."""

    _declarations = (_UNQUALIFIED_DECLARATION,)


class _LimitedProvider(TARGET_REFERENCE.AcmeAuroraTarget):
    """A provider that declares both of its own limits, and nothing observed."""

    _declarations = (_LIMITED_DECLARATION,)


def test_a_third_party_declaration_passes_conformance(declared_target):
    device, _, _ = declared_target

    report = run_device_conformance(device)

    assert report.checks == (
        "manifest_serialization",
        "declaration",
        "snapshot_matching",
        "cleanup",
    )


def test_the_declaration_supplies_the_connectivity_the_compiler_needs(
    declared_target,
):
    _, description, snapshot = declared_target

    # The provider states edges; Compiler's own type consumes them unchanged.
    coupling_map = CouplingMap(description.n_wires, description.coupling_edges)

    assert coupling_map.n_wires == 4
    assert description.native_gates == ("cx", "h", "rz", "s", "sdg", "sx", "x")
    # `ry` is deliberately absent, so the existing native-gate pass must act.
    assert "ry" not in description.native_gates
    assert snapshot.target_identity.provider == "acme_aurora"
    assert snapshot.target_identity.target_revision == "rev7"


def test_a_user_circuit_is_legalized_onto_the_declared_device(declared_target):
    _, description, snapshot = declared_target
    coupling_map = CouplingMap(description.n_wires, description.coupling_edges)
    program = fq.Circuit(4).h(0).ry(3, _ROTATION).cx(0, 3)

    legalized = legalize_circuit_for_target(
        program, backend="provider", snapshot=snapshot, coupling_map=coupling_map
    )

    opcodes = Counter(item.name for item in legalized.program.instructions)
    # Every gate the provider does not offer natively is gone, and only its own
    # gates remain: one `ry` decomposition plus four `swap` decompositions.
    assert "ry" not in opcodes
    assert set(opcodes) <= set(description.native_gates)
    assert opcodes["cx"] == 13
    assert legalized.schedule.depth == 14
    assert legalized.program.dtype == "complex64"
    assert legalized.target_snapshot_id == snapshot.snapshot_id
    assert [
        item.source_opcode for item in legalized.native_gate_legalization.decompositions
    ] == [
        "ry",
        "swap",
        "swap",
        "swap",
        "swap",
    ]


def test_routing_does_not_change_the_noiseless_result(declared_target):
    _, description, snapshot = declared_target
    coupling_map = CouplingMap(description.n_wires, description.coupling_edges)
    program = fq.Circuit(4).h(0).ry(3, _ROTATION).cx(0, 3)
    options = ExecutionOptions(mode="density_matrix")

    legalized = legalize_circuit_for_target(
        program, backend="provider", snapshot=snapshot, coupling_map=coupling_map
    )
    direct = fq.run(program, options=options, outputs=_CORRELATOR)
    routed = fq.run(legalized.program, options=options, outputs=_CORRELATOR)

    direct_value = float(direct.expectations[0])
    routed_value = float(routed.expectations[0])
    # A swap is three `cx`, so this is the routing pass being correct rather than
    # a short circuit that happened to need no swaps.
    assert abs(routed_value - direct_value) < 1e-6
    # And both are the physical answer, not merely equal to each other.
    assert abs(direct_value - math.cos(_ROTATION)) < 1e-6


def test_the_legalized_program_executes_through_an_admitted_provider_route(
    declared_target,
):
    _, description, snapshot = declared_target
    coupling_map = CouplingMap(description.n_wires, description.coupling_edges)
    # This circuit stays inside the reference route's declared gate set, so the
    # same target snapshot drives compilation and execution end to end.
    program = fq.Circuit(4).h(0).rz(3, _ROTATION).cx(0, 1).cx(1, 2).cx(2, 3)
    legalized = legalize_circuit_for_target(
        program, backend="provider", snapshot=snapshot, coupling_map=coupling_map
    )
    assert set(item.name for item in legalized.program.instructions) == {
        "cx",
        "h",
        "rz",
    }

    handle, capabilities = admit_backend_extension(
        BACKEND_REFERENCE.ReferenceStatevectorBackend()
    )
    try:
        routed = fq.run(
            legalized.program,
            options=ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )
        local = fq.run(legalized.program, options=ExecutionOptions(mode="statevector"))
        unrouted = fq.run(program, options=ExecutionOptions(mode="statevector"))
    finally:
        # A live route outlives its handle, so the scenario withdraws it rather
        # than leaving a third-party backend selectable for later tests.
        handle.close()
        withdraw_backend(capabilities.name)

    # The third-party route, the built-in engine, and the pre-routing program all
    # produce the same state, so routing and the route itself are semantics-neutral.
    assert torch.allclose(routed.state, local.state, atol=1e-6)
    assert torch.allclose(routed.state, unrouted.state, atol=1e-6)
    # The withdrawal above must have taken the name out of the registry.
    assert capabilities.name not in list_backends()


def test_declared_gate_errors_damp_the_measured_correlator_as_reported(
    declared_target,
):
    _, description, snapshot = declared_target
    coupling_map = CouplingMap(description.n_wires, description.coupling_edges)
    program = fq.Circuit(4).h(0).ry(3, _ROTATION).cx(0, 3)
    options = ExecutionOptions(mode="density_matrix")
    model = NoiseModel().add("cx", depolarizing_channel(_GATE_ERROR))
    legalized = legalize_circuit_for_target(
        program, backend="provider", snapshot=snapshot, coupling_map=coupling_map
    )

    noiseless = float(
        fq.run(program, options=options, outputs=_CORRELATOR).expectations[0]
    )
    noisy = float(
        fq.run(
            program, options=options, outputs=_CORRELATOR, noise_model=model
        ).expectations[0]
    )
    measured_decay = noisy / noiseless

    assert abs(measured_decay - _EXPECTED_PER_CX_DECAY) < 1e-6

    # Routing the same circuit onto the declared linear device adds ten more
    # two-qubit gates, so the same device error costs more on the real topology.
    routed_noiseless = float(
        fq.run(legalized.program, options=options, outputs=_CORRELATOR).expectations[0]
    )
    routed_noisy = float(
        fq.run(
            legalized.program,
            options=options,
            outputs=_CORRELATOR,
            noise_model=model,
        ).expectations[0]
    )
    routed_decay = routed_noisy / routed_noiseless
    assert routed_decay < measured_decay < 1.0
    assert abs(routed_noiseless - noiseless) < 1e-6


def test_the_declared_topology_is_the_one_the_router_honours(declared_target):
    from flagquantum.compiler.target_legalization import TargetLegalizationError

    _, description, snapshot = declared_target
    program = fq.Circuit(4).h(0).ry(3, _ROTATION).cx(0, 3)

    # The declared device is a linear chain, so the far pair must be swapped over.
    declared = CouplingMap(description.n_wires, tuple(description.coupling_edges))
    # A device that already couples the pair directly needs no swaps for this
    # circuit: the connectivity is a constraint on the router, not decoration.
    direct = CouplingMap(description.n_wires, ((0, 3),))

    on_declared = legalize_circuit_for_target(
        program, backend="provider", snapshot=snapshot, coupling_map=declared
    )
    on_direct = legalize_circuit_for_target(
        program, backend="provider", snapshot=snapshot, coupling_map=direct
    )

    assert len(on_direct.program.instructions) < len(on_declared.program.instructions)
    assert on_direct.schedule.depth < on_declared.schedule.depth

    # A device that cannot carry the pair at all is refused rather than routed
    # through a connection the provider never declared.
    unreachable = CouplingMap(description.n_wires, ((0, 1), (1, 2)))
    with pytest.raises(TargetLegalizationError, match="[Nn]o coupling path"):
        legalize_circuit_for_target(
            program, backend="provider", snapshot=snapshot, coupling_map=unreachable
        )


def test_a_circuit_beyond_the_declared_capacity_is_refused(declared_target):
    from flagquantum.compiler.target_legalization import TargetLegalizationError

    _, description, snapshot = declared_target
    couplings = tuple(description.coupling_edges)
    too_wide = fq.Circuit(description.n_qubits + 1)
    too_wide.h(0)
    too_wide.cx(0, 1)

    with pytest.raises(TargetLegalizationError, match="qubits.logical_capacity"):
        legalize_circuit_for_target(
            too_wide,
            backend="provider",
            snapshot=snapshot,
            coupling_map=CouplingMap(description.n_wires + 1, couplings),
        )


def test_a_declared_target_runs_where_it_is_declared_with_one_call(declared_target):
    _, description, _ = declared_target
    # A GHZ state on the far pair of the declared chain: the provider's topology
    # is what makes this need the router, so the record shows the device working.
    program = fq.Circuit(4, dtype="complex128").h(0).cx(0, 3)

    emulation = emulate_declared_target(
        program, extension=TARGET_REFERENCE.AcmeAuroraTarget(), shots=512, seed=11
    )

    assert emulation.declared_logical_capacity == description.n_qubits
    assert emulation.declared_native_gates == description.native_gates
    assert emulation.emission_profile == "openqasm-3.0"
    # The provider declared a linear chain, so the far pair is swapped over.
    assert emulation.diagnostic.inserted_swap_count > 0
    assert set(emulation.diagnostic.native_opcodes) <= set(description.native_gates)
    # And the answer is the physical one: a two-outcome GHZ correlator.
    assert set(emulation.noiseless.counts) == {"0000", "1001"}
    assert sum(emulation.noiseless.counts.values()) == 512
    assert emulation.noisy is None


def test_the_emulation_record_is_the_emulator_s_own_and_not_the_device_s(
    declared_target,
):
    _, description, _ = declared_target

    emulation = emulate_declared_target(
        fq.Circuit(4).h(0),
        extension=TARGET_REFERENCE.AcmeAuroraTarget(),
        shots=64,
        seed=5,
    )

    identity = emulation.target_identity
    # The provider's own revision and environment describe hardware this run did
    # not touch, so the record names the emulator instead of borrowing them.
    assert identity.target_id == "emulated-acme_aurora-acme-aurora-4q"
    assert identity.provider == "flagquantum.emulated_target"
    assert identity.target_revision != description.target_revision
    assert identity.environment_id != description.environment_id
    # The capacity and gate set are still the provider's statements, unaltered.
    assert emulation.declared_native_gates == description.native_gates


def test_a_declaration_s_own_shot_limit_is_the_limit_enforced(declared_target):
    _, description, _ = declared_target
    device = TARGET_REFERENCE.AcmeAuroraTarget()
    maximum = description.observed_facts["limits.maximum_shots"]
    program = fq.Circuit(4).h(0)

    at_the_limit = emulate_declared_target(
        program, extension=device, shots=maximum, seed=5
    )

    assert sum(at_the_limit.noiseless.counts.values()) == maximum
    with pytest.raises(TargetEmulationError, match="above the declared target"):
        emulate_declared_target(program, extension=device, shots=maximum + 1)


def test_a_provider_offering_several_devices_must_say_which_to_emulate():
    program = fq.Circuit(4).h(0)

    with pytest.raises(TargetEmulationError, match="pass device_id"):
        emulate_declared_target(program, extension=_TwoDeviceProvider(), shots=16)

    chosen = emulate_declared_target(
        program,
        extension=_TwoDeviceProvider(),
        device_id="acme-aurora-8q",
        shots=16,
        seed=5,
    )

    assert chosen.declared_logical_capacity == 8
    # A padded identifier is the identifier, so a caller's stray whitespace is
    # not read as a request for a device that does not exist.
    padded = emulate_declared_target(
        program,
        extension=_TwoDeviceProvider(),
        device_id="  acme-aurora-4q  ",
        shots=16,
        seed=5,
    )

    assert padded.declared_logical_capacity == 4
    with pytest.raises(TargetEmulationError, match="no device 'acme-aurora-16q'"):
        emulate_declared_target(
            program,
            extension=_TwoDeviceProvider(),
            device_id="acme-aurora-16q",
            shots=16,
        )


def test_a_declaration_that_states_no_format_and_no_limit_still_runs():
    # Silence in a declaration narrows nothing, so the module's own default shot
    # limit is what applies rather than the device being refused for not saying.
    emulation = emulate_declared_target(
        fq.Circuit(4).h(0), extension=_UnqualifiedProvider(), shots=16, seed=5
    )

    assert sum(emulation.noiseless.counts.values()) == 16
    assert emulation.emission_profile == "openqasm-3.0"


def test_a_device_that_returns_no_format_this_emulator_produces_is_refused():
    with pytest.raises(TargetEmulationError, match="measurement results expectation"):
        emulate_declared_target(
            fq.Circuit(4).h(0), extension=_ExpectationOnlyProvider(), shots=16
        )


def test_a_text_profile_this_emulator_cannot_emit_never_runs():
    device = TARGET_REFERENCE.AcmeAuroraTarget()

    qcis = emulate_declared_target(
        fq.Circuit(4).h(0),
        extension=device,
        emission_profile="qcis-1.0",
        shots=16,
        seed=5,
    )

    assert qcis.emission_profile == "qcis-1.0"
    # QIR is a target profile Compiler can emit, but it is not text this local
    # entry point executes, so it is refused rather than approximated.
    with pytest.raises(TargetEmulationError, match="not a text profile"):
        emulate_declared_target(
            fq.Circuit(4).h(0), extension=device, emission_profile="qir-2.0"
        )


def test_a_non_device_extension_has_no_target_to_emulate():
    with pytest.raises(TargetDescriptionError, match="only a device extension"):
        emulate_declared_target(
            fq.Circuit(2).h(0),
            extension=BACKEND_REFERENCE.ReferenceStatevectorBackend(),
            shots=16,
        )


def test_the_declared_target_entry_stays_out_of_the_stable_root():
    # Reached by its module path, exactly as `emulate` is, so a provider target
    # adds no Stable Core export and no root namespace name.
    assert not hasattr(fq, "emulate_declared_target")
    assert "emulate_declared_target" not in fq.__all__


def test_the_declared_target_entry_is_the_emulation_modules_own():
    from flagquantum import remote
    from flagquantum.remote import emulation

    assert "emulate_declared_target" in emulation.__all__
    assert callable(emulation.emulate_declared_target)
    # It is not re-exported from the facade, so the remote namespace is unchanged.
    assert not hasattr(remote, "emulate_declared_target")
    assert "emulate_declared_target" not in remote.__all__


def test_a_dialect_this_emulator_has_no_profile_for_is_named_in_the_refusal():
    with pytest.raises(TargetEmulationError, match="unsupported target emission"):
        emulate_declared_target(
            fq.Circuit(4).h(0),
            extension=TARGET_REFERENCE.AcmeAuroraTarget(),
            emission_profile="openqasm-9.9",
            shots=16,
        )


def test_the_callers_dialect_is_honoured_name_for_name():
    device = TARGET_REFERENCE.AcmeAuroraTarget()

    # A padded, case-folded name is the profile it names, and the profile is the
    # one requested rather than the profile this module happens to prefer.
    padded = emulate_declared_target(
        fq.Circuit(4).h(0),
        extension=device,
        emission_profile="  OpenQASM-2.0  ",
        shots=16,
        seed=5,
    )

    assert padded.emission_profile == "openqasm-2.0"
    # And the profile the declaration became states exactly that dialect, because
    # it is the emulated target's own emission capability and not a preference.
    from flagquantum.remote.emulation import _declared_target_profile

    (description,) = check_target_description(device)
    qcis_profile = _declared_target_profile(description, emission_profile="qcis-1.0")
    qasm_profile = _declared_target_profile(
        description, emission_profile="openqasm-2.0"
    )

    assert (qcis_profile.supports_openqasm, qcis_profile.supports_qcis) == (False, True)
    assert (qasm_profile.supports_openqasm, qasm_profile.supports_qcis) == (True, False)


def test_a_declaration_s_declared_limits_are_the_limits_that_apply():
    # This declaration states both limits as its own declarations rather than as
    # observations, so reading one as the other cannot pass.
    limited = emulate_declared_target(
        fq.Circuit(4).h(0).h(1),
        extension=_LimitedProvider(),
        shots=16,
        seed=5,
    )

    assert sum(limited.noiseless.counts.values()) == 16
    with pytest.raises(
        TargetLegalizationError, match="limits.maximum_program_operations"
    ):
        emulate_declared_target(
            fq.Circuit(4).h(0).h(1).h(2),
            extension=_LimitedProvider(),
            shots=16,
        )
    with pytest.raises(TargetEmulationError, match="above the declared target"):
        emulate_declared_target(
            fq.Circuit(4).h(0), extension=_LimitedProvider(), shots=17
        )


def test_the_callers_noise_model_and_seed_reach_the_declared_target_run():
    device = TARGET_REFERENCE.AcmeAuroraTarget()
    program = fq.Circuit(4).h(0).cx(0, 3)

    noisy = emulate_declared_target(
        program,
        extension=device,
        shots=256,
        seed=7,
        noise_model=NoiseModel().add("cx", depolarizing_channel(0.05)),
    )

    # The seed the caller passed is the seed the run used, so the record is
    # reproducible rather than merely plausible.
    assert noisy.seed == 7
    assert noisy.noisy is not None
    assert noisy.noisy.noise_source == "caller_model"
    repeat = emulate_declared_target(
        program,
        extension=device,
        shots=256,
        seed=7,
        noise_model=NoiseModel().add("cx", depolarizing_channel(0.05)),
    )

    assert repeat.noiseless.counts == noisy.noiseless.counts
    assert repeat.noisy.counts == noisy.noisy.counts
