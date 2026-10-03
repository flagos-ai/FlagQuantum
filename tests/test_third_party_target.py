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
from flagquantum.compiler.target_legalization import legalize_circuit_for_target
from flagquantum.ecosystem.extensions.admission import (
    admit_backend_extension,
    withdraw_backend,
)
from flagquantum.ecosystem.extensions.conformance import run_device_conformance
from flagquantum.ecosystem.extensions.target_sdk import (
    check_target_description,
    target_capability_snapshot,
)
from flagquantum.noise import NoiseModel, depolarizing_channel
from flagquantum.observables import Z, expectation
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
