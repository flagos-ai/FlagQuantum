"""End-to-end local target emulation: compile for a target, then run it here.

The unit file next door proves the declarations and the refusals. This file
drives the whole path: a declared third-party profile, the target's own
compilation passes, a verified emission payload, and a local CPU execution of
the compiled program. It is an integration test because it crosses the compiler,
deployment, runtime, and remote boundaries in one process.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.compiler import CouplingMap
from flagquantum.compiler.directed_topology import DirectedCouplingMap
from flagquantum.deployment.cloud import CloudBackendProfile
from flagquantum.errors import ExecutionError
from flagquantum.noise import (
    NoiseModel,
    bit_flip_channel,
    depolarizing_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.noise.device_profile import (
    DeviceNoiseProfile,
    GateDuration,
    QubitNoiseCalibration,
)
from flagquantum.observables import counts, lower_outputs
from flagquantum.remote.emulation import emulate

pytestmark = pytest.mark.integration

_CHAIN_5 = ((0, 1), (1, 2), (2, 3), (3, 4))
_CHAIN_6 = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))


def _device(**overrides: object) -> CloudBackendProfile:
    fields: dict[str, object] = {
        "provider": "acme",
        "name": "snake",
        "n_qubits": 5,
        "basis_gates": ("h", "x", "rz", "cx"),
        "coupling_map": CouplingMap(5, _CHAIN_5),
    }
    fields.update(overrides)
    return CloudBackendProfile(**fields)  # type: ignore[arg-type]


def _nonlocal_ghz() -> fq.Circuit:
    """A GHZ state whose second gate is not adjacent on a line coupling map."""

    return fq.Circuit(3, dtype="complex128").h(0).cx(0, 2)


def test_emulation_routes_a_nonlocal_gate_through_the_target_topology() -> None:
    result = emulate(_nonlocal_ghz(), target=_device(), shots=256, seed=7)
    diagnostic = result.diagnostic

    # Routing is the target's: the nonlocal CX became SWAPs around a legal CX.
    assert diagnostic.routing_strategy == "restore_after_each_gate"
    assert diagnostic.source_instruction_count == 2
    assert diagnostic.routed_instruction_count == 4
    assert diagnostic.inserted_swap_count == 2
    assert diagnostic.direction_semantics == "undirected"
    assert diagnostic.final_logical_to_physical == (0, 1, 2)
    # Routing restores the logical output layout, so results read as written.
    assert result.noiseless.counts == {"000": 119, "101": 137}


def test_emulation_emits_the_program_the_target_would_have_received() -> None:
    result = emulate(_nonlocal_ghz(), target=_device(), shots=64, seed=1)

    assert result.emission_profile == "openqasm-3.0"
    assert result.media_type.startswith("text/x-openqasm;version=3.0")
    # The conformance pass re-parsed this text, so the payload is the verified
    # one and its operation count is a parsed count rather than a declared one.
    assert result.diagnostic.parsed_operation_count == 8
    assert "OPENQASM 3.0;" in result.emitted_program
    assert "measure" in result.emitted_program
    assert result.diagnostic.emission_identity
    assert result.diagnostic.conformance_identity
    # The declared gate set is reported next to the text so a caller can see
    # which native set produced it.
    assert result.declared_native_gates == ("cx", "h", "rz", "x")
    assert result.declared_logical_capacity == 5


def test_emulation_reports_a_decomposition_instead_of_silently_keeping_gates() -> None:
    ibm_like = _device(n_qubits=4, basis_gates=("rz", "sx", "cx"), coupling_map=None)
    circuit = fq.Circuit(2, dtype="complex128").h(0).cx(0, 1)
    result = emulate(circuit, target=ibm_like, shots=64, seed=1)

    rewrites = {
        item.source_opcode: item.replacement_opcodes
        for item in result.diagnostic.decompositions
    }
    # Hadamard in {RZ, SX, CX} is a three-pulse sequence, and it is reported.
    assert rewrites["h"] == ("rz", "sx", "rz")
    assert result.diagnostic.rewritten is True
    assert set(result.diagnostic.native_opcodes) <= {"rz", "sx", "cx", "swap"}
    assert set(result.noiseless.counts) <= {"00", "11"}
    assert sum(result.noiseless.counts.values()) == 64


def test_emulation_reports_no_rewrite_when_the_program_is_already_native() -> None:
    result = emulate(
        fq.Circuit(2, dtype="complex128").h(0).cx(0, 1),
        target=_device(basis_gates=("h", "cx")),
        shots=32,
        seed=1,
    )
    assert result.diagnostic.decompositions == ()
    assert result.diagnostic.rewritten is False


def test_emulation_accepts_a_program_that_already_asks_for_counts() -> None:
    ir = replace(
        _nonlocal_ghz().to_ir(),
        measurements=tuple(lower_outputs(counts(), n_wires=3, shots=128, seed=5)),
    )
    result = emulate(ir, target=_device(), shots=128, seed=5)
    assert sum(result.noiseless.counts.values()) == 128
    assert set(result.noiseless.counts) <= {"000", "101"}


def test_emulation_is_reproducible_where_it_makes_a_reproducible_claim() -> None:
    first = emulate(_nonlocal_ghz(), target=_device(), shots=256, seed=7)
    second = emulate(_nonlocal_ghz(), target=_device(), shots=256, seed=7)

    assert first.emitted_program == second.emitted_program
    assert first.noiseless.counts == second.noiseless.counts
    assert first.diagnostic.routed_instruction_count == (
        second.diagnostic.routed_instruction_count
    )
    # The identities fold in the snapshot, and the probe stamps each capture,
    # so identical work is still two distinct pieces of evidence.
    assert first.target_snapshot_id != second.target_snapshot_id


def test_emulation_uses_the_profile_topology_when_none_is_given_but_can_override_it() -> (
    None
):
    wide = _device(n_qubits=5, coupling_map=None)
    circuit = fq.Circuit(3, dtype="complex128").h(0).cx(0, 2)

    unrouted = emulate(circuit, target=wide, shots=64, seed=1)
    assert unrouted.diagnostic.routing_strategy is None
    assert unrouted.diagnostic.inserted_swap_count == 0
    assert unrouted.diagnostic.physical_plan_version == "1.0"
    assert unrouted.diagnostic.physical_slot_count == 3
    # An undirected run projects nothing, and it says so rather than implying
    # an allocation it did not perform.
    assert unrouted.diagnostic.logical_result_physical_slots == ()
    assert unrouted.diagnostic.allocation_identity is None

    routed = emulate(
        circuit, target=wide, shots=64, seed=1, coupling_map=CouplingMap(5, _CHAIN_5)
    )
    assert routed.diagnostic.routing_strategy == "restore_after_each_gate"
    assert routed.diagnostic.inserted_swap_count == 2


def test_emulation_handles_a_directed_target_and_reports_its_allocation() -> None:
    directed = DirectedCouplingMap(6, _CHAIN_6)
    result = emulate(
        fq.Circuit(3, dtype="complex128").h(0).cx(0, 2),
        target=_device(n_qubits=6, coupling_map=None, basis_gates=("h", "cx")),
        shots=64,
        seed=3,
        emission_profile="qir-2.0",
        coupling_map=directed,
    )
    diagnostic = result.diagnostic

    assert diagnostic.direction_semantics == "directed_cx"
    # Every wire of the directed device is an addressable slot, and only the
    # three logical ones carry results.
    assert diagnostic.physical_plan_version == "3.0"
    assert diagnostic.physical_slot_count == 6
    assert diagnostic.logical_wire_count == 3
    assert diagnostic.logical_result_physical_slots == (0, 1, 2)
    assert diagnostic.allocation_identity
    assert result.media_type == (
        "text/x-llvm-ir;qir-profile=base;qir-version=2.0;charset=utf-8"
    )
    assert set(result.noiseless.counts) <= {"000", "101"}
    assert sum(result.noiseless.counts.values()) == 64


def test_emulation_reports_an_approximate_representation_as_approximate() -> None:
    # A statevector-sized program keeps the exact path, and the record says so.
    result = emulate(_nonlocal_ghz(), target=_device(), shots=64, seed=1)
    record = result.noiseless

    assert record.representation == "statevector"
    assert record.approximate is False
    assert record.exact_channel is False
    assert record.noise_source == "none"
    assert record.noise_model_identity is None
    assert result.noisy is None


def test_emulation_applies_an_opt_in_noise_model_to_the_compiled_program() -> None:
    model = NoiseModel()
    model.add(["cx"], two_qubit_depolarizing_channel(0.05))
    model.add(["h", "x", "rz"], depolarizing_channel(0.01))
    result = emulate(
        _nonlocal_ghz(), target=_device(), shots=256, seed=7, noise_model=model
    )

    assert result.noisy is not None
    assert result.noisy.representation == "density_matrix"
    assert result.noisy.approximate is False
    assert result.noisy.exact_channel is True
    # The caller's own model is named as such, not as device calibration.
    assert result.noisy.noise_source == "caller_model"
    assert result.noisy.noise_model_identity
    assert result.noiseless.noise_source == "none"
    # The noiseless record is still present beside it, so the two are comparable
    # and neither is presented as the other.
    assert result.noiseless.counts == {"000": 119, "101": 137}
    assert result.noisy.counts != result.noiseless.counts
    assert set(result.noisy.counts) - set(result.noiseless.counts)
    assert sum(result.noisy.counts.values()) == 256


def test_emulation_holds_a_qcis_only_target_to_its_own_profile() -> None:
    target = _device(
        provider="originq",
        name="wuyuan",
        n_qubits=5,
        coupling_map=None,
        supports_openqasm=False,
        supports_qcis=True,
    )
    result = emulate(_nonlocal_ghz(), target=target, shots=64, seed=1)

    assert result.emission_profile == "qcis-1.0"
    assert result.diagnostic.backend == "qcis"
    assert result.media_type.startswith("text/x-qcis;version=1.0")
    assert result.noiseless.counts == {"000": 36, "101": 28}


def test_emulation_record_serializes_into_portable_evidence() -> None:
    result = emulate(_nonlocal_ghz(), target=_device(), shots=128, seed=4)
    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["schema"] == "flagquantum.local_target_emulation.v1"
    assert len(payload["emitted_program_sha256"]) == 64
    assert payload["noisy"] is None
    assert sum(payload["noiseless"]["counts"].values()) == 128
    assert payload["noiseless"]["representation"] == "statevector"
    assert payload["noiseless"]["shots"] == 128
    assert payload["noiseless"]["seed"] == 4
    assert payload["noiseless"]["exact_channel"] is False
    assert payload["noiseless"]["approximate"] is False
    assert payload["noiseless"]["noise_source"] == "none"
    # The digest is of the emitted text, so a reader can check the payload a
    # target would have received rather than taking this record's word for it.
    assert (
        payload["emitted_program_sha256"]
        == hashlib.sha256(result.emitted_program.encode("utf-8")).hexdigest()
    )
    assert payload["diagnostic"]["routed_instruction_count"] == 4
    assert payload["diagnostic"]["inserted_swap_count"] == 2
    assert payload["target_identity"]["target_class"] == "emulated_target"
    assert payload["declared_native_gates"] == ["cx", "h", "rz", "x"]


def test_emulation_keeps_the_target_snapshot_and_the_execution_separate() -> None:
    result = emulate(_nonlocal_ghz(), target=_device(), shots=64, seed=1)

    assert result.target_snapshot_id
    assert result.diagnostic.target_snapshot_id == result.target_snapshot_id
    # The execution happened on the local CPU, at the precision the program
    # declared, under the declared target's compiled form.
    assert result.precision == "complex128"
    assert result.shots == 64
    assert result.seed == 1
    assert result.target_identity.target_id == "emulated-acme-snake"


def test_the_example_runs_as_documented_from_the_repository_root() -> None:
    # A golden path nothing runs is the defect the example exists to fix, so it
    # is executed the way the README documents it.
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.remote.emulate_local_target"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )

    output = completed.stdout
    assert "emulated target: emulated-acme-snake" in output
    assert "declared native gates: cx, h, rz, x" in output
    assert (
        "routing: restore_after_each_gate -> 4 instructions, 2 swap(s) inserted"
        in output
    )
    assert "noiseless counts: {'000': 119, '101': 137}" in output
    assert "noisy representation: density_matrix" in output
    assert "calibrated noise source: device_profile" in output
    assert "FlagQuantum local target emulation check passed" in output


def _calibrated_profile() -> DeviceNoiseProfile:
    return DeviceNoiseProfile(
        qubits=tuple(
            QubitNoiseCalibration(wire=wire, t1=50_000.0, t2=70_000.0)
            for wire in range(5)
        ),
        gate_durations=(
            GateDuration("h", 20.0),
            GateDuration("x", 20.0),
            GateDuration("rz", 0.0),
            GateDuration("cx", 200.0),
            GateDuration("swap", 600.0),
            GateDuration("measure", 1000.0),
        ),
        source="example-calibration",
        captured_at="2026-10-03T00:00:00+00:00",
    )


def test_emulation_can_take_its_noise_from_a_declared_calibration() -> None:
    # The compiled program is what the calibration has to cover: routing and
    # native-gate legalization run first, so the profile is asked about the
    # instructions the target would actually run.
    result = emulate(
        _nonlocal_ghz(),
        target=_device(),
        shots=256,
        seed=7,
        device_profile=_calibrated_profile(),
    )

    assert result.noisy is not None
    assert result.noisy.noise_source == "device_profile"
    assert result.noisy.representation == "density_matrix"
    assert result.noisy.exact_channel is True
    assert result.noisy.noise_model_identity
    assert result.noiseless.counts == {"000": 119, "101": 137}
    assert sum(result.noisy.counts.values()) == 256
    assert set(result.noisy.counts) - set(result.noiseless.counts)


def test_emulation_refuses_a_calibration_that_cannot_time_the_program() -> None:
    thin = DeviceNoiseProfile(
        qubits=(QubitNoiseCalibration(wire=0, t1=50_000.0, t2=70_000.0),),
        gate_durations=(GateDuration("h", 20.0),),
        source="thin-calibration",
        captured_at="2026-10-03T00:00:00+00:00",
    )
    with pytest.raises(ExecutionError) as error:
        emulate(_nonlocal_ghz(), target=_device(), shots=8, device_profile=thin)

    message = str(error.value)
    assert "thin-calibration" in message
    assert "'cx'" in message


def test_emulation_refuses_a_model_and_a_calibration_together() -> None:
    model = NoiseModel()
    model.add(["cx"], bit_flip_channel(0.01))
    with pytest.raises(TypeError, match="not both"):
        emulate(
            _nonlocal_ghz(),
            target=_device(),
            shots=8,
            noise_model=model,
            device_profile=_calibrated_profile(),
        )
