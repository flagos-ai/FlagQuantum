"""Emulate a declared third-party target locally before submitting anything.

This example is the ten-minute path for the target-emulation entry: declare a
device the way a provider listing would, compile for it, read what the target
changed, and run the compiled program on this machine. It contacts no provider
and needs no credentials.
"""

from __future__ import annotations

import flagquantum as fq
import flagquantum.compiler as compiler
from flagquantum.deployment.cloud import CloudBackendProfile
from flagquantum.noise import (
    NoiseModel,
    depolarizing_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.noise.device_profile import (
    DeviceNoiseProfile,
    GateDuration,
    QubitNoiseCalibration,
)
from flagquantum.remote.emulation import emulate


def main() -> None:
    # A provider listing as the deployment layer models it: five qubits wired in
    # a line whose native set is {h, x, rz, cx}.
    device = CloudBackendProfile(
        provider="acme",
        name="snake",
        n_qubits=5,
        basis_gates=("h", "x", "rz", "cx"),
        coupling_map=compiler.CouplingMap.line(5),
    )

    # A GHZ state whose second gate is not adjacent on that line.
    circuit = fq.Circuit(n_qubits=3, dtype="complex128").h(0).cx(0, 2)

    result = emulate(circuit, target=device, shots=256, seed=7)

    print("emulated target:", result.target_identity.target_id)
    print("declared capacity:", result.declared_logical_capacity, "qubits")
    print("declared native gates:", ", ".join(result.declared_native_gates))
    print("emission profile:", result.emission_profile)

    diagnostic = result.diagnostic
    print(
        "routing:",
        diagnostic.routing_strategy,
        "->",
        diagnostic.routed_instruction_count,
        "instructions,",
        diagnostic.inserted_swap_count,
        "swap(s) inserted",
    )
    print("logical to physical:", diagnostic.final_logical_to_physical)
    print("schedule depth:", diagnostic.schedule_depth)
    print("conformance parsed operations:", diagnostic.parsed_operation_count)

    print("program the target would have received:")
    print(result.emitted_program)

    # Loopback over one wire of the line: the result must only ever be |000> or
    # |101>, because routing restored the logical output layout.
    assert set(result.noiseless.counts) <= {"000", "101"}
    assert sum(result.noiseless.counts.values()) == 256
    print("noiseless counts:", result.noiseless.counts)
    print("representation:", result.noiseless.representation)

    # Noise is opt-in and reported separately: the noiseless record stays beside
    # it so the two are never confused.
    model = NoiseModel()
    model.add(["cx"], two_qubit_depolarizing_channel(0.05))
    model.add(["h", "x", "rz"], depolarizing_channel(0.01))
    noisy = emulate(circuit, target=device, shots=256, seed=7, noise_model=model)

    assert noisy.noisy is not None
    print("noisy representation:", noisy.noisy.representation)
    print("noisy counts:", noisy.noisy.counts)
    assert set(noisy.noisy.counts) - set(noisy.noiseless.counts)

    # A target can also declare its own calibration, and the noisy record says
    # so rather than passing the calibration off as the caller's model.
    calibration = DeviceNoiseProfile(
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
        ),
        source="acme-snake-2026-10-03",
        captured_at="2026-10-03T00:00:00+00:00",
    )
    calibrated = emulate(
        circuit, target=device, shots=256, seed=7, device_profile=calibration
    )
    assert calibrated.noisy is not None
    print("calibrated noise source:", calibrated.noisy.noise_source)
    print("calibrated counts:", calibrated.noisy.counts)

    # The record is evidence a caller can keep: it separates the declared
    # target, the compilation identities, and the local execution.
    payload = result.to_dict()
    assert payload["schema"] == "flagquantum.local_target_emulation.v1"
    assert payload["noisy"] is None
    print("FlagQuantum local target emulation check passed")


if __name__ == "__main__":
    main()
