"""Run a device a third party declared, without changing FlagQuantum.

This is the ten-minute path for a provider integration: take the provider's own
declaration object, hand it to one entry point, and get back a compiled program,
the target's compilation evidence, and local results. Nothing is registered, no
name is reserved, and no FlagQuantum module is modified to accept the device.

The device is fictional and the run is local, so the record names the emulator
rather than the device. That separation is the point of the example: the
declaration says what the device is, and the execution record says where the
program actually ran.
"""

from __future__ import annotations

import flagquantum as fq
from examples.extensions.reference_target_extension import AcmeAuroraTarget
from flagquantum.remote.emulation import TargetEmulationError, emulate_declared_target


def main() -> None:
    # The provider's own object. It declares a four-qubit chain whose native set
    # is {cx, h, rz, s, sdg, sx, x}: no `ry`, so the compiler must decompose one.
    device = AcmeAuroraTarget()

    # A GHZ state on the far pair of that chain, which needs the router.
    circuit = fq.Circuit(4, dtype="complex128").h(0).cx(0, 3)

    emulation = emulate_declared_target(circuit, extension=device, shots=512, seed=11)

    print("emulated device:", emulation.target_identity.target_id)
    print("declared capacity:", emulation.declared_logical_capacity, "qubits")
    print("declared native gates:", ", ".join(emulation.declared_native_gates))
    print("emission profile:", emulation.emission_profile)

    diagnostic = emulation.diagnostic
    print(
        "routing:",
        diagnostic.routing_strategy,
        "->",
        diagnostic.routed_instruction_count,
        "instructions,",
        diagnostic.inserted_swap_count,
        "swap(s) inserted",
    )
    print("schedule depth:", diagnostic.schedule_depth)

    # The two-outcome GHZ correlator, measured on this machine.
    print("counts:", emulation.noiseless.counts)
    assert set(emulation.noiseless.counts) == {"0000", "1001"}
    assert sum(emulation.noiseless.counts.values()) == 512

    # The declaration's own shot limit is the limit enforced, so a larger
    # request is refused rather than silently served.
    try:
        emulate_declared_target(circuit, extension=device, shots=10_000_001)
    except TargetEmulationError as error:
        print("refused:", error)

    print("FlagQuantum declared target execution check passed")


if __name__ == "__main__":
    main()
