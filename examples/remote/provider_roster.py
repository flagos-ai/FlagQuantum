"""Read a hardware provider roster from vendor declarations and preflight it.

This example is the ten-minute path for the hardware provider expansion: take a
device listing in the shape a vendor publishes it, turn it into a backend
profile the compiler and the deployment layer already understand, and check a
program locally before anything is submitted. It contacts no provider and needs
no credentials, so it is safe to run and to read.

What it demonstrates is the honest half of a hardware integration: the width,
the native gate set, and the connectivity are the vendor's declared facts, and
this package refuses a listing that omits or mistypes one of them instead of
supplying a default. Whether a job then runs on the machine is not something a
test can prove, so this example does not claim it.
"""

from __future__ import annotations

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.remote.qpu import (
    IonQProvider,
    IQMProvider,
    NeutralAtomProvider,
    QuantinuumProvider,
    ionq_backend_profile,
    iqm_backend_profile,
    neutral_atom_backend_profile,
    quantinuum_backend_profile,
)


def main() -> None:
    # A trapped-ion listing: a width and a gate set, and no connectivity,
    # because a shared trap lets every pair interact but a listing that does not
    # say so has not declared it.
    ionq = ionq_backend_profile(
        {"backend": "qpu.aria-1", "qubits": 25, "gates": ["X", "H", "CX"]}
    )
    assert ionq.coupling_map is None
    assert ionq.metadata["connectivity_declared"] is False

    # A superconducting listing: a real lattice, read as declared.
    iqm = iqm_backend_profile(
        {
            "name": "IQM Garnet",
            "qubits": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            "connectivity": [[0, 1], [1, 2]],
            "gates": ["PRX", "CZ"],
        }
    )
    assert iqm.coupling_map is not None
    assert iqm.coupling_map.edges == ((0, 1), (1, 2))
    print("IQM declared lattice:", iqm.coupling_map.edges)

    # A machine-name-only listing and a register of physically placed atoms.
    quantinuum = quantinuum_backend_profile("H2-1E", n_qubits=32)
    register = {
        "name": "aquila",
        "atoms": [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [4.0, 1.0]],
    }
    # The interaction radius is a property of the atom species and the
    # excitation scheme, not of this package, so it is required rather than
    # defaulted: a default would invent a device.
    neutral = neutral_atom_backend_profile(register, interaction_radius=5.0)
    assert neutral.metadata["connectivity_source"] == "derived_from_declared_geometry"
    print("neutral-atom derived couplings:", neutral.coupling_map.edges)
    assert quantinuum.metadata["qubit_count_source"] == "caller"

    # A program built once and preflighted against each device. Nothing is
    # submitted: `dry_run` validates the package locally and returns what would
    # be sent.
    circuit = fq.Circuit(n_qubits=2, dtype="complex128").h(0).cx(0, 1)
    for provider, profile, qasm_version in (
        (IonQProvider(), ionq, 3.0),
        (IQMProvider(), iqm, 3.0),
        (QuantinuumProvider(machine="H2-1E"), quantinuum, 2.0),
    ):
        package = fqd.create_deployment_package(
            circuit, backend=profile, shots=100, qasm_version=qasm_version
        )
        preview = provider.dry_run(package)
        assert preview.compatible, preview.blockers
        print(f"{profile.provider} preflight:", preview.summary())

    # A Quantinuum machine will not accept the QASM 3 package the other two
    # accept, and the adapter refuses it before a paid job exists.
    package = fqd.create_deployment_package(
        circuit, backend=quantinuum, shots=100, qasm_version=3.0
    )
    refused = QuantinuumProvider(machine="H2-1E").dry_run(package)
    assert refused.blockers == ("quantinuum_provider_requires_openqasm_2",)
    print("Quantinuum refusal:", refused.blockers)

    # A neutral-atom machine executes an analog schedule over its register, and
    # this package emits neither that schedule nor a vendor's serialization of
    # it, so the adapter names that gap instead of translating the circuit into
    # a program the hardware would not run as declared.
    package = fqd.create_deployment_package(
        circuit, backend=neutral, shots=100, qasm_version=3.0
    )
    analog = NeutralAtomProvider(register=register, interaction_radius=5.0)
    blocked = analog.dry_run(package)
    assert blocked.compatible is False
    print("neutral-atom blocker:", blocked.blockers)

    print("FlagQuantum hardware provider roster check passed")


if __name__ == "__main__":
    main()
