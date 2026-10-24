"""Build and sample an exclusive two-qubit Pauli circuit-noise model.

Run with an installed package or PYTHONPATH=.; only core Torch is required.
This samples a declared DEM. Circuit execution needs the optional Stim extra.
"""

import torch

from flagquantum.noise import NoiseModel, two_qubit_depolarizing_channel
from flagquantum.qec import DetectorErrorModel, RepetitionCode, build_memory_circuit


def main() -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=2)
    noise = NoiseModel().add(
        "cx",
        two_qubit_depolarizing_channel(0.15, dtype=torch.complex128),
        qubits=(0, 3),
    )
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = model.dem_sampling(shots=4, seed=47)
    print("Detectors, observables:", model.num_detectors, model.num_observables)
    print("Independent circuit locations:", len(model.stated_error_ids()))
    print("Exclusive groups:", len(model.exclusive_groups()))
    print(
        "Sample shapes:", tuple(sample.detectors.shape), tuple(sample.observables.shape)
    )
    try:
        model.to_stim_text()
    except ValueError:
        print("Independent-only text export: refused")


if __name__ == "__main__":
    main()
