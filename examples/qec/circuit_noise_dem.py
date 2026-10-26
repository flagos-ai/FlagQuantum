"""Build a circuit-location noise model beyond the dense statevector regime.

Run with an installed package or PYTHONPATH=. No Stim or PyMatching extra is
needed. This constructs a DEM; it does not execute a 49-qubit dense simulation.
"""

from flagquantum.noise import NoiseModel, bit_flip_channel
from flagquantum.qec import (
    DetectorErrorModel,
    RotatedSurfaceCode,
    build_memory_circuit,
)


def main() -> None:
    code = RotatedSurfaceCode(distance=5)
    memory = build_memory_circuit(code, rounds=3)
    noise = NoiseModel().add("cx", bit_flip_channel(0.01))
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = model.dem_sampling(shots=4, seed=0)
    print("Physical qubits:", code.num_data_qubits + code.num_ancilla_qubits)
    print("Detectors:", model.num_detectors)
    print("Logical observables:", model.num_observables)
    print(
        "Sample shapes:", tuple(sample.detectors.shape), tuple(sample.observables.shape)
    )


if __name__ == "__main__":
    main()
