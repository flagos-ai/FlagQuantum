"""Convert explicit dynamic measurement records, preserving repeated-slot columns."""

import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap


def main() -> None:
    circuit = (
        DynamicCircuit(3)
        .h(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .reset(0)
        .conditional("x", 0, classical_bit=0, equals=0)
        .measure(0, classical_bit=0)
        .conditional("x", 2, classical_bit=0)
        .measure(1, classical_bit=1)
        .measure(2, classical_bit=2)
    )
    converter = circuit.compile_m2d_converter(
        detector_map=MeasurementMap(4, ((0, 1), (0, 2))),
        observable_map=MeasurementMap(4, ((1, 3),)),
    )
    records = circuit.compile_sampler(seed=7).sample(shots=32)
    detectors, observables = converter.convert(
        measurements=records, separate_observables=True
    )
    assert converter.measurement_qubits == (0, 0, 1, 2)
    assert not detectors.any() and not observables.any()
    injected = torch.tensor([[True, False, False, False], [True, False, True, True]])
    dets, obs = converter.convert(measurements=injected, separate_observables=True)
    assert dets.tolist() == [[False, True], [False, False]]
    assert obs.tolist() == [[False], [True]]
    print("record qubits", converter.measurement_qubits)
    print("records", tuple(records.shape))
    print("detectors", tuple(detectors.shape), int(detectors.sum()))
    print("observables", tuple(observables.shape), int(observables.sum()))
    print("injected flips verified")


if __name__ == "__main__":
    main()
