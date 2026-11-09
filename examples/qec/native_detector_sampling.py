"""Sample a noisy repetition parity circuit and decode its exact DEM."""

from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec import PyMatchingDecoder
from flagquantum.qec.execution.context import MeasurementMap


def main() -> None:
    circuit = (
        DynamicCircuit(3)
        .x(0)
        .x(1)
        .x(2)
        .bit_flip(0, 0.1)
        .bit_flip(1, 0.1)
        .bit_flip(2, 0.1)
        .measure(0, classical_bit=0)
        .measure(1, classical_bit=1)
        .measure(2, classical_bit=2)
    )
    detector_map = MeasurementMap(3, ((0, 1), (1, 2)))
    observable_map = MeasurementMap(3, ((0,),))
    sampler = circuit.compile_detector_sampler(
        detector_map=detector_map, observable_map=observable_map, seed=7
    )
    model = circuit.detector_error_model(
        detector_map=detector_map, observable_map=observable_map
    )
    decoder = PyMatchingDecoder.from_detector_error_model(model)
    failures = 0
    for shots in (16, 16, 3):
        result = sampler.sample(shots, separate_observables=True)
        assert isinstance(result, tuple)
        detectors, observables = result
        failures += int(
            (decoder.decode_batch(detectors) ^ observables).any(dim=1).sum()
        )
    print("shots", 35)
    print("logical_failures", failures)
    print("logical_error_rate", failures / 35)


if __name__ == "__main__":
    main()
