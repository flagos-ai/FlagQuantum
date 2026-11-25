"""Sample a noisy repetition parity circuit and decode its exact DEM."""

from pathlib import Path
from tempfile import TemporaryDirectory

import flagquantum as fq
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
    source = (
        DynamicCircuit(1)
        .x(0)
        .measure(0, classical_bit=0)
        .reset(0)
        .measure(0, classical_bit=0)
        .bit_flip(0, 1.0)
        .measure(0, classical_bit=0)
    )
    native = fq.Circuit.from_ir(source.to_ir())
    writer = native.compile_detector_sampler(
        detector_map=MeasurementMap(3, ((0, 1), (1, 2))),
        observable_map=MeasurementMap(3, ((2,),)),
        seed=7,
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "events.b8"
        writer.sample_write(64, filepath=path, format="b8", append_observables=True)
        assert path.read_bytes() == b"\x06" * 64
        print("detector_bytes", path.stat().st_size)


if __name__ == "__main__":
    main()
