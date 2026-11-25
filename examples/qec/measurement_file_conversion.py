"""Convert external occurrence-ordered records without an all-shot tensor."""

from pathlib import Path
from tempfile import TemporaryDirectory

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap


def main() -> None:
    source = (
        DynamicCircuit(1)
        .measure(0, classical_bit=0)
        .measure(0, classical_bit=0)
        .conditional("x", 0, classical_bit=0)
        .measure(0, classical_bit=0)
    )
    converter = fq.Circuit.from_ir(source.to_ir()).compile_m2d_converter(
        detector_map=MeasurementMap(3, ((0, 1), (0, 2))),
        observable_map=MeasurementMap(3, ((1,),)),
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        records, events, logical = (
            root / "measurements.b8",
            root / "events.b8",
            root / "observables.b8",
        )
        # Fixed external fault records (m0, m1, m2) = (0, 1, 0); feedback reads m1.
        records.write_bytes(b"\x02" * 64)
        converter.convert_file(
            measurements_filepath=records,
            measurements_format="b8",
            detection_events_filepath=events,
            detection_events_format="b8",
            obs_out_filepath=logical,
            obs_out_format="b8",
        )
        assert events.read_bytes() == logical.read_bytes() == b"\x01" * 64
        print("shots 64")
        print("detectors 10; observables 1")
        print("file_bytes", events.stat().st_size, logical.stat().st_size)


if __name__ == "__main__":
    main()
