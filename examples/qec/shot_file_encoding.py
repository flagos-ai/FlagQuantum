"""Write packed QEC records and decode them in bounded batches."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder
from flagquantum.qec.execution.shot_data import (
    iter_shot_data_file,
    read_shot_data_file,
    write_shot_data_file,
)


def main() -> None:
    model = DetectorErrorModel(1, 1, (DemError(0.1, (0,), (0,)),))
    decoder = PyMatchingDecoder.from_detector_error_model(model)
    data = torch.tensor([[0], [3]] * 32, dtype=torch.uint8)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "events.b8"
        write_shot_data_file(
            data=data,
            path=path,
            format="b8",
            num_detectors=1,
            num_observables=1,
        )
        assert path.read_bytes() == b"\x00\x03" * 32

        detectors, logicals = read_shot_data_file(
            path=path,
            format="b8",
            num_detectors=1,
            num_observables=1,
            separate_observables=True,
            bit_packed=True,
        )
        assert torch.equal(detectors, logicals)

        shots = batches = failures = 0
        with closing(
            iter_shot_data_file(
                path=path,
                format="b8",
                num_detectors=1,
                num_observables=1,
                separate_observables=True,
                bit_packed=True,
                batch_size=17,
            )
        ) as records:
            for detector_batch, logical_batch in records:
                predictions = decoder.decode_packed_batch(detector_batch)
                failures += int((predictions != logical_batch).any(dim=1).sum().item())
                shots += len(detector_batch)
                batches += 1

        assert (shots, batches, failures) == (64, 4, 0)
        print("shots", shots, "batches", batches, "logical_failures", failures)


if __name__ == "__main__":
    main()
