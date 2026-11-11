"""Decode captured b8 records with an embedded build and the pymatching extra."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder
from flagquantum.qec.execution.shot_data import (
    iter_shot_data_file,
    read_shot_data_file,
)

model = DetectorErrorModel(1, 1, (DemError(0.1, (0,), (0,)),))
decoder = PyMatchingDecoder.from_detector_error_model(model)
with TemporaryDirectory() as directory:
    path = Path(directory) / "events.b8"
    # The two stored bits are D0 and L0; every other shot flips both.
    path.write_bytes(b"\x00\x03" * 32)
    eager_dets, eager_obs = read_shot_data_file(
        path=path,
        format="b8",
        num_detectors=1,
        num_observables=1,
        separate_observables=True,
        bit_packed=True,
    )
    assert eager_dets.dtype == eager_obs.dtype == torch.uint8
    assert eager_dets.shape == eager_obs.shape == (64, 1)
    assert torch.equal(eager_dets, eager_obs)
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
        for dets, observables in records:
            predicted = decoder.decode_packed_batch(dets)
            failures += int((predicted != observables).any(dim=1).sum())
            shots += len(dets)
            batches += 1
    assert (shots, batches, failures) == (64, 4, 0)
    print("shots", shots, "batches", batches, "logical_failures", failures)
