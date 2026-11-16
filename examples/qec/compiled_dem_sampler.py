"""Sample, replay, write and decode an owned independent detector error model."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder
from flagquantum.qec.execution.shot_data import (
    iter_shot_data_file,
    write_shot_data_file,
)

model = DetectorErrorModel(1, 1, (DemError(0.1, (0,), (0,)),))
sampler = model.compile_sampler(seed=7)
sampler_data, logicals, error_bits = sampler.sample(
    64,
    bit_packed=True,
    return_errors=True,
)
assert error_bits is not None
replayed, replayed_logicals, unused = sampler.sample(
    64,
    bit_packed=True,
    recorded_errors_to_replay=error_bits,
)
assert torch.equal(sampler_data, replayed)
assert torch.equal(logicals, replayed_logicals) and unused is None

decoder = PyMatchingDecoder.from_detector_error_model(model)
with TemporaryDirectory() as directory:
    root = Path(directory)
    write_shot_data_file(
        data=error_bits,
        path=root / "input-errors.b8",
        format="b8",
        num_measurements=model.num_errors,
    )
    sampler.sample_write(
        64,
        det_out_file=root / "detectors.b8",
        det_out_format="b8",
        obs_out_file=root / "observables.b8",
        obs_out_format="b8",
        err_out_file=root / "replayed-errors.b8",
        err_out_format="b8",
        replay_err_in_file=root / "input-errors.b8",
        replay_err_in_format="b8",
    )
    shots = batches = failures = 0
    with closing(
        iter_shot_data_file(
            path=root / "detectors.b8",
            format="b8",
            num_detectors=1,
            bit_packed=True,
            batch_size=17,
        )
    ) as records:
        for events in records:
            assert isinstance(events, torch.Tensor)
            predictions = decoder.decode_packed_batch(events)
            failures += int(
                (predictions != logicals[shots : shots + len(events)])
                .any(dim=1)
                .sum()
                .item()
            )
            shots += len(events)
            batches += 1
    assert (shots, batches, failures) == (64, 4, 0)
    print("shots", shots, "batches", batches, "logical_failures", failures)
