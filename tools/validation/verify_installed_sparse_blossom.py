"""Smoke-test the installed QEC experiment with no PyMatching Python package."""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath

if __package__ in (None, ""):
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import math
from importlib.util import find_spec


def main() -> None:
    if find_spec("pymatching") is not None:
        raise RuntimeError(
            "Installed Sparse Blossom smoke requires no pymatching package"
        )
    import torch

    from flagquantum.experimental.qec import SparseBlossomDecoder
    from flagquantum.qec import (
        DecodingGraph,
        DemError,
        DetectorErrorModel,
        PhenomenologicalNoise,
        RotatedSurfaceCode,
        build_memory_circuit,
    )
    from flagquantum.qec.execution.experiments import (
        run_memory_experiment,
        run_stim_experiment,
    )

    model = DetectorErrorModel(1, 9, (DemError(0.1, (0,), (8,)),))
    packed = SparseBlossomDecoder(DecodingGraph.from_detector_error_model(model))
    prediction = packed.decode_packed_batch(
        torch.tensor([[0], [1], [0]], dtype=torch.uint8)
    )
    assert prediction.dtype == torch.uint8
    assert prediction.tolist() == [[0, 0], [0, 1], [0, 0]]

    weighted_model = DetectorErrorModel(
        9, 9, tuple(DemError(0.125, (i,), (8 - i,)) for i in range(9))
    )
    weighted = SparseBlossomDecoder(
        DecodingGraph.from_detector_error_model(weighted_model)
    )
    dense = torch.zeros((3, 9), dtype=torch.uint8)
    dense[1, (0, 8)] = 1
    dense[2, 8] = 1
    packed_events = torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.uint8)
    packed_expected = torch.tensor([[0, 0], [1, 1], [1, 0]], dtype=torch.uint8)
    for packed_input in (False, True):
        for packed_output in (False, True):
            bits, weights = weighted.decode_batch_with_weights(
                packed_events if packed_input else dense,
                bit_packed_shots=packed_input,
                bit_packed_predictions=packed_output,
            )
            expected = packed_expected if packed_output else dense.flip(1).bool()
            assert torch.equal(bits, expected)
            assert weights.dtype == torch.float64
            torch.testing.assert_close(
                weights,
                torch.tensor([0.0, 2 * math.log(7), math.log(7)], dtype=torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )

    source = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    options = dict(decoder="sparse_blossom", shots=137, batch_size=24, seed=23)
    stim = run_stim_experiment(source, **options)
    memory = run_memory_experiment(
        build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3),
        noise=PhenomenologicalNoise(data_flip=0.005, measurement_flip=0.005),
        **options,
    )
    assert stim.shots == memory.shots == 137 and stim.logical_failures == 0
    assert memory.sampling_source == "memory_circuit"
    print(
        "Installed Sparse Blossom decoded weighted/packed rows and both QEC experiments without pymatching."
    )


if __name__ == "__main__":
    main()
