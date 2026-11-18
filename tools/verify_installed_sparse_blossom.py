"""Smoke-test the installed QEC experiment with no PyMatching Python package."""

from __future__ import annotations

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
        "Installed Sparse Blossom decoded packed rows and both QEC experiments without pymatching."
    )


if __name__ == "__main__":
    main()
