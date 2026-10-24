"""Use either experiment entrance with the same owned sampler and decoder."""

from __future__ import annotations

from examples.qec.stim_pymatching_baseline import _collect_counts, _wilson_interval
from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    PyMatchingDecoder,
    RepetitionCode,
    build_memory_circuit,
)
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)


def main() -> None:
    """Compile original circuit text, then decode each sampled observable flip."""
    native = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    native_text = memory_to_stim_text(
        native, noise=PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)
    )
    external_text = """
        R 0
        X_ERROR(0.1) 0
        M 0
        DETECTOR(0) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    for name, text in (("native", native_text), ("external", external_text)):
        model = DetectorErrorModel.from_stim_text(stim_circuit_to_dem_text(text))
        decoder = PyMatchingDecoder.from_detector_error_model(
            model.merge_duplicate_mechanisms()
        )
        with compile_stim_sampler(text, seed=53) as sampler:
            completed, failures, _ = _collect_counts(
                sampler,
                decoder,
                shots=35,
                batch_size=16,
                observables=model.num_observables,
            )
        print(name, completed, model.num_detectors, model.num_observables, failures)
        print("fixed-shot Wilson 95%", _wilson_interval(failures, completed))


if __name__ == "__main__":
    main()
