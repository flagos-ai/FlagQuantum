"""Report paired native matching and PyMatching outcomes on 35 Surface shots."""

from __future__ import annotations

import json
from numbers import Integral

from flagquantum.errors import ValidationError
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)
from flagquantum.qec.circuits.memory import build_memory_circuit
from flagquantum.qec.codes.surface import RotatedSurfaceCode
from flagquantum.qec.decoders.batch import decode_batch
from flagquantum.qec.decoders.registry import get_decoder
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.execution.experiments import _check_bits
from flagquantum.qec.noise import PhenomenologicalNoise

_DECODERS = ("minimum_weight_matching", "pymatching")
_BATCHES = (16, 16, 3)
_BINS = ("both_correct", "pymatching_only_wrong", "native_only_wrong", "both_wrong")
_SEED = 53


def _predict(indices: tuple[int, ...]) -> bool:
    if (
        not isinstance(indices, tuple)
        or len(indices) > 1
        or any(
            isinstance(index, bool) or not isinstance(index, Integral) or index != 0
            for index in indices
        )
    ):
        raise ValidationError("decoder returned invalid observable indices")
    return bool(indices)


def _report() -> dict[str, object]:
    memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
    rates = dict(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009, both_flip=0.0
    )
    text = memory_to_stim_text(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=rates["data_flip"],
            phase_flip=rates["phase_flip"],
            measurement_flip=rates["measurement_flip"],
            both_flip=rates["both_flip"],
        ),
    )
    model = DetectorErrorModel.from_stim_text(
        stim_circuit_to_dem_text(text, decompose_errors=False)
    )
    detectors = len(memory.detectors)
    if (model.num_detectors, model.num_observables) != (detectors, 1) or len(
        memory.observables
    ) != 1:
        raise ValidationError("strict model dimensions do not match the memory source")
    native, matching = (get_decoder(name, model) for name in _DECODERS)
    decode_batch(native, ())
    decode_batch(matching, ())
    counts = dict.fromkeys(_BINS, 0)
    sampler = compile_stim_sampler(text, seed=_SEED)
    try:
        for size in _BATCHES:
            batch = sampler.sample(shots=size)
            _check_bits(batch.detectors, size, detectors)
            _check_bits(batch.observables, size, 1)
            rows = tuple(
                tuple(row.nonzero().flatten().tolist()) for row in batch.detectors
            )
            left = decode_batch(native, rows)
            right = decode_batch(matching, rows)
            if len(left) != size or len(right) != size:
                raise ValidationError(
                    "decoder returned an unexpected number of results"
                )
            for first, second, observed in zip(
                left, right, batch.observables[:, 0], strict=True
            ):
                native_wrong = _predict(first.observables) != bool(observed)
                matching_wrong = _predict(second.observables) != bool(observed)
                counts[_BINS[2 * native_wrong + matching_wrong]] += 1
    except BaseException as primary:
        try:
            sampler.close()
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    else:
        sampler.close()
    shots = sum(_BATCHES)
    if sum(counts.values()) != shots:
        raise ValidationError("paired counts do not cover the completed shots")
    return {
        "basis": "z",
        "distance": 3,
        "rounds": 3,
        "noise": rates,
        "seed": _SEED,
        "batch_sizes": _BATCHES,
        "shots": shots,
        "decoders": _DECODERS,
        "counts": counts,
    }


def main() -> None:
    """Write one JSON record after both decoders complete all batches."""
    print(json.dumps(_report(), sort_keys=True))


if __name__ == "__main__":
    main()
