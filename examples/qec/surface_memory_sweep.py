"""Report eight fixed-shot CPU Surface configurations using existing QEC APIs.

Install the existing ``stim,pymatching`` extras. Output is illustrative JSON
lines, not a library persistence schema. Intervals are marginal per point.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from importlib.metadata import version

from flagquantum.qec import (
    Pauli,
    PhenomenologicalNoise,
    RotatedSurfaceCode,
    build_memory_circuit,
)
from flagquantum.qec.execution.experiments import (
    MemoryExperimentResult,
    run_memory_experiment,
)

_SEEDS = {
    ("x", 3, "low"): 53,
    ("x", 3, "high"): 54,
    ("x", 5, "low"): 55,
    ("x", 5, "high"): 56,
    ("z", 3, "low"): 57,
    ("z", 3, "high"): 58,
    ("z", 5, "low"): 59,
    ("z", 5, "high"): 60,
}
_NOISE = {"low": (0.005, 0.007, 0.009), "high": (0.01, 0.014, 0.018)}


def _points() -> tuple[tuple[str, int, str, int], ...]:
    return tuple((*key, seed) for key, seed in _SEEDS.items())


def _configuration(
    basis: str, distance: int, profile: str, seed: int
) -> dict[str, object]:
    data, phase, measurement = _NOISE[profile]
    return dict(
        basis=basis,
        distance=distance,
        noise_profile=profile,
        noise=dict(
            data_flip=data,
            phase_flip=phase,
            measurement_flip=measurement,
            both_flip=0.0,
        ),
        seed=seed,
        rounds=3,
        shots=35,
        batch_size=16,
        decoder="pymatching",
    )


def _validate_points(points: tuple[tuple[str, int, str, int], ...]) -> None:
    keys = [(basis, distance, profile) for basis, distance, profile, _ in points]
    if len(keys) != 8 or len(set(keys)) != 8 or set(keys) != set(_SEEDS):
        raise ValueError("the example requires all eight unique fixed configurations")
    for basis, distance, profile, seed in points:
        if type(seed) is not int or seed != _SEEDS[(basis, distance, profile)]:
            raise ValueError("a point must retain its declared seed")


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in ("torch", "stim", "pymatching")},
    }


def _result_fields(result: MemoryExperimentResult, seed: int) -> dict[str, object]:
    if (
        type(result.shots) is not int
        or result.shots != 35
        or type(result.logical_failures) is not int
        or not 0 <= result.logical_failures <= 35
        or not isinstance(result.observable_failures, tuple)
        or len(result.observable_failures) != 1
        or type(result.observable_failures[0]) is not int
        or result.observable_failures != (result.logical_failures,)
        or type(result.seed) is not int
        or result.seed != seed
        or result.sampling_source != "memory_circuit"
        or type(result.logical_error_rate) is not float
        or result.logical_error_rate != result.logical_failures / 35
    ):
        raise ValueError(
            "experiment returned inconsistent fixed-shot counts or provenance"
        )
    if (
        not isinstance(result.confidence_interval, tuple)
        or len(result.confidence_interval) != 2
    ):
        raise ValueError("experiment returned an invalid interval carrier")
    low, high = result.confidence_interval
    if not (
        math.isfinite(low)
        and math.isfinite(high)
        and 0 <= low <= result.logical_error_rate <= high <= 1
        and (result.logical_failures != 0 or high > 0)
    ):
        raise ValueError("experiment returned an invalid marginal interval")
    return dict(
        completed_shots=result.shots,
        logical_failures=result.logical_failures,
        logical_error_rate=result.logical_error_rate,
        observable_failures=result.observable_failures,
        confidence_interval=result.confidence_interval,
        sampling_source=result.sampling_source,
    )


def main() -> None:
    """Run the declared grid once; only full success emits completion."""
    points = _points()
    _validate_points(points)
    configurations = [_configuration(*point) for point in points]
    print(
        json.dumps(
            dict(
                record_type="configuration",
                code="rotated_surface",
                versions=_versions(),
                points=configurations,
            ),
            allow_nan=False,
        ),
        flush=True,
    )
    for point, configuration in zip(points, configurations, strict=True):
        basis, distance, profile, seed = point
        try:
            code = RotatedSurfaceCode(distance=distance)
            product = Pauli(x_qubits=code.data_qubits) if basis == "x" else None
            memory = build_memory_circuit(code, rounds=3, product=product)
            data, phase, measurement = _NOISE[profile]
            result = run_memory_experiment(
                memory,
                noise=PhenomenologicalNoise(
                    data_flip=data, phase_flip=phase, measurement_flip=measurement
                ),
                decoder="pymatching",
                shots=35,
                batch_size=16,
                seed=seed,
            )
            fields = _result_fields(result, seed)
            print(
                json.dumps(
                    dict(record_type="point", **configuration, **fields),
                    allow_nan=False,
                ),
                flush=True,
            )
        except BaseException as error:
            # The existing experiment owns sampler cleanup; retain its exception.
            print(
                json.dumps(
                    dict(
                        record_type="error",
                        point=configuration,
                        error_type=type(error).__name__,
                    )
                ),
                file=sys.stderr,
                flush=True,
            )
            raise
    print(
        json.dumps(dict(record_type="complete", points=8, total_requested_shots=280)),
        flush=True,
    )


if __name__ == "__main__":
    main()
