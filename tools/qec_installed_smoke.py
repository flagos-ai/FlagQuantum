"""Exercise public QEC workflows in a real isolated installation, outside source."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

PROFILES = {
    "core": (False, False),
    "stim": (True, False),
    "pymatching": (False, True),
    "both": (True, True),
}
SOURCE = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"


def _sample_cell(text: str, *, matching: bool) -> dict[str, object]:
    from flagquantum.qec import DetectorErrorModel, PyMatchingDecoder
    from flagquantum.qec.adapters.stim import (
        compile_stim_sampler,
        stim_circuit_to_dem_text,
    )

    model = DetectorErrorModel.from_stim_text(stim_circuit_to_dem_text(text))
    decoder = (
        PyMatchingDecoder.from_detector_error_model(model.merge_duplicate_mechanisms())
        if matching
        else None
    )
    completed, failures = 0, 0
    with compile_stim_sampler(text, seed=53) as sampler:
        for shots in (16, 16, 3):
            sample = sampler.sample(shots=shots)
            assert sample.shots == shots
            assert sample.detectors.shape == (shots, model.num_detectors)
            assert sample.observables.shape == (shots, model.num_observables)
            if decoder is not None:
                predicted = decoder.decode_batch(sample.detectors)
                assert predicted.shape == sample.observables.shape
                failures += int((predicted != sample.observables).any(dim=1).sum())
            completed += shots
    assert completed == 35
    sampler.close()
    try:
        sampler.sample(shots=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed sampler remained usable")
    return {
        "shots": completed,
        "detectors": model.num_detectors,
        "observables": model.num_observables,
        "logical_failures": failures if matching else None,
    }


def _exercise(profile: str) -> list[dict[str, object]]:
    import torch

    from flagquantum import qec
    from flagquantum.qec.adapters.stim import (
        compile_stim_sampler,
        memory_to_stim_text,
        stim_circuit_to_dem_text,
    )
    from flagquantum.simulation.stabilizer import StabilizerDependencyError

    stim, matching = PROFILES[profile]
    model = qec.DetectorErrorModel(1, 1, (qec.DemError(0.1, (0,), (0,)),))
    sample = model.dem_sampling(shots=35, seed=53)
    assert sample.detectors.shape == sample.observables.shape == (35, 1)
    native = qec.MinimumWeightMatchingDecoder.from_detector_error_model(model)
    rows = torch.tensor([[0], [1]])
    assert native.decode(()).observables == ()
    assert native.decode((0,)).observables == (0,)
    if matching:
        decoder = qec.PyMatchingDecoder.from_detector_error_model(model)
        assert decoder.decode_batch(rows).tolist() == [[False], [True]]
    else:
        try:
            qec.PyMatchingDecoder.from_detector_error_model(model)
        except qec.MatchingDependencyError:
            pass
        else:
            raise AssertionError("missing PyMatching did not fail explicitly")

    memory = qec.build_memory_circuit(qec.RepetitionCode(distance=3), rounds=3)
    if not stim:
        for operation in (compile_stim_sampler, stim_circuit_to_dem_text):
            try:
                operation(SOURCE)
            except StabilizerDependencyError:
                pass
            else:
                raise AssertionError("missing Stim did not fail explicitly")
        try:
            memory_to_stim_text(memory, noise=qec.PhenomenologicalNoise())
        except StabilizerDependencyError:
            return []
        raise AssertionError("native export ignored missing Stim")

    surface = qec.RotatedSurfaceCode(distance=3)
    experiments = (
        ("repetition_z", memory),
        ("surface_z", qec.build_memory_circuit(surface, rounds=3)),
        (
            "surface_x",
            qec.build_memory_circuit(
                surface, rounds=3, product=qec.Pauli(x_qubits=surface.data_qubits)
            ),
        ),
    )
    records = []
    for name, memory in experiments:
        with qec.compile_memory_sampler(
            memory, noise=qec.PhenomenologicalNoise(), seed=53
        ) as sampler:
            sample = sampler.sample(shots=2)
            assert not bool(sample.detectors.any())
            assert not bool(sample.observables.any())
        text = memory_to_stim_text(
            memory,
            noise=qec.PhenomenologicalNoise(
                data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
            ),
        )
        records.append({"entrance": name, **_sample_cell(text, matching=matching)})
    records.append({"entrance": "external", **_sample_cell(SOURCE, matching=matching)})
    return records


def _exercise_stim_experiment(profile: str) -> dict[str, object] | None:
    from flagquantum.qec.execution.experiments import run_stim_experiment
    from flagquantum.simulation.stabilizer import StabilizerDependencyError

    source = (
        "R 0 1\nX_ERROR(0.125) 0\nX_ERROR(0.25) 1\nM 0 1\n"
        "DETECTOR rec[-2]\nDETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-2]\n"
    )
    stim_available, matching = PROFILES[profile]
    decoder = "pymatching" if stim_available and matching else "minimum_weight_matching"
    if not stim_available:
        try:
            run_stim_experiment(
                source, decoder=decoder, shots=35, batch_size=16, seed=53
            )
        except StabilizerDependencyError:
            return None
        raise AssertionError("missing Stim did not refuse the experiment")

    result = run_stim_experiment(
        source, decoder=decoder, shots=35, batch_size=16, seed=53
    )
    assert result.shots == 35 and result.logical_failures == 0
    assert result.observable_failures == (0, 0)
    assert result.logical_error_rate == 0.0
    assert result.sampling_source == "stim_circuit"
    assert result.seed == 53
    assert result.confidence_interval[0] == 0.0
    assert 0.0 < result.confidence_interval[1] < 1.0

    # Independent Stim buffers must carry the swapped logical columns. The
    # report comparison is based on the analytic two-independent-fault law.
    import stim

    reference = stim.Circuit(source).compile_detector_sampler(seed=53)
    for size in (16, 16, 3):
        detectors, observables = reference.sample(shots=size, separate_observables=True)
        assert (detectors[:, ::-1] == observables).all()
    # The second logical has ideal raw value one but no logical *flip*.
    # Treating measurement bits as flips would create a false logical failure.
    reference_ones = (
        "R 0 1\nX 1\nX_ERROR(0.125) 0\nM 0 1\n"
        "DETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
        "OBSERVABLE_INCLUDE(1) rec[-1]\n"
    )
    ideal = stim.Circuit(reference_ones).compile_sampler(seed=53).sample(shots=1)
    assert bool(ideal[0, 1])
    normalized = run_stim_experiment(
        reference_ones, decoder=decoder, shots=35, batch_size=16, seed=53
    )
    assert normalized.logical_failures == 0
    assert normalized.observable_failures == (0, 0)
    assert normalized.sampling_source == "stim_circuit"
    return {
        "shots": result.shots,
        "logical_failures": result.logical_failures,
        "observable_failures": list(result.observable_failures),
        "sampling_source": result.sampling_source,
        "decoder": decoder,
        "reference_ones_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    args = parser.parse_args()
    expected = PROFILES[args.profile]
    available = tuple(
        importlib.util.find_spec(name) is not None for name in ("stim", "pymatching")
    )
    assert available == expected

    import torch

    import flagquantum
    from flagquantum import qec
    from flagquantum.qec.adapters import stim as bridge
    from flagquantum.simulation.native_cpu import adjoint

    assert "pymatching" in qec.decoder_names()
    assert callable(bridge.memory_to_stim_text)
    assert "stim" not in sys.modules and "pymatching" not in sys.modules
    assert sys.prefix != sys.base_prefix
    assert (
        Path(flagquantum.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    )
    assert not Path.cwd().resolve().is_relative_to(Path(__file__).resolve().parents[1])
    assert (
        "include-system-site-packages = false"
        in (Path(sys.prefix) / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    )
    assert adjoint.native_cpu_adjoint_available()
    records = _exercise(args.profile)
    experiment = _exercise_stim_experiment(args.profile)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "package": flagquantum.__file__,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "availability": dict(
                    zip(("stim", "pymatching"), available, strict=True)
                ),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name, present in zip(
                        ("stim", "pymatching"), expected, strict=True
                    )
                    if present
                },
                "cells": records,
                "stim_experiment": experiment,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
