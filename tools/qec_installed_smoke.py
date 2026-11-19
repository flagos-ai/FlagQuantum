"""Exercise public QEC workflows in a real isolated installation, outside source."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypedDict

PROFILES = {
    "core": (False, False),
    "stim": (True, False),
    "pymatching": (False, True),
    "both": (True, True),
}
SOURCE = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"


class _MemoryExperimentAcceptance(TypedDict):
    entrance: str
    shots: int
    logical_failures: int
    sampling_source: Literal["memory_circuit"]


class _StimExperimentAcceptance(TypedDict):
    shots: int
    logical_failures: int
    observable_failures: list[int]
    sampling_source: Literal["stim_circuit"]
    decoder: str
    reference_ones_verified: bool
    memory_experiments: list[_MemoryExperimentAcceptance]


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


def _exercise_stim_experiment(profile: str) -> _StimExperimentAcceptance | None:
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
    try:
        import stim
    except ModuleNotFoundError as exc:
        raise StabilizerDependencyError(
            "Stim is required for the independent reference"
        ) from exc

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
    memory_reports = _exercise_memory_experiments(decoder)
    return {
        "shots": result.shots,
        "logical_failures": result.logical_failures,
        "observable_failures": list(result.observable_failures),
        "sampling_source": result.sampling_source,
        "decoder": decoder,
        "reference_ones_verified": True,
        "memory_experiments": memory_reports,
    }


def _exercise_memory_experiments(decoder: str) -> list[_MemoryExperimentAcceptance]:
    from flagquantum import qec
    from flagquantum.qec.execution.experiments import run_memory_experiment

    surface = qec.RotatedSurfaceCode(distance=3)
    entrances = (
        (
            "repetition_z",
            qec.build_memory_circuit(qec.RepetitionCode(distance=3), rounds=1),
        ),
        ("surface_z", qec.build_memory_circuit(surface, rounds=3)),
        (
            "surface_x",
            qec.build_memory_circuit(
                surface, rounds=3, product=qec.Pauli(x_qubits=surface.data_qubits)
            ),
        ),
    )
    records: list[_MemoryExperimentAcceptance] = []
    for name, memory in entrances:
        result = run_memory_experiment(
            memory,
            noise=qec.PhenomenologicalNoise(
                data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
            ),
            decoder=decoder,
            shots=35,
            batch_size=16,
            seed=53,
        )
        assert result.shots == 35 and result.seed == 53
        assert 0 <= result.logical_failures <= result.shots
        assert result.observable_failures == (result.logical_failures,)
        assert result.logical_error_rate == result.logical_failures / result.shots
        assert (
            result.confidence_interval[0]
            <= result.logical_error_rate
            <= result.confidence_interval[1]
        )
        assert result.sampling_source == "memory_circuit"
        records.append(
            {
                "entrance": name,
                "shots": result.shots,
                "logical_failures": result.logical_failures,
                "sampling_source": result.sampling_source,
            }
        )
    return records


def _exercise_decoder_records(profile: str) -> list[str]:
    import torch

    from flagquantum import qec
    from flagquantum.qec.decoders.batch import decode_batch

    model = qec.DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L1\nerror(0.25) D1 L0\ndetector D0\ndetector D1\n"
        "logical_observable L0\nlogical_observable L1\n"
    )
    names = [
        "minimum_weight_matching",
        "sliding_window_matching",
        "belief_propagation",
        "belief_propagation_osd",
    ]
    if PROFILES[profile][1]:
        names.append("pymatching")
    rows = ((), (0,), (1,), (0, 1))
    expected = ((), (1,), (0,), (0, 1))
    for name in names:
        decoder = qec.get_decoder(name, model)
        records = decode_batch(decoder, rows)
        assert records == tuple(decoder.decode(row) for row in rows)
        assert tuple(record.observables for record in records) == expected
    if PROFILES[profile][1]:
        matching = qec.PyMatchingDecoder.from_detector_error_model(model)
        dense = matching.decode_batch(torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]))
        assert dense.tolist() == [
            [False, False],
            [False, True],
            [True, False],
            [True, True],
        ]
    return names


def _exercise_packed_predictions(profile: str) -> list[list[int]] | None:
    if not PROFILES[profile][1]:
        return None
    import torch

    from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder

    model = DetectorErrorModel(
        3,
        3,
        (
            DemError(0.1, (0,), (0,)),
            DemError(0.2, (1,), (1,)),
            DemError(0.3, (2,), (2,)),
        ),
    )
    decoder = PyMatchingDecoder.from_detector_error_model(model)
    result = decoder.decode_packed_batch(
        torch.tensor([[0], [1], [2], [4], [7]], dtype=torch.uint8)
    )
    expected = [[0], [1], [2], [4], [7]]
    assert result.tolist() == expected
    return expected


def _exercise_sdk_weights(profile: str) -> dict[str, object] | None:
    if not PROFILES[profile][1]:
        return None
    import torch

    from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder

    decoder = PyMatchingDecoder.from_detector_error_model(
        DetectorErrorModel(3, 3, tuple(DemError(0.125, (i,), (i,)) for i in range(3)))
    )
    dense = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.uint8)
    for packed_in in (False, True):
        events = (
            torch.tensor([[0], [1], [7]], dtype=torch.uint8) if packed_in else dense
        )
        for packed_out in (False, True):
            bits, costs = decoder.decode_batch_with_weights(
                events, bit_packed_shots=packed_in, bit_packed_predictions=packed_out
            )
            expected = [[0], [1], [7]] if packed_out else dense.bool().tolist()
            assert bits.tolist() == expected
            assert costs.dtype == torch.float64 and costs.shape == (3,)
            assert bool(torch.isfinite(costs).all()) and costs[0] == 0 and costs[1] > 0
    return {"shots": 3, "layouts": 4, "weight_source": "sdk_matching"}


def _exercise_matched_detectors(profile: str) -> list[list[int | None]] | None:
    if not PROFILES[profile][1]:
        return None
    from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder

    model = DetectorErrorModel(
        3, 1, (DemError(0.125, (0,), (0,)), DemError(0.1, (1, 2), ()))
    )
    pairs = PyMatchingDecoder.from_detector_error_model(
        model
    ).decode_to_matched_detectors((0, 1, 2))
    assert pairs == ((0, None), (1, 2))
    return [[first, second] for first, second in pairs]


def _exercise_stim_conversion(profile: str) -> dict[str, object] | None:
    import torch

    from flagquantum.qec.execution.measurement_conversion import (
        convert_measurement_records,
    )
    from flagquantum.simulation.stabilizer import StabilizerDependencyError

    text = "R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    raw = torch.tensor([[1], [0]], dtype=torch.uint8)
    if not PROFILES[profile][0]:
        try:
            convert_measurement_records(
                text, raw, circuit_format="stim", bit_packed_measurements=True
            )
        except StabilizerDependencyError:
            return None
        raise AssertionError("conversion ignored missing Stim")
    batch = convert_measurement_records(
        text, raw, circuit_format="stim", bit_packed_measurements=True
    )
    assert batch.detectors.tolist() == batch.observables.tolist() == [[False], [True]]
    return {
        "detectors": batch.detectors.tolist(),
        "observables": batch.observables.tolist(),
    }


def _embedded_profile() -> bool:
    from flagquantum.simulation.stabilizer import measurement_sampler

    return (
        Path(measurement_sampler.__file__)
        .with_name("_embedded_stim_profile.txt")
        .is_file()
    )


def _exercise_compiled_measurement_sampler(profile: str) -> dict[str, object] | None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.simulation.stabilizer import StabilizerDependencyError

    circuit = DynamicCircuit(1).x(0).measure(0).reset(0).measure(0)
    try:
        sampler = circuit.compile_sampler(seed=7)
    except StabilizerDependencyError:
        if PROFILES[profile][0]:
            raise
        return None
    assert (
        PROFILES[profile][0] or _embedded_profile()
    ), "prepared sampler ignored missing Stim"
    samples = sampler.sample(2, bit_packed=True)
    assert samples.tolist() == [[1], [1]]
    with TemporaryDirectory() as directory:
        output = Path(directory) / "records.b8"
        sampler.sample_write(2, filepath=output, format="b8")
        packed = output.read_bytes()
        assert packed == b"\x01\x01"
        formats: dict[Literal["01", "b8", "r8", "hits", "dets", "ptb64"], bytes] = {
            "01": b"10\n" * 64,
            "b8": b"\x01" * 64,
            "r8": b"\x00\x01" * 64,
            "hits": b"0\n" * 64,
            "dets": b"shot M0\n" * 64,
            "ptb64": b"\xff" * 8 + b"\x00" * 8,
        }
        for format, reference in formats.items():
            output = Path(directory) / f"format-{format}"
            sampler.sample_write(64, filepath=output, format=format)
            assert output.read_bytes() == reference
    return {
        "packed_samples": samples.tolist(),
        "b8_hex": packed.hex(),
        "file_format_sizes": {format: len(data) for format, data in formats.items()},
    }


def _exercise_native_detector_execution(profile: str) -> dict[str, object] | None:
    if not PROFILES[profile][0] and not _embedded_profile():
        return None
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.qec.execution.context import MeasurementMap

    circuit = DynamicCircuit(1).x(0).bit_flip(0, 1.0).measure(0)
    maps = MeasurementMap(1, ((0,),))
    sampler = circuit.compile_detector_sampler(
        detector_map=maps, observable_map=maps, seed=7
    )
    packed = sampler.sample(2, append_observables=True, bit_packed=True)
    import torch

    assert isinstance(packed, torch.Tensor)
    assert packed.tolist() == [[3], [3]]
    witnesses: dict[Literal["01", "b8", "r8", "hits", "dets", "ptb64"], bytes] = {
        "01": b"11\n" * 64,
        "b8": b"\x03" * 64,
        "r8": b"\x00\x00\x00" * 64,
        "hits": b"0,1\n" * 64,
        "dets": b"shot D0 L0\n" * 64,
        "ptb64": b"\xff" * 16,
    }
    with TemporaryDirectory() as directory:
        for encoding, witness in witnesses.items():
            path = Path(directory) / encoding
            sampler.sample_write(
                64, filepath=path, format=encoding, append_observables=True
            )
            assert path.read_bytes() == witness
    model = circuit.detector_error_model(detector_map=maps, observable_map=maps)
    assert model.num_detectors == model.num_observables == 1
    assert len(model.errors) == 1 and model.errors[0].probability == 1.0
    return {
        "packed_detector_observable": packed.tolist(),
        "mechanisms": len(model.errors),
        "detector_file_sizes": {
            encoding: len(data) for encoding, data in witnesses.items()
        },
    }


def _exercise_measurement_file_conversion(profile: str) -> dict[str, object] | None:
    if not PROFILES[profile][0] and not _embedded_profile():
        return None
    import flagquantum as fq
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.qec.execution.context import MeasurementMap

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
    inputs: dict[Literal["01", "b8", "r8", "ptb64", "hits", "dets"], bytes] = {
        "01": b"010\n" * 64,
        "b8": b"\x02" * 64,
        "r8": b"\x01\x01" * 64,
        "hits": b"1\n" * 64,
        "dets": b"shot M1\n" * 64,
        "ptb64": b"\0" * 8 + b"\xff" * 8 + b"\0" * 8,
    }
    outputs: dict[Literal["01", "b8", "r8", "hits", "dets"], bytes] = {
        "01": b"101\n" * 64,
        "b8": b"\x05" * 64,
        "r8": b"\x00\x01\x00" * 64,
        "hits": b"0,2\n" * 64,
        "dets": b"shot D0 L0\n" * 64,
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for input_format, raw in inputs.items():
            source_path = root / ("input." + input_format)
            source_path.write_bytes(raw)
            for output_format, expected in outputs.items():
                target = root / (input_format + "." + output_format)
                logical = root / (input_format + "." + output_format + ".obs")
                converter.convert_file(
                    measurements_filepath=source_path,
                    measurements_format=input_format,
                    detection_events_filepath=target,
                    detection_events_format=output_format,
                    append_observables=True,
                    obs_out_filepath=logical,
                    obs_out_format="b8",
                )
                assert (
                    target.read_bytes() == expected
                    and logical.read_bytes() == b"\x01" * 64
                )
    return {
        "shots_per_case": 64,
        "input_formats": list(inputs),
        "output_formats": list(outputs),
        "byte_cases": len(inputs) * len(outputs),
        "occurrence_columns": 3,
        "appended_and_separate": True,
    }


def _exercise_terminal_conversion(profile: str) -> dict[str, object] | None:
    import torch

    import flagquantum as fq
    from flagquantum.qec.execution.context import MeasurementMap
    from flagquantum.qec.execution.measurement_conversion import compile_m2d_converter
    from flagquantum.simulation.stabilizer import StabilizerDependencyError

    try:
        converter = compile_m2d_converter(
            fq.Circuit(1).x(0),
            measurement_qubits=(0,),
            detector_map=MeasurementMap(1, ((0,),)),
            observable_map=MeasurementMap(1, ()),
        )
    except StabilizerDependencyError:
        if PROFILES[profile][0]:
            raise
        return None
    assert PROFILES[profile][0], "terminal conversion ignored missing Stim"
    results = []
    for value in (True, False):
        detectors, observables = converter.convert(
            measurements=torch.tensor([[value]]), separate_observables=True
        )
        assert detectors.tolist() == [[not value]]
        assert observables.shape == (1, 0)
        results.extend(detectors.tolist())
    return {"detectors": results, "observables": [[], []]}


def _exercise_matching_graph(profile: str) -> dict[str, object] | None:
    if not PROFILES[profile][1]:
        return None
    import torch

    from flagquantum.qec.adapters.pymatching_graph import PyMatchingGraphDecoder
    from flagquantum.qec.decoders.matching_graph import MatchingGraph

    checks = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
    faults = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.uint8)
    graph, columns = MatchingGraph.from_check_matrix(
        checks,
        faults_matrix=faults,
        weights=torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64),
    )
    solution = PyMatchingGraphDecoder(graph).decode((0, 1))
    assert columns == (0, 1, 2) and solution.edge_indices == (2,)
    assert solution.fault_ids == () and solution.weight == 1.0
    return {
        "column_to_edge": list(columns),
        "selected_edges": list(solution.edge_indices),
        "weight": solution.weight,
    }


def _exercise_shot_file_reading(profile: str) -> list[list[int]] | None:
    """Exercise file reading in real installed SDK and embedded profiles."""
    import torch

    from flagquantum.errors import CapabilityError
    from flagquantum.qec.execution.shot_data import read_shot_data_file

    try:
        read_shot_data_file(path="unused", format="b8", num_detectors=0)
    except CapabilityError:
        pass
    else:
        raise AssertionError("zero stored columns were admitted")
    if not PROFILES[profile][0] and not _embedded_profile():
        return None
    with TemporaryDirectory() as directory:
        path = Path(directory) / "events.b8"
        path.write_bytes(b"\x00\x01")
        result = read_shot_data_file(
            path=path, format="b8", num_detectors=1, bit_packed=True
        )
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [[0], [1]]
        return [[0], [1]]


def _exercise_shot_data_writing(profile: str) -> dict[str, object] | None:
    """Exercise all public shot-data encodings in an installed profile."""
    import torch

    from flagquantum.qec.execution.shot_data import write_shot_data_file

    if not PROFILES[profile][0] and not _embedded_profile():
        return None
    witnesses: dict[Literal["01", "b8", "r8", "ptb64", "hits", "dets"], bytes] = {
        "01": b"111\n" * 64,
        "b8": b"\x07" * 64,
        "r8": b"\x00\x00\x00\x00" * 64,
        "ptb64": b"\xff" * 24,
        "hits": b"0,1,2\n" * 64,
        "dets": b"shot D0 D1 D2\n" * 64,
    }
    with TemporaryDirectory() as directory:
        for encoding, expected in witnesses.items():
            path = Path(directory) / encoding
            write_shot_data_file(
                data=torch.ones((64, 3), dtype=torch.bool),
                path=path,
                format=encoding,
                num_detectors=3,
            )
            assert path.read_bytes() == expected
    return {"formats": list(witnesses), "shots_per_format": 64, "exact_bytes": True}


def _exercise_compiled_dem_sampler(profile: str) -> dict[str, object] | None:
    """Record/replay canonical error columns and stream three independent files."""
    import torch

    from flagquantum.qec import DemError, DetectorErrorModel

    if not PROFILES[profile][0] and not _embedded_profile():
        return None
    model = DetectorErrorModel(1, 1, (DemError(0.25, (0,), (0,)),))
    sampler = model.compile_sampler(seed=7)
    detectors, logicals, errors = sampler.sample(
        64, bit_packed=True, return_errors=True
    )
    assert errors is not None
    replayed, replayed_logicals, unused = sampler.sample(
        64, bit_packed=True, recorded_errors_to_replay=errors
    )
    assert torch.equal(detectors, replayed) and torch.equal(logicals, replayed_logicals)
    assert unused is None
    with TemporaryDirectory() as directory:
        root = Path(directory)
        replay = root / "input.b8"
        replay.write_bytes(bytes(errors[:, 0].tolist()))
        sampler.sample_write(
            64,
            det_out_file=root / "d",
            det_out_format="b8",
            obs_out_file=root / "o",
            obs_out_format="b8",
            err_out_file=root / "e",
            err_out_format="b8",
            replay_err_in_file=replay,
            replay_err_in_format="b8",
        )
        assert (
            (root / "d").read_bytes()
            == (root / "o").read_bytes()
            == (root / "e").read_bytes()
            == replay.read_bytes()
        )
    return {"shots": 64, "error_columns": 1, "exact_replay": True, "file_streams": 3}


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
    record_decoders = _exercise_decoder_records(args.profile)
    experiment = _exercise_stim_experiment(args.profile)
    print(
        json.dumps(
            {
                "shot_file_reading": _exercise_shot_file_reading(args.profile),
                "shot_data_writing": _exercise_shot_data_writing(args.profile),
                "compiled_dem_sampler": _exercise_compiled_dem_sampler(args.profile),
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
                "matching_graph": _exercise_matching_graph(args.profile),
                "packed_predictions": _exercise_packed_predictions(args.profile),
                "sdk_weighted_batches": _exercise_sdk_weights(args.profile),
                "matched_detectors": _exercise_matched_detectors(args.profile),
                "stim_experiment": experiment,
                "full_record_decoders": record_decoders,
                "native_detector_execution": _exercise_native_detector_execution(
                    args.profile
                ),
                "stim_conversion": _exercise_stim_conversion(args.profile),
                "measurement_file_conversion": _exercise_measurement_file_conversion(
                    args.profile
                ),
                "terminal_conversion": _exercise_terminal_conversion(args.profile),
                "compiled_measurement_sampler": _exercise_compiled_measurement_sampler(
                    args.profile
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
