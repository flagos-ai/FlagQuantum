"""User workflows across native memory, annotated text and owned sampling."""

from __future__ import annotations

import copy
import inspect
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import flagquantum.qec as qec
from flagquantum.errors import CapabilityError
from flagquantum.noise import NoiseModel, two_qubit_depolarizing_channel
from flagquantum.qec.adapters import stim as bridge
from flagquantum.simulation.stabilizer import StabilizerDependencyError, engine

pytestmark = pytest.mark.unit

SOURCE = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"


def test_approved_opt_in_surface_only() -> None:
    contract = json.loads(
        (
            Path(__file__).parents[2] / "contracts/qec-stim-experiment-bridge-v1.json"
        ).read_text()
    )
    assert tuple(contract["signatures"]) == bridge.__all__
    for name, signature in contract["signatures"].items():
        assert str(inspect.signature(getattr(bridge, name))) == signature
        assert not hasattr(qec, name)


@pytest.mark.parametrize("operation", bridge.__all__)
def test_missing_engine_is_explicit_and_lazy(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> None:
        raise StabilizerDependencyError("install flagquantum[stim]")

    monkeypatch.setattr(engine, "_stim", missing)
    with pytest.raises(StabilizerDependencyError, match="stim"):
        if operation == "memory_to_stim_text":
            memory = qec.build_memory_circuit(qec.RepetitionCode(distance=3), rounds=1)
            bridge.memory_to_stim_text(memory, noise=qec.PhenomenologicalNoise())
        else:
            getattr(bridge, operation)(SOURCE)


@pytest.mark.parametrize("text", [None, 3, b"R 0"])
@pytest.mark.parametrize(
    "operation", ["compile_stim_sampler", "stim_circuit_to_dem_text"]
)
def test_invalid_source_types(text: object, operation: str) -> None:
    with pytest.raises(TypeError, match="string"):
        getattr(bridge, operation)(text)


@pytest.mark.parametrize("seed", [True, 1.5, "53"])
def test_seed_types_refused_before_engine(seed: object) -> None:
    with pytest.raises(TypeError):
        bridge.compile_stim_sampler(SOURCE, seed=seed)  # type: ignore[arg-type]


def test_explicit_matching_decomposition_preserves_the_raw_law() -> None:
    stim = pytest.importorskip("stim")
    text = str(
        stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=2,
            after_clifford_depolarization=0.02,
            before_measure_flip_probability=0.02,
            after_reset_flip_probability=0.02,
        )
    )
    raw = qec.DetectorErrorModel.from_stim_text(bridge.stim_circuit_to_dem_text(text))
    decomposed_text = bridge.stim_circuit_to_dem_text(text, decompose_errors=True)
    assert "^" in decomposed_text
    decomposed = qec.DetectorErrorModel.from_stim_text(decomposed_text)
    assert torch.allclose(
        raw.detector_rates(), decomposed.detector_rates(), atol=1e-12, rtol=0
    )
    assert torch.allclose(
        raw.observable_rates(), decomposed.observable_rates(), atol=1e-12, rtol=0
    )
    expected = stim.Circuit(text).compile_detector_sampler(seed=53)
    with bridge.compile_stim_sampler(text, seed=53) as handle:
        batch = handle.sample(shots=35)
    reference = torch.as_tensor(expected.sample(shots=35, append_observables=True))
    assert torch.equal(batch.detectors, reference[:, : raw.num_detectors])
    assert torch.equal(batch.observables, reference[:, raw.num_detectors :])


def test_external_seeded_batches_and_same_sample_decoding() -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    model = qec.DetectorErrorModel.from_stim_text(
        bridge.stim_circuit_to_dem_text(SOURCE)
    )
    decoder = qec.PyMatchingDecoder.from_detector_error_model(model)
    with (
        bridge.compile_stim_sampler(SOURCE, seed=53) as left,
        bridge.compile_stim_sampler(SOURCE, seed=53) as right,
    ):
        for size in (16, 16, 3):
            batch = left.sample(shots=size)
            assert batch == right.sample(shots=size)
            assert batch.detectors.dtype == batch.observables.dtype == torch.int8
            assert batch.detectors.device.type == batch.observables.device.type == "cpu"
            assert torch.equal(batch.detectors, batch.observables)
            assert torch.equal(decoder.decode_batch(batch.detectors), batch.observables)
            # Caller ownership cannot corrupt the next draw.
            batch.detectors.fill_(7)
    with pytest.raises(RuntimeError, match="closed"):
        left.sample(shots=1)


def test_lifecycle_and_invalid_draws_reuse_existing_handle() -> None:
    pytest.importorskip("stim")
    with bridge.compile_stim_sampler(SOURCE, seed=53) as handle:
        with bridge.compile_stim_sampler(SOURCE, seed=53) as reference:
            assert type(handle) is qec.MemorySampler
            for invalid in (0, -1, True, 1.5):
                with pytest.raises((TypeError, ValueError)):
                    handle.sample(shots=invalid)  # type: ignore[arg-type]
            assert handle.sample(shots=35) == reference.sample(shots=35)
        for action in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(TypeError):
                action(handle)
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            pytest.raises(RuntimeError, match="creator process and thread"),
        ):
            pool.submit(handle.sample, shots=1).result(timeout=10)
    handle.close()


@pytest.mark.parametrize("observables", ["", "OBSERVABLE_INCLUDE(2) rec[-1]"])
def test_reference_ones_are_flips_zero_and_logical_shape_is_complete(
    observables: str,
) -> None:
    pytest.importorskip("stim")
    text = "R 0\nX 0\nM 0\nDETECTOR rec[-1]\n" + observables
    with bridge.compile_stim_sampler(text, seed=53) as handle:
        batch = handle.sample(shots=5)
    assert not batch.detectors.any() and not batch.observables.any()
    expected = 3 if observables else 0
    assert batch.observables.shape == (5, expected)
    model = qec.DetectorErrorModel.from_stim_text(bridge.stim_circuit_to_dem_text(text))
    assert (model.num_detectors, model.num_observables) == (1, expected)


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("rounds", [1, 3])
def test_native_surface_export_matches_independent_measurement_conversion(
    basis: str, distance: int, rounds: int
) -> None:
    stim = pytest.importorskip("stim")
    code = qec.RotatedSurfaceCode(distance=distance)
    product = qec.Pauli(x_qubits=code.data_qubits) if basis == "x" else None
    memory = qec.build_memory_circuit(code, rounds=rounds, product=product)
    noise = qec.PhenomenologicalNoise(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
    )
    text = bridge.memory_to_stim_text(memory, noise=noise)
    circuit = stim.Circuit(text)
    assert circuit.num_detectors == len(memory.detectors)
    assert circuit.num_observables == len(memory.observables)
    assert not circuit.get_detector_coordinates()[0]  # Native geometry is not invented.
    raw = circuit.compile_sampler(seed=53).sample(shots=35)
    detectors, observables = circuit.compile_m2d_converter().convert(
        measurements=raw, separate_observables=True
    )
    with qec.compile_memory_sampler(memory, noise=noise, seed=53) as native:
        batch = native.sample(shots=35)
    assert torch.equal(batch.detectors, torch.as_tensor(detectors))
    assert torch.equal(batch.observables, torch.as_tensor(observables))
    with bridge.compile_stim_sampler(text, seed=53) as external:
        drawn = external.sample(shots=35)
    assert drawn.detectors.shape == batch.detectors.shape
    assert drawn.observables.shape == batch.observables.shape


def test_correlated_native_export_keeps_the_actual_positioned_law() -> None:
    stim = pytest.importorskip("stim")
    memory = qec.build_memory_circuit(qec.RepetitionCode(distance=3), rounds=1)
    channel = two_qubit_depolarizing_channel(0.3, dtype=torch.complex128)
    noise = NoiseModel().add("cx", channel)
    circuit = stim.Circuit(bridge.memory_to_stim_text(memory, noise=noise))
    assert "PAULI_CHANNEL_2" in str(circuit)
    # Sampling is supported independently of an exact noisy DEM representation.
    with bridge.compile_stim_sampler(str(circuit), seed=53) as handle:
        assert handle.sample(shots=35).shots == 35
    with pytest.raises(CapabilityError, match="exact DEM"):
        bridge.stim_circuit_to_dem_text(str(circuit))


def test_replaceable_numerical_boundary_and_failure_poisoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Replacement:
        closed = False
        fail = False

        def sample(self, *, shots: int) -> torch.Tensor:
            if self.fail:
                raise ArithmeticError("numerical draw failed")
            return torch.tensor([[1, 0, 1, 1]]).expand(shots, -1)

        def close(self) -> None:
            self.closed = True

    replacement = Replacement()
    monkeypatch.setattr(
        bridge, "_prepare_stim_experiment", lambda *args, **kwargs: (replacement, 2, 2)
    )
    handle = bridge.compile_stim_sampler(SOURCE, seed=53)
    batch = handle.sample(shots=3)
    assert torch.equal(batch.detectors, torch.tensor([[1, 0]]).expand(3, -1))
    assert torch.equal(batch.observables, torch.ones((3, 2), dtype=torch.int8))
    replacement.fail = True
    with pytest.raises(ArithmeticError, match="numerical draw failed"):
        handle.sample(shots=3)
    assert replacement.closed
    with pytest.raises(RuntimeError, match="closed"):
        handle.sample(shots=3)


def test_both_entrances_use_existing_fixed_shot_statistics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    from examples.qec.stim_experiment_bridge import main

    main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:4] == ["native", "35", "8", "1"]
    assert lines[2].split()[:4] == ["external", "35", "1", "1"]
    for index in (0, 2):
        assert 0 <= int(lines[index].split()[-1]) <= 35
        assert lines[index + 1].startswith("fixed-shot Wilson 95%")


def test_partial_factory_failure_releases_prepared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Prepared:
        closed = False

        def close(self) -> None:
            self.closed = True

    prepared = Prepared()

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("handle assembly failed")

    monkeypatch.setattr(
        bridge, "_prepare_stim_experiment", lambda *args, **kwargs: (prepared, 2, 1)
    )
    monkeypatch.setattr(bridge, "_memory_sampler", fail)
    with pytest.raises(RuntimeError, match="handle assembly failed"):
        bridge.compile_stim_sampler(SOURCE)
    assert prepared.closed
