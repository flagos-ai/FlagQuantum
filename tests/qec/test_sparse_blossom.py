"""Conformance tests for the bundled Sparse Blossom CPU path."""

import sys

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.experimental.qec import (
    SparseBlossomDecoder,
    run_stim_matching_experiment,
)
from flagquantum.qec import DecodingGraph, DetectorErrorModel, PyMatchingDecoder
from flagquantum.qec.adapters.stim import stim_circuit_to_dem_text
from flagquantum.qec.dem.graph import DecodingGraphEdge

pytestmark = pytest.mark.integration

pytest.importorskip("flagquantum.qec.decoders._sparse_blossom")


def test_batched_matching_without_pymatching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pymatching", None)
    graph = DecodingGraph(
        3,
        2,
        (
            DecodingGraphEdge((0, 1), 0.15, (0,)),
            DecodingGraphEdge((0, 3), 0.1, (1,)),
            DecodingGraphEdge((1, 3), 0.1),
        ),
    )
    decoder = SparseBlossomDecoder(graph)
    syndromes = torch.tensor(
        [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.int8
    ).T
    expected = torch.tensor(
        [[False, False], [False, True], [True, False], [False, False]]
    )
    assert torch.equal(decoder.decode_batch(syndromes), expected)
    with pytest.raises(CapabilityError, match="No perfect matching"):
        decoder.decode_batch(torch.tensor([[0, 0, 1]]))
    assert torch.equal(decoder.decode_batch(syndromes), expected)
    assert decoder.decode_batch(syndromes[:0]).shape == (0, 2)


def test_rejects_parallel_edges_and_nonbinary_input() -> None:
    edge = DecodingGraphEdge((0, 1), 0.1)
    with pytest.raises(CapabilityError, match="one edge"):
        SparseBlossomDecoder(
            DecodingGraph(1, 1, (edge, DecodingGraphEdge((0, 1), 0.1, (0,))))
        )
    decoder = SparseBlossomDecoder(DecodingGraph(1, 1, (edge,)))
    with pytest.raises(ValueError, match="only binary"):
        decoder.decode_batch(torch.tensor([[256]]))


@pytest.mark.parametrize("distance", (3, 5, 7))
def test_surface_code_batch_matches_pymatching(distance: int) -> None:
    stim = pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=3,
        after_clifford_depolarization=0.0005,
        after_reset_flip_probability=0.0005,
        before_measure_flip_probability=0.0005,
    )
    text = str(circuit)
    model = DetectorErrorModel.from_stim_text(
        stim_circuit_to_dem_text(text, decompose_errors=True),
        use_decomp_suggestions=True,
    ).merge_duplicate_mechanisms()
    graph = DecodingGraph.from_detector_error_model(model)
    events, _ = circuit.compile_detector_sampler(seed=31).sample(
        shots=8192, separate_observables=True
    )
    dense = torch.from_numpy(events)
    native = SparseBlossomDecoder(graph)
    external = PyMatchingDecoder.from_decoding_graph(graph)
    assert torch.equal(native.decode_batch(dense), external.decode_batch(dense))


def test_stim_experiment_without_pymatching(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("stim")
    monkeypatch.setitem(sys.modules, "pymatching", None)
    source = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    graph = DecodingGraph(1, 1, (DecodingGraphEdge((0, 1), 0.1, (0,)),))
    result = run_stim_matching_experiment(
        source, graph=graph, shots=53, batch_size=8, seed=7
    )
    assert result.shots == 53
    assert result.logical_failures == result.logical_error_rate == 0
    assert result.observable_failures == (0,)


def test_public_stim_experiment_uses_bundled_batch_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("stim")
    from flagquantum.qec.execution import experiments

    monkeypatch.setitem(sys.modules, "pymatching", None)

    def unexpected_sparse_decode(*args: object, **kwargs: object) -> None:
        pytest.fail("The bundled experiment should use dense batch predictions")

    monkeypatch.setattr(experiments, "decode_batch", unexpected_sparse_decode)
    text = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    graph = DecodingGraph(1, 1, (DecodingGraphEdge((0, 1), 0.1, (0,)),))
    options = dict(shots=541, batch_size=32, seed=53)
    expected = run_stim_matching_experiment(text, graph=graph, **options)
    result = experiments.run_stim_experiment(text, decoder="sparse_blossom", **options)
    assert result == expected
    assert result.logical_failures == 0


def test_public_memory_experiment_supports_uniform_gate_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("stim")
    from flagquantum.noise import (
        NoiseModel,
        depolarizing_channel,
        two_qubit_depolarizing_channel,
    )
    from flagquantum.qec import (
        RotatedSurfaceCode,
        build_memory_circuit,
    )
    from flagquantum.qec.adapters.stim import memory_to_stim_text
    from flagquantum.qec.execution import experiments

    monkeypatch.setitem(sys.modules, "pymatching", None)
    memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
    noise = (
        NoiseModel()
        .add("h", depolarizing_channel(0.01, dtype=torch.complex128))
        .add("cx", two_qubit_depolarizing_channel(0.01, dtype=torch.complex128))
    )
    source = memory_to_stim_text(memory, noise=noise)
    assert "PAULI_CHANNEL_2" in source
    options = dict(decoder="sparse_blossom", shots=12_345, batch_size=256, seed=11)
    result = experiments.run_memory_experiment(memory, noise=noise, **options)
    exported = experiments.run_stim_experiment(source, **options)
    assert result.logical_failures > 0
    assert result.logical_failures == exported.logical_failures
    assert result.confidence_interval == exported.confidence_interval
    assert result.sampling_source == "memory_circuit"
