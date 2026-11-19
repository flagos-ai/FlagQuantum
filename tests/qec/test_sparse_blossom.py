"""Conformance tests for the bundled Sparse Blossom CPU path."""

import runpy
import sys
from pathlib import Path

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
from flagquantum.qec.execution.detector_sampling import _packed

pytestmark = pytest.mark.integration

pytest.importorskip("flagquantum.qec.decoders._sparse_blossom")


def test_batched_matching_without_optional_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pymatching", None)
    monkeypatch.setattr(torch.Tensor, "numpy", None)
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
    packed = _packed(dense)
    assert torch.equal(
        native.decode_packed_batch(packed), external.decode_packed_batch(packed)
    )
    for packed_input in (False, True):
        for packed_output in (False, True):
            options = dict(
                bit_packed_shots=packed_input, bit_packed_predictions=packed_output
            )
            batch = packed if packed_input else dense
            predictions, weights = native.decode_batch_with_weights(batch, **options)
            expected, expected_weights = external.decode_batch_with_weights(
                batch, **options
            )
            assert torch.equal(predictions, expected)
            assert torch.equal(weights, expected_weights)


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


@pytest.mark.parametrize(
    ("decoder_args", "expected"),
    [
        ([], "sparse_blossom"),
        (["--decoder", "minimum_weight_matching"], "minimum_weight_matching"),
    ],
)
def test_memory_example_selects_one_decoder_without_pymatching(
    decoder_args: list[str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("stim")
    from examples.qec import memory_experiment

    monkeypatch.setitem(sys.modules, "pymatching", None)
    monkeypatch.setattr(sys, "argv", ["memory_experiment.py", *decoder_args])
    memory_experiment.main()
    reports = capsys.readouterr().out.splitlines()
    assert len(reports) == 1
    assert reports[0].split()[:2] == [expected, "35"]


@pytest.mark.parametrize(
    ("width", "observables"),
    ((1, 1), (7, 9), (8, 8), (9, 7), (17, 64), (64, 64), (65, 64)),
)
def test_native_packed_boundary_oracle_without_optional_packages(
    width: int, observables: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pymatching", None)
    monkeypatch.setattr(torch.Tensor, "numpy", None)
    graph = DecodingGraph(
        width,
        observables,
        tuple(
            DecodingGraphEdge((i, width), 0.1, (observables - 1 - i % observables,))
            for i in range(width)
        ),
    )
    decoder = SparseBlossomDecoder(graph)
    dense = torch.randint(2, (37, width), generator=torch.Generator().manual_seed(23))
    dense[0] = 0
    dense[1] = 1
    packed = _packed(dense)
    logicals = torch.zeros((37, observables), dtype=torch.uint8)
    for i in range(width):
        logicals[:, observables - 1 - i % observables] ^= dense[:, i].to(torch.uint8)
    expected = _packed(logicals)
    storage = torch.zeros((37, packed.shape[1] * 2), dtype=torch.uint8)
    storage[:, ::2] = packed
    strided = storage[:, ::2]

    # Packed decoding must use the native packed route.
    def reject_dense(*args: object) -> None:
        pytest.fail("packed decoding expanded into the dense decoder")

    monkeypatch.setattr(decoder, "_decode", reject_dense)
    result = decoder.decode_packed_batch(strided)
    assert torch.equal(result, expected)
    assert torch.equal(strided, packed)
    result.fill_(0)
    assert torch.equal(decoder.decode_packed_batch(packed), expected)
    empty = decoder.decode_packed_batch(packed[:0])
    assert empty.shape == (0, (observables + 7) // 8) and empty.dtype == torch.uint8


def test_native_packed_validation_and_recovery() -> None:
    graph = DecodingGraph(9, 1, (DecodingGraphEdge((0, 9), 0.1, (0,)),))
    decoder = SparseBlossomDecoder(graph)
    good = torch.tensor([[1, 0]], dtype=torch.uint8)
    for invalid in (None, [[1, 0]], good.bool(), good.to(torch.int32)):
        with pytest.raises(TypeError):
            decoder.decode_packed_batch(invalid)
    for invalid in (good.flatten(), good[:, :1], good.reshape(1, 1, 2)):
        with pytest.raises(ValueError, match="shape"):
            decoder.decode_packed_batch(invalid)
    with pytest.raises(CapabilityError, match="CPU"):
        decoder.decode_packed_batch(
            torch.empty((1, 2), device="meta", dtype=torch.uint8)
        )
    with pytest.raises(ValueError, match="padding"):
        decoder.decode_packed_batch(torch.tensor([[1, 2]], dtype=torch.uint8))
    from flagquantum.qec.decoders import _sparse_blossom

    with pytest.raises(ValueError, match="padding"):
        _sparse_blossom.decode_packed_batch(decoder._matcher, b"\x01\x02")
    for _ in range(8):
        # A failed batch may already have decoded a valid row and matched
        # some events in the failing row. Reuse the same solver in both modes.
        with pytest.raises(CapabilityError, match="No perfect matching"):
            decoder.decode_packed_batch(
                torch.tensor([[1, 0], [3, 0]], dtype=torch.uint8)
            )
        assert decoder.decode_packed_batch(good).tolist() == [[1]]
        dense = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
        assert decoder.decode_batch(dense).tolist() == [[True]]
        dense[0, 1] = 1
        with pytest.raises(CapabilityError, match="No perfect matching"):
            decoder.decode_batch(dense)
        assert decoder.decode_packed_batch(good).tolist() == [[1]]


@pytest.mark.parametrize(
    "name", ("shot_file_decoding", "shot_file_encoding", "compiled_dem_sampler")
)
def test_packed_examples_without_pymatching(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from flagquantum.simulation.stabilizer import engine

    if not Path(engine.__file__).with_name("_embedded_stim_profile.txt").is_file():
        pytest.skip("streaming shot examples require the embedded Stim build")
    monkeypatch.setitem(sys.modules, "pymatching", None)
    path = Path(__file__).resolve().parents[2] / "examples" / "qec" / f"{name}.py"
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out.strip() == "shots 64 batches 4 logical_failures 0"
