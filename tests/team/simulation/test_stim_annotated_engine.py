"""Independent SDK references for strict annotated experiment conversion."""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.noise import bit_flip_channel
from flagquantum.simulation.stabilizer import engine

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "probability", [1e-10, 0.005, 0.1234567890123456, 0.999999, 1.0]
)
def test_native_text_round_trip_retains_executed_probability(
    probability: float,
) -> None:
    stim = pytest.importorskip("stim")
    channel = bit_flip_channel(probability)
    mixture = channel.unitary_mixture
    assert mixture is not None
    expected = float(mixture.probabilities[-1])
    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                "bit_flip", (0,), matrix=channel.kraus, metadata={"is_channel": True}
            ),
        ),
    )
    text = engine._annotated_stim_text(
        program,
        terminal_qubits=(0,),
        detector_columns=((0,),),
        observable_columns=((0,),),
    )
    reconstructed = stim.Circuit(text)
    assert reconstructed[0].gate_args_copy()[0] == expected
    assert reconstructed.num_detectors == reconstructed.num_observables == 1


@pytest.mark.parametrize("text", ["", " ", "INVALID_GATE 0"])
def test_invalid_text_has_actionable_category(text: str) -> None:
    pytest.importorskip("stim")
    with pytest.raises(ValueError):
        engine._prepare_stim_experiment(text, seed=53)


@pytest.mark.parametrize(
    "text",
    [
        "R 4096\nM 4096\nDETECTOR rec[-1]",
        "R 0\nREPEAT 1000001 {\nX 0\n}\nM 0\nDETECTOR rec[-1]",
        "R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(1000000) rec[-1]",
        "#" + "é" * 524288,
    ],
)
def test_limits_refuse_before_compilation(text: str) -> None:
    pytest.importorskip("stim")
    with pytest.raises(CapabilityError, match="exceeds"):
        engine._prepare_stim_experiment(text, seed=53)


@pytest.mark.parametrize(
    "text",
    [
        "H 0\nM 0\nDETECTOR rec[-1]",
        "H 0\nR 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]",
    ],
)
def test_nondeterministic_detector_or_logical_is_refused(text: str) -> None:
    pytest.importorskip("stim")
    with pytest.raises(CapabilityError, match="nondeterministic") as caught:
        engine._prepare_stim_experiment(text, seed=53)
    assert caught.value.__cause__ is not None
    with pytest.raises(CapabilityError, match="exact DEM"):
        engine._stim_experiment_dem_text(text, decompose_errors=False)


COORDINATED = """
    R 0
    M 0
    REPEAT 3 {
        X_ERROR(0.125) 0
        M 0
        DETECTOR(2, 3) rec[-1] rec[-2]
        SHIFT_COORDS(0, 1)
    }
    OBSERVABLE_INCLUDE(2) rec[-1]
"""


def test_repeat_coordinate_and_sparse_logical_shape_round_trip() -> None:
    stim = pytest.importorskip("stim")
    original = stim.Circuit(COORDINATED)
    expected = original.detector_error_model().flattened()
    text = engine._stim_experiment_dem_text(COORDINATED, decompose_errors=False)
    canonical = stim.DetectorErrorModel(text)
    assert canonical.get_detector_coordinates() == expected.get_detector_coordinates()
    assert canonical.num_detectors == original.num_detectors == 3
    assert canonical.num_observables == original.num_observables == 3
    # Added shape lines must not affect any error or decomposition target.
    assert [str(x) for x in canonical if x.type == "error"] == [
        str(x) for x in expected if x.type == "error"
    ]
    assert "logical_observable L0" in text and "logical_observable L1" in text


def test_record_controls_and_repeated_detector_stream_match_sdk() -> None:
    stim = pytest.importorskip("stim")
    text = """
        R 0 1
        X_ERROR(0.125) 0
        M 0
        CX rec[-1] 1
        M 1
        DETECTOR(1, 2) rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    expected = stim.Circuit(text).compile_detector_sampler(seed=53)
    prepared, detectors, observables = engine._prepare_stim_experiment(text, seed=53)
    assert (detectors, observables) == (1, 1)
    try:
        for size in (16, 16, 3):
            actual = prepared.sample(shots=size)
            assert torch.equal(
                actual,
                torch.as_tensor(expected.sample(shots=size, append_observables=True)),
            )
    finally:
        prepared.close()


def test_exclusive_circuit_sampling_does_not_use_an_approximated_dem() -> None:
    pytest.importorskip("stim")
    text = """
        R 0 1
        CORRELATED_ERROR(0.2) X0
        ELSE_CORRELATED_ERROR(0.375) X1
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
    """
    prepared, _, _ = engine._prepare_stim_experiment(text, seed=53)
    try:
        record = prepared.sample(shots=10000)
    finally:
        prepared.close()
    assert not (record[:, 0].bool() & record[:, 1].bool()).any()
    assert torch.equal(record[:, 2], record[:, 0] ^ record[:, 1])
    # Fixed sample budget and simultaneous Hoeffding bound declared in advance.
    bound = math.sqrt(math.log(6 / 1e-6) / (2 * 10000))
    for index, probability in enumerate((0.2, 0.3, 0.5)):
        assert abs(float(record[:, index].double().mean()) - probability) <= bound
    with pytest.raises(CapabilityError, match="exact DEM"):
        engine._stim_experiment_dem_text(text, decompose_errors=False)


def test_invalid_prepared_detector_records_keep_existing_validation() -> None:
    class Malformed:
        def sample(self, *, shots: int, append_observables: bool) -> torch.Tensor:
            assert append_observables
            return torch.full((shots, 2), 7)

    prepared = engine._PreparedMeasurements(Malformed(), 2, True)
    with pytest.raises(ValidationError, match="nonbinary"):
        prepared.sample(shots=3)
    prepared.close()
