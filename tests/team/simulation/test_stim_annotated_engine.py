"""Independent SDK references for strict annotated experiment conversion."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operators.channels import bit_flip_channel
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.qec.execution.experiments import run_stim_experiment
from flagquantum.simulation.stabilizer import _embedded_sampler, engine

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


def test_record_controls_and_repeated_detector_stream_match_sdk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stim = pytest.importorskip("stim")
    # This assertion pins the SDK stream independently of the native stream.
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
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


@pytest.fixture
def embedded_text_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    if not Path(engine.__file__).with_name("_embedded_stim_profile.txt").is_file():
        pytest.skip("requires an embedded Stim installation")

    def refuse_sdk() -> None:
        pytest.fail("the embedded text workflow attempted to load the SDK")

    monkeypatch.setattr(engine, "_stim", refuse_sdk)


NOISY_TEXT = """\
R 0
X 0
X_ERROR(0.25) 0
M 0
DETECTOR(2, 3) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.mark.usefixtures("embedded_text_engine")
def test_embedded_text_runs_original_noisy_circuit_and_reports_decoding() -> None:
    prepared, detectors, observables = engine._prepare_stim_experiment(
        NOISY_TEXT, seed=53
    )
    try:
        records = prepared.sample(shots=8192)
    finally:
        prepared.close()
    assert (detectors, observables) == (1, 1)
    assert torch.equal(records[:, 0], records[:, 1])
    assert abs(float(records[:, 0].double().mean()) - 0.25) < 0.04
    result = run_stim_experiment(
        NOISY_TEXT, decoder="minimum_weight_matching", shots=35, batch_size=16, seed=53
    )
    assert result.shots == 35
    assert result.logical_failures == 0
    assert result.sampling_source == "stim_circuit"


@pytest.mark.usefixtures("embedded_text_engine")
def test_embedded_text_matches_stim_116_exact_dem_and_deterministic_records() -> None:
    stim = pytest.importorskip("stim")
    assert stim.__version__ == "1.16.0"
    original = stim.Circuit(COORDINATED)
    expected = original.detector_error_model().flattened()
    exact = stim.DetectorErrorModel(
        engine._stim_experiment_dem_text(COORDINATED, decompose_errors=False)
    )
    assert exact.get_detector_coordinates() == expected.get_detector_coordinates()
    assert exact.num_observables == original.num_observables == 3
    assert [str(op) for op in exact if op.type == "error"] == [
        str(op) for op in expected if op.type == "error"
    ]

    deterministic = NOISY_TEXT.replace("0.25", "1")
    sdk_sampler = stim.Circuit(deterministic).compile_detector_sampler(seed=53)
    stream, _, _ = engine._prepare_stim_experiment(deterministic, seed=53)
    try:
        for shots in (16, 16, 3):
            assert torch.equal(
                stream.sample(shots=shots),
                torch.from_numpy(
                    sdk_sampler.sample(shots=shots, append_observables=True)
                ),
            )
    finally:
        stream.close()


@pytest.mark.usefixtures("embedded_text_engine")
def test_embedded_text_preserves_exact_decomposition_for_bundled_matching() -> None:
    source = NOISY_TEXT.replace("X_ERROR(0.25)", "PAULI_CHANNEL_1(0.125,0.125,0.125)")
    result = run_stim_experiment(
        source, decoder="sparse_blossom", shots=35, batch_size=16, seed=53
    )
    assert result.shots == 35 and result.logical_failures == 0


@pytest.mark.usefixtures("embedded_text_engine")
@pytest.mark.parametrize(
    ("text", "error_type", "message"),
    [
        ("INVALID_GATE 0", ValueError, "invalid Stim circuit text"),
        ("M 0", ValueError, "requires at least one detector"),
        ("#" + "é" * 524288, CapabilityError, "1 MiB"),
        ("M 4096\nDETECTOR rec[-1]", CapabilityError, "4,096 qubits"),
        (
            "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(1000000) rec[-1]",
            CapabilityError,
            "recorded columns",
        ),
        (
            "REPEAT 1000001 {\nX 0\n}\nM 0\nDETECTOR rec[-1]",
            CapabilityError,
            "expanded instructions",
        ),
    ],
)
def test_embedded_text_preserves_source_limits(
    text: str, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        engine._prepare_stim_experiment(text, seed=53)
    with pytest.raises(error_type, match=message):
        engine._stim_experiment_dem_text(text, decompose_errors=False)


def test_external_text_consumer_uses_native_boundary_with_original_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[object, ...]] = []
    handle = object()

    def prepare(text: str, seed: int) -> tuple[object, int, int]:
        calls.append(("prepare", text, seed))
        return handle, 1, 1

    def analyze(text: str, decompose: bool, normalize: bool) -> str:
        calls.append(("DEM", text, decompose, normalize))
        return "error(0.25) D0 L0\ndetector D0\nlogical_observable L0"

    def sample(state: object, shots: int) -> tuple[bytearray, bytearray]:
        assert state is handle
        calls.append(("sample", shots))
        return bytearray([1] * shots), bytearray([1] * shots)

    mark = tmp_path / "_embedded_stim_profile.txt"
    mark.write_text("stim-1.16.0\n")
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
    monkeypatch.setattr(
        _embedded_sampler,
        "import_module",
        lambda _: SimpleNamespace(
            prepare_text_experiment=prepare,
            text_experiment_dem=analyze,
            sample_detectors=sample,
        ),
    )
    result = run_stim_experiment(
        NOISY_TEXT, decoder="minimum_weight_matching", shots=35, batch_size=16, seed=53
    )
    assert (result.shots, result.logical_failures) == (35, 0)
    assert calls == [
        ("DEM", NOISY_TEXT, False, False),
        ("prepare", NOISY_TEXT, 53),
        ("sample", 16),
        ("sample", 16),
        ("sample", 3),
    ]


@pytest.mark.parametrize(
    ("native_error", "message"),
    [
        (OverflowError("Stim experiment exceeds 4,096 qubits"), "4,096 qubits"),
        (RuntimeError("strict refusal"), "declared parities|exact DEM"),
    ],
)
def test_native_adapter_preserves_limits_and_strict_analysis_failures(
    monkeypatch: pytest.MonkeyPatch,
    native_error: Exception,
    message: str,
) -> None:
    def reject(*args: object) -> NoReturn:
        raise native_error

    monkeypatch.setattr(
        _embedded_sampler,
        "import_module",
        lambda _: SimpleNamespace(
            prepare_text_experiment=reject, text_experiment_dem=reject
        ),
    )
    with pytest.raises(CapabilityError, match=message) as sampling:
        _embedded_sampler._prepare_text_experiment(NOISY_TEXT, 53)
    with pytest.raises(CapabilityError, match=message) as analysis:
        _embedded_sampler._text_experiment_dem(NOISY_TEXT, False, False)
    assert sampling.value.__cause__ is analysis.value.__cause__ is native_error


def test_native_adapter_rejects_a_missing_binary_without_sdk_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> NoReturn:
        raise ImportError("extension not found")

    monkeypatch.setattr(_embedded_sampler, "import_module", missing)
    with pytest.raises(engine.StabilizerDependencyError) as caught:
        _embedded_sampler._prepare_text_experiment(NOISY_TEXT, 53)
    assert isinstance(caught.value.__cause__, ImportError)


def test_native_adapter_checks_dem_text_and_detector_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _embedded_sampler,
        "import_module",
        lambda _: SimpleNamespace(
            prepare_text_experiment=lambda text, seed: (object(), 1, 0),
            text_experiment_dem=lambda *args: 1,
            sample_detectors=lambda handle, shots: (bytearray(), bytearray()),
        ),
    )
    with pytest.raises(CapabilityError, match="invalid text result"):
        _embedded_sampler._text_experiment_dem(NOISY_TEXT, False, False)
    sampler, detectors, observables = _embedded_sampler._prepare_text_experiment(
        NOISY_TEXT, 53
    )
    assert (detectors, observables) == (1, 0)
    with pytest.raises(ValueError, match="invalid bytes"):
        sampler.sample(shots=1)
