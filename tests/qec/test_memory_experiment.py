"""Fixed-shot composition, independent counts and unchanged owning boundaries."""

from __future__ import annotations

import dataclasses
import inspect
from enum import IntEnum
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.noise import NoiseModel, bit_flip_channel
from flagquantum.qec import (
    DetectorErrorModel,
    Pauli,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
    get_decoder,
)
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)
from flagquantum.qec.decoders import registry
from flagquantum.qec.decoders.batch import decode_batch
from flagquantum.qec.execution import experiments as ex
from flagquantum.simulation.stabilizer import StabilizerDependencyError, engine


class Integer(IntEnum):
    SHOTS = 35
    BATCH = 16
    SEED = 53


pytestmark = pytest.mark.unit
NAMES = (
    "minimum_weight_matching",
    "sliding_window_matching",
    "belief_propagation",
    "belief_propagation_osd",
    "pymatching",
)


def memory() -> Any:
    return build_memory_circuit(RepetitionCode(distance=3), rounds=1)


def run(**options: Any) -> ex.MemoryExperimentResult:
    values: dict[str, Any] = dict(
        noise=PhenomenologicalNoise(),
        decoder=NAMES[0],
        shots=35,
        batch_size=16,
        seed=53,
    )
    values.update(options)
    return ex.run_memory_experiment(values.pop("memory", memory()), **values)


@pytest.fixture
def controlled(monkeypatch: pytest.MonkeyPatch) -> Any:
    calls: list[Any] = []

    class Sampler:
        cursor = 0
        closed = 0
        error: BaseException | None = None
        cleanup: BaseException | None = None
        observed = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.int8)

        def sample(self, *, shots: int) -> Any:
            calls.append(("sample", shots))
            if self.error is not None:
                raise self.error
            index = torch.arange(self.cursor, self.cursor + shots) % 4
            self.cursor += shots
            return SimpleNamespace(
                detectors=torch.zeros((shots, 2), dtype=torch.int8),
                observables=self.observed[index],
            )

        def close(self) -> None:
            self.closed += 1
            if self.cleanup is not None:
                raise self.cleanup

    handle = Sampler()
    model = DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\ndetector D0\ndetector D1\n"
        "logical_observable L0\nlogical_observable L1\n"
    )
    m = memory()
    object.__setattr__(m, "observables", (0, 1))
    object.__setattr__(m, "detectors", (0, 1))

    def export(m: Any, **kw: Any) -> str:
        calls.append("export")
        return "ORIGINAL"

    monkeypatch.setattr(ex, "memory_to_stim_text", export)

    def dem(text: str, *, decompose_errors: bool) -> str:
        assert text == "ORIGINAL" and decompose_errors is False
        calls.append("strict")
        return model.to_stim_text()

    monkeypatch.setattr(ex, "stim_circuit_to_dem_text", dem)

    def compile(text: str, *, seed: int | None) -> Any:
        assert text == "ORIGINAL"
        calls.append(("compile", seed))
        return handle

    monkeypatch.setattr(ex, "compile_stim_sampler", compile)
    return SimpleNamespace(handle=handle, calls=calls, memory=m)


def test_exact_tail_multiple_observables_and_immutable_result(controlled: Any) -> None:
    result = run(
        memory=controlled.memory,
        shots=Integer.SHOTS,
        batch_size=Integer.BATCH,
        seed=Integer.SEED,
    )
    assert result.logical_failures == 27
    assert result.observable_failures == (18, 18)
    assert result.logical_error_rate == 27 / 35
    assert result.seed == 53 and type(result.seed) is int
    assert result.sampling_source == "memory_circuit"
    assert controlled.calls == [
        "export",
        "strict",
        ("compile", 53),
        ("sample", 16),
        ("sample", 16),
        ("sample", 3),
    ]
    assert controlled.handle.closed == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "shots", 1)
    assert [f.name for f in dataclasses.fields(result)] == [
        "shots",
        "logical_failures",
        "logical_error_rate",
        "observable_failures",
        "confidence_interval",
        "seed",
        "sampling_source",
    ]
    assert all(f.default is dataclasses.MISSING for f in dataclasses.fields(result))


@pytest.mark.parametrize("observed,failures", [(0, 0), (1, 35)])
def test_zero_all_and_batch_larger_than_shots(
    controlled: Any, observed: int, failures: int
) -> None:
    controlled.handle.observed.fill_(observed)
    result = run(memory=controlled.memory, batch_size=100, seed=None)
    assert result.logical_failures == failures
    assert result.observable_failures == (failures, failures)
    assert result.seed is None
    assert controlled.calls[-1] == ("sample", 35)


@pytest.mark.parametrize(
    "n,k", [(1, 0), (1, 1), (35, 0), (35, 35), (35, 7), (1000, 531)]
)
def test_interval_by_inverting_the_score_test(n: int, k: int) -> None:
    low, high = ex._wilson_interval(k, n)
    assert (low, high) == pytest.approx(_score_interval(k, n), abs=1e-12, rel=0)
    z = NormalDist().inv_cdf(0.975)

    def score(p: float) -> float:
        return (k - n * p) ** 2 - z * z * n * p * (1 - p)

    assert 0 <= low <= k / n <= high <= 1
    assert score(low) == pytest.approx(0, abs=1e-9)
    assert score(high) == pytest.approx(0, abs=1e-9)
    if k == 0:
        assert low == 0 and high > 0
    if k == n:
        assert high == 1 and low < 1


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("shots", True, TypeError),
        ("shots", 1.5, TypeError),
        ("shots", 0, ValueError),
        ("batch_size", False, TypeError),
        ("batch_size", "16", TypeError),
        ("batch_size", -1, ValueError),
        ("seed", True, TypeError),
        ("seed", 1.5, TypeError),
        ("seed", -1, ValidationError),
        ("seed", 2**64, ValidationError),
        ("noise", None, TypeError),
        ("memory", "text", TypeError),
        ("decoder", object(), TypeError),
        ("decoder", "absent", ValueError),
    ],
)
def test_invalid_inputs_do_not_export_or_compile(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any, error: type[Exception]
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("preparation before validation")

    monkeypatch.setattr(ex, "memory_to_stim_text", unexpected)
    monkeypatch.setattr(ex, "compile_stim_sampler", unexpected)
    with pytest.raises(error):
        run(**{field: value})


@pytest.mark.parametrize("field", ["detectors", "observables"])
def test_missing_parity_refused_before_export(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    m = memory()
    object.__setattr__(m, field, ())
    monkeypatch.setattr(
        ex, "memory_to_stim_text", lambda *a, **k: pytest.fail("export")
    )
    with pytest.raises(ValueError):
        run(memory=m)


@pytest.mark.parametrize("stage", ["sample", "decode"])
def test_primary_failure_and_cleanup_context(
    controlled: Any, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    primary = RuntimeError("primary")
    cleanup = OSError("cleanup")
    controlled.handle.cleanup = cleanup
    if stage == "sample":
        controlled.handle.error = primary
    else:
        original = decode_batch

        def decode(decoder: Any, rows: Any) -> Any:
            if rows:
                raise primary
            return original(decoder, rows)

        monkeypatch.setattr(ex, "decode_batch", decode)
    with pytest.raises(RuntimeError) as caught:
        run(memory=controlled.memory)
    assert caught.value is primary and caught.value.__cause__ is cleanup
    assert controlled.handle.closed == 1


def test_sole_cleanup_failure_returns_no_result(controlled: Any) -> None:
    error = OSError("close")
    controlled.handle.cleanup = error
    with pytest.raises(OSError) as caught:
        run(memory=controlled.memory)
    assert caught.value is error


@pytest.mark.parametrize("indices", [(True,), (0.5,), (-1,), (2,), (0, 0)])
def test_malformed_prediction_closes(
    controlled: Any, monkeypatch: pytest.MonkeyPatch, indices: Any
) -> None:
    original = decode_batch

    def decode(decoder: Any, rows: Any) -> Any:
        return (
            tuple(SimpleNamespace(observables=indices) for _ in rows)
            if rows
            else original(decoder, rows)
        )

    monkeypatch.setattr(ex, "decode_batch", decode)
    with pytest.raises(ValidationError):
        run(memory=controlled.memory)
    assert controlled.handle.closed == 1


@pytest.mark.parametrize(
    "bad",
    [torch.ones((3, 1)), torch.full((16, 2), 2), torch.empty((16, 2), device="meta")],
)
def test_malformed_samples_close(controlled: Any, bad: torch.Tensor) -> None:
    controlled.handle.sample = lambda **kw: SimpleNamespace(
        detectors=bad, observables=bad
    )
    with pytest.raises(ValidationError):
        run(memory=controlled.memory)
    assert controlled.handle.closed == 1


def test_registry_replacement_is_resolved_and_refused_before_compile(
    controlled: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Plugin:
        @classmethod
        def from_detector_error_model(cls, model: Any) -> Any:
            return cls()

        def decode(self, row: Any) -> Any:
            return SimpleNamespace(observables=())

    monkeypatch.setitem(registry._DECODERS, NAMES[0], Plugin)
    assert isinstance(
        get_decoder(NAMES[0], DetectorErrorModel.from_stim_text("detector D0")), Plugin
    )
    with pytest.raises(CapabilityError):
        run(memory=controlled.memory)
    assert controlled.calls == ["export", "strict"]


def test_missing_stim_stays_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> Any:
        raise StabilizerDependencyError("stim absent")

    monkeypatch.setattr(engine, "_stim", missing)
    with pytest.raises(StabilizerDependencyError):
        run()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("surface", [False, True])
@pytest.mark.parametrize("noise", [PhenomenologicalNoise(), NoiseModel()])
def test_real_source_and_five_decoders(name: str, surface: bool, noise: Any) -> None:
    pytest.importorskip("stim")
    if name == "pymatching":
        pytest.importorskip("pymatching")
    m = build_memory_circuit(
        RotatedSurfaceCode(distance=3) if surface else RepetitionCode(distance=3),
        rounds=1,
    )
    if name == "pymatching":
        with pytest.raises(CapabilityError, match="flipped by no mechanism"):
            run(memory=m, noise=noise, decoder=name)
        return
    result = run(memory=m, noise=noise, decoder=name)
    assert result.shots == 35 and result.logical_failures == 0
    assert result.observable_failures == (0,)


def test_opt_in_surface_only() -> None:
    import flagquantum.qec as qec
    import flagquantum.qec.execution as execution

    assert ex.__all__ == (
        "MemoryExperimentResult",
        "run_memory_experiment",
        "StimExperimentResult",
        "run_stim_experiment",
    )
    assert not hasattr(qec, "run_memory_experiment")
    assert not hasattr(execution, "run_memory_experiment")
    assert list(inspect.signature(ex.run_memory_experiment).parameters) == [
        "memory",
        "noise",
        "decoder",
        "shots",
        "batch_size",
        "seed",
    ]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "profile,noise",
    [
        ("repetition", PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)),
        ("repetition", NoiseModel().add("cx", bit_flip_channel(0.125))),
        *(
            (
                basis,
                PhenomenologicalNoise(
                    data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
                ),
            )
            for basis in ("surface_x", "surface_z")
        ),
    ],
)
def test_noisy_actual_law_counts_match_original_text_and_single_shot_reference(
    name: str, profile: str, noise: Any
) -> None:
    pytest.importorskip("stim")
    if name == "pymatching":
        pytest.importorskip("pymatching")
    m = memory()
    if profile.startswith("surface"):
        code = RotatedSurfaceCode(distance=3)
        product = Pauli(x_qubits=code.data_qubits) if profile == "surface_x" else None
        m = build_memory_circuit(code, rounds=3, product=product)
        assert m.readout_basis == profile[-1]
        assert (len(m.detectors), len(m.observables)) == (24, 1)
    text = memory_to_stim_text(m, noise=noise)
    model = DetectorErrorModel.from_stim_text(
        stim_circuit_to_dem_text(text, decompose_errors=False)
    )
    selected = get_decoder(name, model)
    counts = [0] * model.num_observables
    failed = 0
    with compile_stim_sampler(text, seed=53) as sampler:
        for size in (16, 16, 3):
            batch = sampler.sample(shots=size)
            for detection, observed in zip(
                batch.detectors, batch.observables, strict=True
            ):
                answer = selected.decode(detection.nonzero().flatten().tolist())
                wrong = [
                    int(bool(observed[i]) != (i in answer.observables))
                    for i in range(len(observed))
                ]
                failed += int(any(wrong))
                counts = [a + b for a, b in zip(counts, wrong, strict=True)]
    result = run(memory=m, noise=noise, decoder=name)
    assert result.shots == 35 and result.seed == 53
    assert result.sampling_source == "memory_circuit"
    assert result.logical_failures == failed
    assert result.logical_error_rate == failed / 35
    assert result.observable_failures == tuple(counts)
    assert result.confidence_interval == ex._wilson_interval(failed, 35)
    assert result.confidence_interval == pytest.approx(
        _score_interval(failed, 35), abs=1e-12, rel=0
    )


def _score_interval(failures: int, shots: int) -> tuple[float, float]:
    """Invert the binomial score test without using the production formula."""
    quantile_squared = NormalDist().inv_cdf(0.975) ** 2
    proportion = failures / shots

    def score(p: float) -> float:
        return (failures - shots * p) ** 2 - quantile_squared * shots * p * (1 - p)

    low, high = 0.0, proportion
    for _ in range(64):
        middle = (low + high) / 2
        if score(middle) > 0:
            low = middle
        else:
            high = middle
    lower = (low + high) / 2
    low, high = proportion, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if score(middle) > 0:
            high = middle
        else:
            low = middle
    return lower, (low + high) / 2


@pytest.mark.parametrize("basis", ("x", "z"))
def test_three_round_surface_zero_noise_matching_refusal(basis: str) -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    code = RotatedSurfaceCode(distance=3)
    product = Pauli(x_qubits=code.data_qubits) if basis == "x" else None
    m = build_memory_circuit(code, rounds=3, product=product)
    with pytest.raises(CapabilityError, match="flipped by no mechanism"):
        run(memory=m, noise=PhenomenologicalNoise(), decoder="pymatching")


def test_late_decode_failure_never_returns_partial_counts(
    controlled: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = decode_batch
    calls = 0
    primary = RuntimeError("late batch")

    def decode(decoder: Any, rows: Any) -> Any:
        nonlocal calls
        if rows:
            calls += 1
        if calls == 2:
            raise primary
        return original(decoder, rows)

    monkeypatch.setattr(ex, "decode_batch", decode)
    with pytest.raises(RuntimeError) as caught:
        run(memory=controlled.memory)
    assert caught.value is primary and controlled.handle.cursor == 32
    assert controlled.handle.closed == 1


def test_constructor_failure_is_not_replaced(
    controlled: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValueError("constructor rejected model")

    def constructor(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(ex, "get_decoder", constructor)
    with pytest.raises(ValueError) as caught:
        run(memory=controlled.memory)
    assert caught.value is error
    assert controlled.calls == ["export", "strict"]


def test_model_dimensions_rejected_before_compile(controlled: Any) -> None:
    object.__setattr__(controlled.memory, "detectors", (0,))
    with pytest.raises(ValidationError, match="dimensions"):
        run(memory=controlled.memory)
    assert controlled.calls == ["export", "strict"]


@pytest.mark.parametrize("indices", [None, "0", [0]])
def test_invalid_prediction_carrier_is_validation_error(
    controlled: Any, monkeypatch: pytest.MonkeyPatch, indices: Any
) -> None:
    original = decode_batch
    monkeypatch.setattr(
        ex,
        "decode_batch",
        lambda decoder, rows: (
            tuple(SimpleNamespace(observables=indices) for _ in rows)
            if rows
            else original(decoder, rows)
        ),
    )
    with pytest.raises(ValidationError):
        run(memory=controlled.memory)
    assert controlled.handle.closed == 1


def test_nonconverged_bp_record_is_counted(
    controlled: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.qec.decoders.belief_propagation import BeliefPropagationDecoder

    original = BeliefPropagationDecoder.decode

    def decode(self: Any, row: Any) -> Any:
        return dataclasses.replace(original(self, row), converged=False)

    monkeypatch.setattr(BeliefPropagationDecoder, "decode", decode)
    assert (
        run(memory=controlled.memory, decoder="belief_propagation").logical_failures
        == 27
    )


@pytest.mark.parametrize("subclass", [False, True])
def test_noncertified_decoder_refused_even_if_it_decodes(
    controlled: Any, monkeypatch: pytest.MonkeyPatch, subclass: bool
) -> None:
    from flagquantum.qec.decoders.matching import MinimumWeightMatchingDecoder
    from flagquantum.qec.decoders.sliding_window import SlidingWindowDecoder

    class Derived(MinimumWeightMatchingDecoder):
        pass

    model = DetectorErrorModel.from_stim_text(
        "error(.1) D0 L0\ndetector D0\nlogical_observable L0"
    )
    selected: MinimumWeightMatchingDecoder | SlidingWindowDecoder
    if subclass:
        selected = Derived.from_detector_error_model(model)
    else:
        # The independent-shot contract refuses a stateful stream by type.
        selected = object.__new__(SlidingWindowDecoder)
    monkeypatch.setattr(ex, "get_decoder", lambda *args: selected)
    with pytest.raises(CapabilityError):
        run(memory=controlled.memory)
    assert controlled.calls == ["export", "strict"]


def test_missing_matching_dependency_is_not_substituted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("stim")
    from flagquantum.qec.adapters import pymatching as adapter

    error = adapter.MatchingDependencyError("pymatching absent")

    def missing() -> Any:
        raise error

    monkeypatch.setattr(adapter, "_pymatching", missing)
    with pytest.raises(adapter.MatchingDependencyError) as caught:
        run(
            noise=PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01),
            decoder="pymatching",
        )
    assert caught.value is error


def test_unsupported_channel_keeps_strict_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("stim")
    from flagquantum.noise import amplitude_damping_channel

    monkeypatch.setattr(
        ex, "compile_stim_sampler", lambda *a, **k: pytest.fail("compile after refusal")
    )
    with pytest.raises(CapabilityError):
        run(noise=NoiseModel().add("cx", amplitude_damping_channel(0.1)))


@pytest.mark.parametrize("batch_size", [1, 7, 35, 64])
def test_ordered_mixed_rows_across_chunk_schedules(
    controlled: Any, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    detections = [(bool(i % 2), bool((i // 2) % 2)) for i in range(35)]
    observed = [(i % 5 == 0, i % 7 < 2) for i in range(35)]
    decoded_rows: list[tuple[int, ...]] = []
    original_decode = decode_batch

    def sample(*, shots: int) -> SimpleNamespace:
        controlled.calls.append(("sample", shots))
        start = controlled.handle.cursor
        controlled.handle.cursor += shots
        return SimpleNamespace(
            detectors=torch.tensor(detections[start : start + shots]),
            observables=torch.tensor(observed[start : start + shots]),
        )

    def decode(decoder: Any, rows: tuple[tuple[int, ...], ...]) -> tuple[Any, ...]:
        if not rows:
            return original_decode(decoder, rows)
        decoded_rows.extend(rows)
        return tuple(
            SimpleNamespace(
                observables=tuple(
                    index
                    for index, bit in enumerate((0 in row, (0 in row) != (1 in row)))
                    if bit
                )
            )
            for row in rows
        )

    monkeypatch.setattr(controlled.handle, "sample", sample)
    monkeypatch.setattr(ex, "decode_batch", decode)
    result = run(memory=controlled.memory, batch_size=batch_size)
    wrong = [
        (first != actual[0], (first != second) != actual[1])
        for (first, second), actual in zip(detections, observed, strict=True)
    ]
    failures = sum(any(row) for row in wrong)
    assert result.shots == 35 and result.logical_failures == failures
    assert result.logical_error_rate == failures / 35
    assert result.observable_failures == tuple(
        sum(row[column] for row in wrong) for column in range(2)
    )
    assert result.confidence_interval == pytest.approx(
        _score_interval(failures, 35), abs=1e-12, rel=0
    )
    assert decoded_rows == [
        tuple(index for index, bit in enumerate(row) if bit) for row in detections
    ]
    expected = {1: [1] * 35, 7: [7] * 5, 35: [35], 64: [35]}[batch_size]
    assert controlled.calls == [
        "export",
        "strict",
        ("compile", 53),
        *(("sample", size) for size in expected),
    ]
    assert controlled.handle.cursor == 35 and controlled.handle.closed == 1


@pytest.mark.parametrize(
    "fault",
    [
        "second_sample",
        "tail_sample",
        "tail_detectors",
        "tail_observables",
        "tail_prediction",
        "tail_result_count",
        "tail_decode_cleanup",
        "tail_interrupt",
    ],
)
def test_late_chunk_failure_stops_and_preserves_exception(
    controlled: Any, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    original_sample = controlled.handle.sample
    original_decode = decode_batch
    requests: list[int] = []
    decoded_sizes: list[int] = []
    primary: BaseException = (
        KeyboardInterrupt("tail interrupted")
        if fault == "tail_interrupt"
        else RuntimeError("late chunk failure")
    )
    cleanup = OSError("late cleanup failure")
    if fault == "tail_decode_cleanup":
        controlled.handle.cleanup = cleanup

    def sample(*, shots: int) -> Any:
        requests.append(shots)
        if (fault == "second_sample" and len(requests) == 2) or (
            len(requests) == 3 and fault in ("tail_sample", "tail_interrupt")
        ):
            raise primary
        batch = original_sample(shots=shots)
        if len(requests) == 3 and fault in ("tail_detectors", "tail_observables"):
            field = fault.removeprefix("tail_")
            setattr(batch, field, torch.zeros((shots, 3), dtype=torch.int8))
        return batch

    def decode(decoder: Any, rows: tuple[tuple[int, ...], ...]) -> tuple[Any, ...]:
        if not rows:
            return original_decode(decoder, rows)
        decoded_sizes.append(len(rows))
        if len(requests) == 3:
            if fault == "tail_decode_cleanup":
                raise primary
            if fault == "tail_prediction":
                return tuple(SimpleNamespace(observables=(2,)) for _ in rows)
            if fault == "tail_result_count":
                return original_decode(decoder, rows)[:-1]
        return original_decode(decoder, rows)

    monkeypatch.setattr(controlled.handle, "sample", sample)
    monkeypatch.setattr(ex, "decode_batch", decode)
    injected = fault in (
        "second_sample",
        "tail_sample",
        "tail_decode_cleanup",
        "tail_interrupt",
    )
    expected_error = type(primary) if injected else ValidationError
    with pytest.raises(expected_error) as caught:
        run(memory=controlled.memory)
    if injected:
        assert caught.value is primary
    if fault == "tail_decode_cleanup":
        assert caught.value.__cause__ is cleanup
    else:
        assert caught.value.__cause__ is None
    assert requests == ([16, 16] if fault == "second_sample" else [16, 16, 3])
    if fault == "second_sample":
        expected_decodes = [16]
    elif fault in (
        "tail_sample",
        "tail_interrupt",
        "tail_detectors",
        "tail_observables",
    ):
        expected_decodes = [16, 16]
    else:
        expected_decodes = [16, 16, 3]
    assert decoded_sizes == expected_decodes
    assert controlled.handle.closed == 1


@pytest.mark.parametrize("basis", ("x", "z"))
@pytest.mark.parametrize("name", ("minimum_weight_matching", "pymatching"))
@pytest.mark.parametrize("batch_size", (1, 35, 64))
def test_surface_counts_against_same_schedule_single_shot_reference(
    basis: str, name: str, batch_size: int
) -> None:
    pytest.importorskip("stim")
    if name == "pymatching":
        pytest.importorskip("pymatching")
    code = RotatedSurfaceCode(distance=3)
    product = Pauli(x_qubits=code.data_qubits) if basis == "x" else None
    m = build_memory_circuit(code, rounds=3, product=product)
    noise = PhenomenologicalNoise(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
    )
    text = memory_to_stim_text(m, noise=noise)
    model = DetectorErrorModel.from_stim_text(
        stim_circuit_to_dem_text(text, decompose_errors=False)
    )
    selected = get_decoder(name, model)
    failures = 0
    # Each reference has its own identical call schedule, not cross-size samples.
    schedule = [1] * 35 if batch_size == 1 else [35]
    with compile_stim_sampler(text, seed=53) as sampler:
        for size in schedule:
            batch = sampler.sample(shots=size)
            for detection, observed in zip(
                batch.detectors, batch.observables, strict=True
            ):
                answer = selected.decode(detection.nonzero().flatten().tolist())
                failures += int(bool(observed[0]) != (0 in answer.observables))
    result = run(memory=m, noise=noise, decoder=name, batch_size=batch_size)
    assert result.shots == 35 and result.seed == 53
    assert result.sampling_source == "memory_circuit"
    assert result.logical_failures == failures
    assert result.observable_failures == (failures,)
    assert result.logical_error_rate == failures / 35
    assert result.confidence_interval == pytest.approx(
        _score_interval(failures, 35), abs=1e-12, rel=0
    )
