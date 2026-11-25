"""Fixed-shot source composition with independent labels and failure ownership.

Pure fixtures exercise the contract in core lanes. Real SDK/profile acceptance
is mandatory in tools/qec_installed_smoke.py, not replaced by optional skips.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import flagquantum.qec as qec
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.qec.decoders.batch import decode_batch
from flagquantum.qec.execution import experiments as ex
from flagquantum.simulation.stabilizer import StabilizerDependencyError

pytestmark = pytest.mark.unit
SOURCE = "original supported text"
MODEL = "error(0.125) D0 L0\ndetector D0\ndetector D1\nlogical_observable L0\nlogical_observable L1\n"


class Handle:
    def __init__(self, calls: list[tuple[str, int | None]]) -> None:
        self.calls = calls
        self.cursor = 0
        self.closed = 0
        self.error: BaseException | None = None
        self.cleanup: BaseException | None = None
        self.observed = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.int8)
        self.overrides: dict[str, Any] = {}

    def sample(self, *, shots: int) -> SimpleNamespace:
        self.calls.append(("sample", shots))
        if self.error is not None:
            raise self.error
        index = torch.arange(self.cursor, self.cursor + shots) % 4
        self.cursor += shots
        values = dict(
            detectors=torch.zeros((shots, 2), dtype=torch.int8),
            observables=self.observed[index],
        )
        values.update(self.overrides)
        return SimpleNamespace(**values)

    def close(self) -> None:
        self.closed += 1
        if self.cleanup is not None:
            raise self.cleanup


@pytest.fixture
def controlled(monkeypatch: pytest.MonkeyPatch) -> Handle:
    calls: list[tuple[str, int | None]] = []
    handle = Handle(calls)

    def strict(text: str, *, decompose_errors: bool) -> str:
        assert text == SOURCE and decompose_errors is False
        calls.append(("strict", None))
        return MODEL

    def compile_source(text: str, *, seed: int | None) -> Handle:
        assert text == SOURCE
        calls.append(("compile", seed))
        return handle

    monkeypatch.setattr(ex, "stim_circuit_to_dem_text", strict)
    monkeypatch.setattr(ex, "compile_stim_sampler", compile_source)
    return handle


def run(**options: Any) -> ex.StimExperimentResult:
    values: dict[str, Any] = dict(
        decoder="minimum_weight_matching", shots=35, batch_size=16, seed=53
    )
    values.update(options)
    return ex.run_stim_experiment(values.pop("circuit_text", SOURCE), **values)


def test_original_source_tail_and_all_observable_counts(controlled: Handle) -> None:
    result = run()
    assert result.shots == 35 and result.seed == 53
    assert result.logical_failures == 27
    assert result.observable_failures == (18, 18)
    assert result.logical_error_rate == 27 / 35
    assert result.sampling_source == "stim_circuit"
    assert controlled.calls == [
        ("strict", None),
        ("compile", 53),
        ("sample", 16),
        ("sample", 16),
        ("sample", 3),
    ]
    assert controlled.closed == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "shots", 36)
    quantile = NormalDist().inv_cdf(0.975)
    for endpoint in result.confidence_interval:
        proportion = result.logical_error_rate
        residual = (proportion - endpoint) ** 2 - quantile**2 * endpoint * (
            1 - endpoint
        ) / 35
        assert abs(residual) < 1e-12


@pytest.mark.parametrize(
    "shots,batch,draws", [(1, 8, [1]), (4, 4, [4]), (5, 2, [2, 2, 1])]
)
def test_draw_boundaries(
    controlled: Handle, shots: int, batch: int, draws: list[int]
) -> None:
    result = run(shots=shots, batch_size=batch)
    assert result.shots == shots
    assert [size for name, size in controlled.calls if name == "sample"] == draws
    assert controlled.closed == 1


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("circuit_text", None, TypeError),
        ("circuit_text", b"text", TypeError),
        ("decoder", 1, TypeError),
        ("decoder", "missing", ValueError),
        ("shots", True, TypeError),
        ("shots", 0, ValueError),
        ("shots", 1.5, TypeError),
        ("batch_size", False, TypeError),
        ("batch_size", 0, ValueError),
        ("batch_size", "2", TypeError),
        ("seed", True, TypeError),
        ("seed", 1.5, TypeError),
        ("seed", -1, ValidationError),
        ("seed", 2**64, ValidationError),
    ],
)
def test_invalid_values_precede_preparation(
    controlled: Handle, field: str, value: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        run(**{field: value})
    assert controlled.calls == [] and controlled.closed == 0


@pytest.mark.parametrize("model", ["detector D0\n", "logical_observable L0\n"])
def test_missing_parities_precede_compilation(
    controlled: Handle, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setattr(ex, "stim_circuit_to_dem_text", lambda *args, **kw: model)
    with pytest.raises(ValueError, match="detector"):
        run()
    assert controlled.calls == [] and controlled.closed == 0


def test_missing_stim_preserves_dependency_error(
    controlled: Handle, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = StabilizerDependencyError("install the stim extra")

    def absent(*args: Any, **kwargs: Any) -> str:
        raise original

    monkeypatch.setattr(ex, "stim_circuit_to_dem_text", absent)
    with pytest.raises(StabilizerDependencyError) as caught:
        run()
    assert caught.value is original and controlled.closed == 0


def test_refused_decoder_precedes_compilation(
    controlled: Handle, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise CapabilityError("strict model cannot be decoded")

    monkeypatch.setattr(ex, "get_decoder", refused)
    with pytest.raises(CapabilityError):
        run()
    assert controlled.calls == [("strict", None)] and controlled.closed == 0


def test_empty_batch_admission_precedes_sampling(
    controlled: Handle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ex, "get_decoder", lambda *args: object())
    with pytest.raises(TypeError):
        run()
    assert controlled.calls == [("strict", None)] and controlled.closed == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("detectors", torch.zeros((1, 2), dtype=torch.int8)),
        ("detectors", torch.full((16, 2), 2, dtype=torch.int8)),
        ("observables", torch.zeros((16, 1), dtype=torch.int8)),
        ("observables", torch.full((16, 2), float("nan"))),
    ],
)
def test_invalid_backend_sample_closes(
    controlled: Handle, field: str, value: torch.Tensor
) -> None:
    controlled.overrides[field] = value
    with pytest.raises(ValidationError):
        run()
    assert controlled.closed == 1


@pytest.mark.parametrize("bad", [None, (True,), (2,), (0, 0), [0]])
def test_invalid_predictions_close(
    controlled: Handle, monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    def decode(selected: Any, rows: Any) -> tuple[Any, ...]:
        if not rows:
            return ()
        return tuple(SimpleNamespace(observables=bad) for _ in rows)

    monkeypatch.setattr(ex, "decode_batch", decode)
    with pytest.raises(ValidationError):
        run()
    assert controlled.closed == 1


def test_actual_labels_do_not_enter_decoder(
    controlled: Handle, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = decode_batch
    seen: list[tuple[tuple[int, ...], ...]] = []

    def observe(selected: Any, rows: tuple[tuple[int, ...], ...]) -> Any:
        seen.append(rows)
        return original(selected, rows)

    monkeypatch.setattr(ex, "decode_batch", observe)
    run()
    assert seen == [
        (),
        tuple(() for _ in range(16)),
        tuple(() for _ in range(16)),
        tuple(() for _ in range(3)),
    ]
    assert controlled.closed == 1


def test_primary_exception_preserved_with_cleanup_cause(controlled: Handle) -> None:
    primary = RuntimeError("sample failure")
    cleanup = RuntimeError("close failure")
    controlled.error = primary
    controlled.cleanup = cleanup
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is primary and caught.value.__cause__ is cleanup
    assert controlled.closed == 1


def test_successful_samples_do_not_hide_close_failure(controlled: Handle) -> None:
    error = RuntimeError("close failure")
    controlled.cleanup = error
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is error and controlled.closed == 1


def test_independent_four_fault_distribution_and_swapped_logicals() -> None:
    probabilities = (
        Fraction(21, 32),
        Fraction(3, 32),
        Fraction(7, 32),
        Fraction(1, 32),
    )
    faults = ((False, False), (True, False), (False, True), (True, True))
    assert sum(probabilities) == 1
    assert (
        tuple(
            (Fraction(1, 8) if a else Fraction(7, 8))
            * (Fraction(1, 4) if b else Fraction(3, 4))
            for a, b in faults
        )
        == probabilities
    )
    model = qec.DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L1\nerror(0.25) D1 L0\ndetector D0\ndetector D1\nlogical_observable L0\nlogical_observable L1\n"
    )
    decoder = qec.get_decoder("minimum_weight_matching", model)
    for first, second in faults:
        syndrome = tuple(i for i, flipped in enumerate((first, second)) if flipped)
        predicted = decoder.decode(syndrome).observables
        assert predicted == tuple(
            i for i, flipped in enumerate((second, first)) if flipped
        )


def test_approved_module_contract_and_old_record_unchanged() -> None:
    import flagquantum.qec.execution as execution

    contract = json.loads(
        (
            Path(__file__).parents[2] / "contracts/qec-stim-text-experiment-v1.json"
        ).read_text()
    )
    assert (
        list(inspect.signature(ex.run_stim_experiment).parameters)
        == contract["parameters"]
    )
    assert [
        field.name for field in dataclasses.fields(ex.StimExperimentResult)
    ] == contract["result_fields"]
    assert [
        field.name for field in dataclasses.fields(ex.MemoryExperimentResult)
    ] == contract["result_fields"]
    assert (
        ex.StimExperimentResult.__annotations__["sampling_source"]
        == "Literal['stim_circuit']"
    )
    assert (
        ex.MemoryExperimentResult.__annotations__["sampling_source"]
        == "Literal['memory_circuit']"
    )
    assert ex.__all__ == tuple(contract["module_exports"])
    assert not hasattr(qec, "run_stim_experiment") and not hasattr(
        execution, "run_stim_experiment"
    )
    assert not hasattr(qec, "StimExperimentResult") and not hasattr(
        execution, "StimExperimentResult"
    )
