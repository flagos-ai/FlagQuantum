"""The memory exporter can replace its SDK parser with the embedded engine."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operators.channels import bit_flip_channel
from flagquantum.errors import CapabilityError
from flagquantum.qec import PhenomenologicalNoise, RepetitionCode, build_memory_circuit
from flagquantum.qec.adapters.stim import memory_to_stim_text
from flagquantum.simulation.stabilizer import _embedded_sampler, engine

pytestmark = pytest.mark.unit


def _select_embedded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "_embedded_stim_profile.txt").write_text("stim-1.16.0\n")
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))


def test_export_replaces_sdk_without_changing_its_consumer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=2)
    validated: list[str] = []

    def no_sdk() -> None:
        pytest.fail("embedded memory export imported the SDK")

    _select_embedded(monkeypatch, tmp_path)
    monkeypatch.setattr(engine, "_stim", no_sdk)
    monkeypatch.setattr(
        _embedded_sampler,
        "_text_experiment_engine",
        lambda: SimpleNamespace(validate_text_experiment=validated.append),
    )
    text = memory_to_stim_text(memory, noise=PhenomenologicalNoise(data_flip=0.1))
    assert validated == [text]
    assert "PAULI_CHANNEL_1" in text
    assert text.count("DETECTOR") == len(memory.detectors)
    assert text.count("OBSERVABLE_INCLUDE") == len(memory.observables)


@pytest.mark.parametrize("column", [-1, 1])
def test_export_rejects_invalid_parity_before_native_validation(
    column: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _select_embedded(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="parity column outside"):
        engine._annotated_stim_text(
            CircuitIR(n_wires=1, instructions=()),
            terminal_qubits=(0,),
            detector_columns=((column,),),
            observable_columns=(),
        )


def test_export_serializes_record_controls_and_exclusive_faults() -> None:
    text = _embedded_sampler._annotated_operations_text(
        (
            ("R", (0, 1, 2), None),
            ("CORRELATED_ERROR", ((0, "X"), (1, "Y"), (2, "Z")), 0.25),
            ("ELSE_CORRELATED_ERROR", ((2, "X"),), 0.5),
            ("M", (0,), None),
            ("CX", (-1, 1), None),
        ),
        (1, 2),
        ((0, 1), ()),
        ((2,),),
    )
    assert "CORRELATED_ERROR(0.25) X0 Y1 Z2" in text
    assert "ELSE_CORRELATED_ERROR(0.5) X2" in text
    assert "CX rec[-1] 1" in text
    assert "DETECTOR rec[-3] rec[-2]\nDETECTOR\n" in text
    assert text.endswith("OBSERVABLE_INCLUDE(0) rec[-1]")


def test_native_limit_keeps_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overflow(text: str) -> None:
        raise OverflowError("Stim experiment exceeds 4,096 qubits")

    monkeypatch.setattr(
        _embedded_sampler,
        "_text_experiment_engine",
        lambda: SimpleNamespace(validate_text_experiment=overflow),
    )
    with pytest.raises(CapabilityError, match="4,096") as caught:
        _embedded_sampler._validate_text_experiment("M 0\nDETECTOR rec[-1]")
    assert isinstance(caught.value.__cause__, OverflowError)


@pytest.mark.parametrize("probability", [0.0, 1e-10, 0.1234567890123456, 1.0])
def test_embedded_export_matches_sdk_binary64_law(
    probability: float, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stim = pytest.importorskip("stim")
    pytest.importorskip("flagquantum.simulation.stabilizer._embedded_stim")
    channel = bit_flip_channel(probability)
    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                "bit_flip", (0,), matrix=channel.kraus, metadata={"is_channel": True}
            ),
        ),
    )
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
    reference = engine._annotated_stim_text(
        program,
        terminal_qubits=(0,),
        detector_columns=((0,),),
        observable_columns=((0,),),
    )
    _select_embedded(monkeypatch, tmp_path)
    actual = engine._annotated_stim_text(
        program,
        terminal_qubits=(0,),
        detector_columns=((0,),),
        observable_columns=((0,),),
    )
    assert stim.Circuit(actual) == stim.Circuit(reference)
    assert (
        stim.Circuit(actual)[0].gate_args_copy()
        == stim.Circuit(reference)[0].gate_args_copy()
    )


def test_embedded_export_does_not_add_determinism_requirement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("flagquantum.simulation.stabilizer._embedded_stim")
    _select_embedded(monkeypatch, tmp_path)
    text = engine._annotated_stim_text(
        CircuitIR(n_wires=1, instructions=(Instruction("h", (0,)),)),
        terminal_qubits=(0,),
        detector_columns=((0,),),
        observable_columns=(),
    )
    assert "H 0" in text
    with pytest.raises(CapabilityError, match="nondeterministic"):
        engine._prepare_stim_experiment(text, seed=53)


@pytest.mark.parametrize(
    ("text", "exception", "message"),
    [
        ("M 0", ValueError, "at least one detector"),
        ("BAD_GATE 0\nDETECTOR", ValueError, "invalid Stim circuit"),
        ("M 4096\nDETECTOR rec[-1]", CapabilityError, "4,096"),
        (
            "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(1000000) rec[-1]",
            CapabilityError,
            "recorded columns",
        ),
        (
            "REPEAT 1000001 {\nM 0\n}\nDETECTOR rec[-1]",
            CapabilityError,
            "expanded instructions",
        ),
    ],
)
def test_embedded_export_keeps_parser_limits(
    text: str, exception: type[Exception], message: str
) -> None:
    pytest.importorskip("flagquantum.simulation.stabilizer._embedded_stim")
    with pytest.raises(exception, match=message):
        _embedded_sampler._validate_text_experiment(text)
