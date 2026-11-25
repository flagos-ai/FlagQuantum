"""Explicit quantization is a captured public marker with a checked contract."""

import inspect
import json
from pathlib import Path

import pytest

from flagquantum import synth
from flagquantum.services import synthesis
from tools.public_api_snapshot import validate

pytestmark = pytest.mark.unit


def test_quantize_public_identity_signature_and_host_refusal() -> None:
    root = Path(__file__).resolve().parents[2]
    contract = json.loads(
        (
            root / "contracts/synthesis-native-quantized-expressions-candidate.json"
        ).read_text()
    )
    assert synth.quantize is synthesis.quantize
    assert str(inspect.signature(synth.quantize)) == contract["signature"]
    assert synth.__all__.count("quantize") == 1
    assert synthesis.__all__ == synth.__all__
    assert not validate()
    with pytest.raises(TypeError, match="captured inside qfunc"):
        synth.quantize(0.1, fractional_bits=8, literal_bits=20)


def test_quantize_signature_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def wrong(expression: float) -> float:
        return expression

    monkeypatch.setattr(synth, "quantize", wrong)
    assert any("quantize differs" in message for message in validate())
