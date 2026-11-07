"""The classical-binding successor protects its exact additive API scope."""

import copy
import inspect
import json
from pathlib import Path

import pytest

from flagquantum import synth
from tools.public_api_snapshot import _validate_authorized_synthesis_classical

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def contract() -> dict[str, object]:
    return json.loads(
        (ROOT / "contracts/synthesis-native-classical-candidate.json").read_text()
    )


def test_successor_and_predecessor_signatures_are_both_protected() -> None:
    active = contract()
    assert _validate_authorized_synthesis_classical(active) == []
    signature = inspect.signature(synth.synthesize)
    assert str(signature) == active["synthesize_signature"]
    prior = signature.replace(
        parameters=[
            p for name, p in signature.parameters.items() if name != "classical_inputs"
        ]
    )
    assert str(prior) == active["previous_synthesize_signature"]
    assert synth.RX is __import__("flagquantum.services.synthesis", fromlist=["RX"]).RX
    with pytest.raises(TypeError, match="captured"):
        synth.RX(0.3, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("added_exports", ["RX", "RY"]),
        ("synthesize_signature", "object -> object"),
        ("previous_synthesize_signature", "object -> object"),
        ("binding", "runtime"),
        ("annotations", ["float", "list[list[float]]"]),
        ("max_classical_elements", 4097),
        ("implementation_authorized", 1),
        ("release_authorized", True),
        ("new_ir", True),
        ("unknown", True),
    ],
)
def test_successor_rejects_scope_drift(field: str, value: object) -> None:
    altered = copy.deepcopy(contract())
    altered[field] = value
    assert _validate_authorized_synthesis_classical(altered)
