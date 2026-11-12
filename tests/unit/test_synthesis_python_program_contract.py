"""The authorized UP-01 successor protects an exact additive namespace delta."""

import copy
import json
from pathlib import Path

import pytest

from flagquantum import synth
from tools.public_api_snapshot import _validate_authorized_synthesis_python_program

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def contract() -> dict[str, object]:
    return json.loads(
        (ROOT / "contracts/synthesis-python-program-v1-candidate.json").read_text()
    )


def test_successor_is_authorized_without_changing_predecessor_exports() -> None:
    active = contract()
    assert _validate_authorized_synthesis_python_program(active) == []
    previous = json.loads(
        (ROOT / "contracts/synthesis-phase-v1-candidate.json").read_text()
    )
    assert active["previous_namespace_exports"] == previous["namespace_exports"]
    assert active["release_authorized"] is False
    namespace: dict[str, object] = {}
    exec("from flagquantum.synth import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(synth.__all__)


@pytest.mark.parametrize(
    "field,value",
    [
        ("namespace_exports", ["qfunc", "synthesize"]),
        ("implementation_authorized", False),
        ("implementation_authorized", 1),
        ("release_authorized", True),
        ("release_authorized", 0),
        ("root_roles", ["Const", "Output"]),
        ("helper_roles", ["Input", "Output"]),
        ("source_memory_default", None),
        ("new_executor", True),
        ("serialization_change", True),
        ("typed_program_overload", False),
        ("human_trial_records", ["not a real user trial"]),
        ("new_operation_signatures", {}),
        ("historical_contracts_sha256", {}),
        ("synthesize_signature", "object -> object"),
        ("unknown_field", True),
    ],
)
def test_successor_guard_refuses_unapproved_expansion(
    field: str, value: object
) -> None:
    altered = copy.deepcopy(contract())
    altered[field] = value
    assert _validate_authorized_synthesis_python_program(altered)


@pytest.mark.parametrize(
    ("operation", "diagnostic"),
    [
        ("phase", "native program phase differs"),
        ("quantize", "native quantize differs"),
        ("RESET", "native RESET differs"),
        ("control", "native coherent control differs"),
        ("qct_type2", "native trigonometric transforms differ"),
        ("drop", "native hidden storage differs"),
        ("qpe", "native matrix qpe differs"),
        ("select_rotation", "native rotation tables differ"),
        ("qft", "native Fourier operation differs"),
        ("unitary", "native unitary operation differs"),
        ("modular_exp", "native modular exponentiation differs"),
        ("qdrift", "native sampled Pauli evolution differs"),
        ("min", "native numeric interpretation differs"),
    ],
)
def test_successor_checks_each_capability_signature(
    monkeypatch: pytest.MonkeyPatch, operation: str, diagnostic: str
) -> None:
    def unsupported() -> None:
        pass

    monkeypatch.setattr(synth, operation, unsupported)
    errors = _validate_authorized_synthesis_python_program(contract())
    assert any(diagnostic in error for error in errors)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: synth.allocate(2),
        lambda: synth.hadamard_transform(None),
        lambda: synth.QNum(),
    ],
)
def test_annotation_operations_never_allocate_in_host_python(operation) -> None:
    with pytest.raises(TypeError):
        operation()
