from __future__ import annotations

import inspect
import json
from dataclasses import fields
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.errors as fqe
import flagquantum.interop as fqi

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "interop-protocol-v1-candidate.json"


def _load() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_interop_protocol_is_candidate_stable_not_frozen() -> None:
    candidate = _load()

    assert candidate["status"] == "implemented_pending_review"
    assert candidate["implementation_authorized"] is True
    assert candidate["root_manifest_change"] is False
    assert candidate["rules"]["candidate_is_frozen_contract"] is False
    assert candidate["protocol_semantics"]["adapter_implementations_stable"] is False


def test_interop_namespace_matches_candidate_exactly() -> None:
    extension = _load()["stable_extension"]

    assert extension["namespace"] == "flagquantum.interop"
    assert list(fqi.__all__) == extension["additions"]
    assert extension["new_namespace_only"] == extension["additions"]
    assert set(extension["additions"]).isdisjoint(dir(fq))
    assert {"DEFAULT_INTEROP_REGISTRY", "pennylane", "qiskit"}.isdisjoint(fqi.__all__)


def test_interop_function_signatures_match_candidate() -> None:
    signatures = _load()["public_signatures"]

    for name, expected in signatures.items():
        assert str(inspect.signature(getattr(fqi, name), eval_str=False)) == expected


def test_interop_record_fields_match_candidate() -> None:
    expected_fields = _load()["record_fields"]

    for name, expected in expected_fields.items():
        assert [field.name for field in fields(getattr(fqi, name))] == expected


def test_interop_errors_join_stable_error_boundary() -> None:
    assert issubclass(fqi.InteropError, fqe.FlagQuantumError)
    assert issubclass(fqi.InteropDependencyError, fqi.InteropError)
    assert issubclass(fqi.InteropDependencyError, ImportError)
    assert issubclass(fqi.InteropConversionError, fqi.InteropError)
    assert issubclass(fqi.InteropConversionError, ValueError)
    assert issubclass(fqi.InteropRegistryError, fqi.InteropError)
    assert issubclass(fqi.InteropRegistryError, RuntimeError)
    assert fqi.InteropError.category == "interop"


def test_experimental_interop_contains_implementations_not_protocol_aliases() -> None:
    assert fq.experimental.interop.__all__ == ("pennylane", "qiskit")
    assert fq.experimental.interop.qiskit is fqi.qiskit
    assert fq.experimental.interop.pennylane is fqi.pennylane
    with pytest.raises(AttributeError):
        getattr(fq.experimental.interop, "from_qiskit")
