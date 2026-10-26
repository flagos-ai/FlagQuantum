"""Canonical encoding keeps frozen attributes independent and byte-exact."""

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from flagquantum.compiler._hybrid import model

pytestmark = pytest.mark.unit


def _program(attributes: dict[str, Any]) -> model.HybridProgram:
    value = model.Value(model.ValueId("scope", 0), model.tensor_type("float64", (2,)))
    operation = model.Operation(
        "test.payload", operands=(value,), results=(value,), attributes=attributes
    )
    return model.HybridProgram(
        "encoding", model.Region((model.Block((value,), (operation,)),))
    )


def _canonical(data: object) -> bytes:
    return json.dumps(
        data, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"z": 1, "a": 2},
        {"z": {"last": 0, "first": [True, None, {"y": 2, "x": 1}]}},
        {"unicode": "\u03bb\U0001f600", "escaped": '\n\t"\\'},
        {"negative_zero": -0.0, "large": 2**100, "small": 1e-300},
        {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")},
        {"tuple": ({"z": 1, "a": 0}, (3, 2, 1))},
    ],
)
def test_canonical_bytes_and_identity_match_independent_data(
    attributes: dict[str, Any],
) -> None:
    program = _program(attributes)
    # State the wire representation independently of model.to_data/_encode.
    value = {
        "id": {"scope": "scope", "index": 0},
        "type": {"kind": "tensor", "parameters": ["float64", [2]], "linear": False},
    }
    expected = {
        "kind": "flagquantum.hybrid_program",
        "name": "encoding",
        "body": {
            "blocks": [
                {
                    "arguments": [value],
                    "operations": [
                        {
                            "name": "test.payload",
                            "operands": [value],
                            "results": [value],
                            "attributes": attributes,
                            "regions": [],
                        }
                    ],
                }
            ]
        },
    }
    canonical = _canonical(expected)
    assert _canonical(program.to_data()) == canonical
    assert program.semantic_identity == hashlib.sha256(canonical).hexdigest()
    located = replace(program, location=model.SourceLocation("diagnostic.py", 9))
    assert located.semantic_identity == program.semantic_identity


def test_serialization_does_not_resort_frozen_attribute_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _program({"z": [{"y": 2, "a": 1}], "a": {"x": 3}})
    expected = _canonical(program.to_data())

    def unexpected_sort(*args: object, **kwargs: object) -> None:
        raise AssertionError("frozen attributes were sorted again during encoding")

    monkeypatch.setattr(model, "sorted", unexpected_sort, raising=False)
    assert _canonical(program.to_data()) == expected
    assert program.semantic_identity == hashlib.sha256(expected).hexdigest()


def test_serialized_containers_are_fresh_and_keep_sorted_attribute_order() -> None:
    attributes = {"z": {"last": [3, 2, 1], "first": 0}, "a": True}
    program = _program(attributes)
    expected = _canonical(program.to_data())
    first = program.to_data()
    payload = first["body"]["blocks"][0]["operations"][0]["attributes"]
    assert list(payload) == ["a", "z"]
    assert list(payload["z"]) == ["first", "last"]
    payload["z"]["last"].append(4)
    attributes["z"]["last"].append(5)
    assert _canonical(program.to_data()) == expected
    assert program.semantic_identity == hashlib.sha256(expected).hexdigest()


def test_unfrozen_mapping_encoding_keeps_existing_sorted_order() -> None:
    encoded = model._encode({"z": {"y": 2, "a": 1}, "a": 0})
    assert list(encoded) == ["a", "z"]
    assert list(encoded["z"]) == ["a", "y"]
