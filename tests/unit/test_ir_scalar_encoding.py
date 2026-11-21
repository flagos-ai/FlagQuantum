"""Preserve source bytes and protocol precedence while encoding scalar leaves."""

from __future__ import annotations

import hashlib
import json
from collections import UserDict
from decimal import Decimal
from enum import Enum
from fractions import Fraction

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction, IRSerializationError
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        True,
        0,
        2**80,
        0.125,
        -0.0,
        float("inf"),
        float("nan"),
        "",
        "\N{GREEK SMALL LETTER THETA}",
    ],
)
def test_builtin_metadata_leaves_retain_standard_json_bytes(value: object) -> None:
    source = CircuitIR(1, (), metadata={"value": value})
    expected = {
        "kind": "flagquantum.circuit_ir",
        "version": "1.0",
        "n_wires": 1,
        "dtype": "complex64",
        "shape": [2],
        "instructions": [],
        "observables": [],
        "measurements": [],
        "metadata": {"value": value},
    }
    for indent in (None, 2):
        encoded = json.dumps(
            expected,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )
        assert source.to_json(indent=indent) == encoded
        if indent is None:
            assert source.content_hash == hashlib.sha256(encoded.encode()).hexdigest()


class _ScalarClaimMeta(type):
    def __eq__(cls, other: object) -> bool:
        return True

    __hash__ = type.__hash__


class _TensorInteger(int, metaclass=_ScalarClaimMeta):
    def detach(self) -> torch.Tensor:
        return torch.tensor([3.0], dtype=torch.float64)

    def tolist(self) -> list[float]:
        return [3.0]


class _Label(str, Enum):
    SOURCE = "source"


def test_scalar_subclasses_keep_existing_tensor_and_string_protocols() -> None:
    source = CircuitIR(
        1, (), metadata={"tensor": _TensorInteger(7), "label": _Label.SOURCE}
    )
    assert source.to_dict()["metadata"] == {
        "tensor": {"$tensor": {"dtype": "float64", "shape": [1], "data": [3.0]}},
        "label": _Label.SOURCE,
    }
    assert json.loads(source.to_json())["metadata"]["label"] == "source"


def test_nested_symbolic_complex_and_tensor_values_keep_tags_and_mutation() -> None:
    angle = torch.tensor(0.125, dtype=torch.float64)
    parameter = Parameter("theta")
    source = CircuitIR(
        1,
        (Instruction("rx", (0,), {"theta": angle}),),
        metadata=UserDict(
            {"nested": (parameter + 0.25, 1 + 2j, {"enabled": True, "note": None})}
        ),
    )
    assert source.to_dict()["metadata"] == {
        "nested": [
            {"$expression": {"op": "add", "args": [{"$parameter": "theta"}, 0.25]}},
            {"$complex": [1.0, 2.0]},
            {"enabled": True, "note": None},
        ]
    }
    before = source.to_json()
    original_hash = source.content_hash
    assert CircuitIR.from_json(before).to_json() == before
    angle.add_(0.5)
    assert source.content_hash != original_hash
    assert source.to_dict()["instructions"][0]["params"] == {
        "theta": {"$tensor": {"dtype": "float64", "shape": [], "data": 0.625}}
    }


@pytest.mark.parametrize("value", [object(), Decimal("0.25"), Fraction(1, 4)])
def test_unsupported_numeric_and_object_types_still_fail_closed(value: object) -> None:
    source = CircuitIR(1, (), metadata={"value": value})
    with pytest.raises(
        IRSerializationError, match="not deterministically serializable"
    ):
        source.to_json()
