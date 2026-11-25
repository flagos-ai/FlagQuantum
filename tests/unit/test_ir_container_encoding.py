"""Keep canonical bytes and protocol precedence for nested IR containers."""

from __future__ import annotations

import hashlib
import json

import pytest

from flagquantum.core.ir import CircuitIR
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit


def test_nested_containers_keep_bytes_keys_and_fresh_output() -> None:
    source = CircuitIR(1, (), metadata={"nested": {7: ([1, 2], (3, {False: 4}))}})
    expected = {
        "kind": "flagquantum.circuit_ir",
        "version": "1.0",
        "n_wires": 1,
        "dtype": "complex64",
        "shape": [2],
        "instructions": [],
        "observables": [],
        "measurements": [],
        "metadata": {"nested": {"7": [[1, 2], [3, {"False": 4}]]}},
    }
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    assert source.to_json() == encoded
    assert source.content_hash == hashlib.sha256(encoded.encode()).hexdigest()
    assert CircuitIR.from_json(encoded).to_json() == encoded
    exposed = source.to_dict()["metadata"]["nested"]["7"]
    exposed[0].append(99)
    exposed[1][1]["False"] = -1
    assert source.to_json() == encoded
    assert source.to_dict()["metadata"]["nested"]["7"] is not exposed


class _SymbolicDictionary(dict[str, object], Parameter):
    def __init__(self, name: str) -> None:
        Parameter.__init__(self, name)
        dict.__init__(self, ignored=object())


class _SymbolicList(list[object], Parameter):
    def __init__(self, name: str) -> None:
        Parameter.__init__(self, name)
        list.__init__(self, [object()])


@pytest.mark.parametrize("container", [_SymbolicDictionary, _SymbolicList])
def test_container_subclasses_retain_symbolic_protocol_priority(
    container: type[_SymbolicDictionary] | type[_SymbolicList],
) -> None:
    source = CircuitIR(1, (), metadata={"parameter": container("theta")})
    assert source.to_dict()["metadata"] == {"parameter": {"$parameter": "theta"}}
