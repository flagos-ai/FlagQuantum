"""Compare complete physical initialization grammar with unchanged source."""

import json
from pathlib import Path
from random import Random

import pytest

from flagquantum.compiler.formats._register_initialization import (
    RegisterInitializationText,
    parse_initialization,
    parse_initializations,
)
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.core._fixed_width_integer import FixedWidthInteger
from flagquantum.ecosystem import kmx_sampling_init
from flagquantum.simulation.physical_basis_shots import sample_basis_batch

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "physical_initialization.json").read_text()
)


@pytest.mark.parametrize("key", _DATA["observations"])
def test_complete_original_initialization_records(key: str) -> None:
    mode, encoded = key.split(":")
    text = bytes.fromhex(encoded).decode()
    expected = _DATA["observations"][key]
    if expected.startswith("error:"):
        with pytest.raises(ValueError) as failure:
            parse_initializations(text) if mode == "1" else parse_initialization(text)
        message = str(failure.value)
        if "Invalid character" in expected:
            assert "Invalid unsigned integer digits" in message
        elif "Empty string can't be parsed" in expected:
            assert "Empty text cannot be parsed as an integer" in message
        else:
            assert "error:" + message + "\n" == expected
        if mode == "1":
            assert message.endswith(
                "\n    Examples of valid init strings: r0=5, r1<0x500, r2=*"
            )
        return
    items = (
        parse_initializations(text) if mode == "1" else (parse_initialization(text),)
    )
    actual = f"ok:{len(items)}\n"
    for item in items:
        actual += f"{item.register}:{item.value.width}:{item.value.hexadecimal()}:{int(item.randomize)}:{item}\n"
    actual += (
        "".join(str(int(left == right)) for left in items for right in items) + "\n"
    )
    assert actual == expected


def test_equal_values_keep_distinct_declared_widths() -> None:
    small = parse_initialization("r0=1")
    padded = parse_initialization("r0=0b0001")
    assert small == padded
    assert small.value.width == 1 and padded.value.width == 4
    assert small != parse_initialization("r0<1")
    assert small != parse_initialization("r1=1")
    assert small != FixedWidthInteger(1, 1)


def test_large_zero_prefixed_index_does_not_invoke_python_decimal_limit() -> None:
    assert parse_initialization("r" + "0" * 10000 + "2=1").register == 2
    with pytest.raises(ValueError, match="out of range"):
        parse_initialization("r" + "9" * 10000 + "=1")


def test_ecosystem_binds_the_shared_parser_with_existing_numerical_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = read_kmx_circuit(
        "APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER b0 r0\nAPPEND_TO_REGISTER q1 r1\nX q0\nCX q0 q1"
    )
    original = parse_initializations
    called: list[str] = []

    def replacement(text: str) -> tuple[RegisterInitializationText, ...]:
        called.append(text)
        return original(text)

    monkeypatch.setattr(kmx_sampling_init, "parse_initializations", replacement)
    assignments = kmx_sampling_init.parse_register_initialization(circuit, "r0=3,r1=0,")
    batch = sample_basis_batch(
        circuit, shots=4, rng=Random(0), initialization=assignments, capture_before=True
    )
    assert called == ["r0=3,r1=0,"]
    assert batch.before is not None
    assert batch.before.qubits == {0: 15, 1: 0} and batch.before.bits == {0: 15}
    assert batch.qubits == {0: 0, 1: 0} and batch.bits == {0: 15}
    assert batch.phase_units == (0, 0, 0, 0) and batch.counts == {"X": 4, "CX": 4}
    with pytest.raises(ValueError, match="not declared"):
        kmx_sampling_init.parse_register_initialization(circuit, "r4294967295=0")
    with pytest.raises(ValueError, match="modulus is zero"):
        kmx_sampling_init.parse_register_initialization(circuit, "r0<0b0")
