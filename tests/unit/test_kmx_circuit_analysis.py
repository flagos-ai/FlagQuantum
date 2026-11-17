"""Replay locked physical analysis SVGs at the element and attribute boundary."""

import json
import math
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest

from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.compiler.formats.kmx_circuit import inspect_kmx, read_kmx_circuit
from flagquantum.drawer.reversible_phase_analysis import (
    render_reversible_phase_analysis_svg,
)

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_circuit_analysis.json").read_text()
)
_COORDINATES = {"x", "y", "width", "height", "font-size"}


@pytest.mark.parametrize("name", tuple(_CASES))
def test_locked_physical_analysis_svg(name: str) -> None:
    case = _CASES[name]
    kmx: str = case["kmx"]
    expected = fromstring(case["svg"])
    rendered = render_reversible_phase_analysis_svg(kmx, reference_qubit_count=2)
    actual = fromstring(rendered)
    assert actual.tag == expected.tag
    assert actual.attrib == expected.attrib
    for local, source in zip(actual, expected, strict=True):
        assert local.tag == source.tag
        assert local.text == source.text
        assert local.attrib.keys() == source.attrib.keys()
        for key, value in source.attrib.items():
            if key in _COORDINATES:
                target = float(value)
                if math.isnan(target):
                    assert name == "empty"
                    assert float(local.attrib[key]) == 0
                else:
                    assert float(local.attrib[key]) == pytest.approx(target, abs=1e-5)
            else:
                assert local.attrib[key] == value
    assert (
        render_reversible_phase_analysis_svg(
            decode_kmb(encode_kmb(kmx)), reference_qubit_count=2
        )
        == rendered
    )


def test_records_preserve_order_guards_and_quoted_tags() -> None:
    kmx = 'INFO_TAG "@enter:mix"\nCSWAP q2 q1 q0 if b4\nZ_POW q1 0.125\nHMR q2 b0'
    circuit = read_kmx_circuit(kmx)
    assert circuit.qubits == 3
    records = circuit.records
    assert [record.to_kmx() for record in records] == kmx.splitlines()
    assert records[1].qubits == (2, 1, 0)
    assert records[1].condition_bit == 4
    assert records[3].result_bit == 0
    source_count = inspect_kmx(kmx)
    assert source_count.ccz_state_weight == 2
    assert dict(source_count.opcode_counts) == {
        "CSWAP_IF": 1,
        "HMR": 1,
        "INFO_TAG": 1,
        "Z_POW": 1,
    }
    assert "3 qubits, 5 CCZ states" in render_reversible_phase_analysis_svg(kmx)


def test_negative_reference_spacing_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        render_reversible_phase_analysis_svg("", reference_qubit_count=-1)
