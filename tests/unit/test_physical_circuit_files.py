# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Main scenarios adapted from locked Kickmix py/main_test.py.

"""Close main-registration workflows through existing native owning modules."""

import io
import json
from base64 import b64decode
from pathlib import Path

import pytest

from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.core.ir import Instruction
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession
from flagquantum.ecosystem._physical_circuit_files import (
    read_physical_circuit,
    write_physical_circuit,
)

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "physical_circuit_files.json").read_text()
)


@pytest.mark.parametrize("case", _DATA["cases"])
@pytest.mark.parametrize(
    "format,kind",
    [
        (format, kind)
        for format in ("kmx", "kmb")
        for kind in ("path", "str", "bytes", "binary")
    ]
    + [("kmx", "text")],
)
def test_original_file_routes_and_exact_payloads(
    case: dict[str, object], format: str, kind: str, tmp_path: Path
) -> None:
    circuit = read_kmx_circuit(str(case["input"]))
    payloads = case["payloads"]
    assert isinstance(payloads, dict)
    expected = b64decode(payloads[format])
    destination: object
    path = tmp_path / ("data." + format)
    if kind == "binary":
        destination = io.BytesIO()
    elif kind == "text":
        destination = io.StringIO()
    elif kind == "str":
        destination = str(path)
    elif kind == "bytes":
        destination = bytes(path)
    else:
        destination = path
    write_physical_circuit(circuit, destination, format)
    if isinstance(destination, (io.BytesIO, io.StringIO)):
        observed = destination.getvalue()
        assert (
            observed.encode() if isinstance(observed, str) else observed
        ) == expected
        destination.seek(0)
    else:
        assert path.read_bytes() == expected
    assert read_physical_circuit(destination, format) == circuit
    if isinstance(destination, (io.BytesIO, io.StringIO)):
        assert not destination.closed
        destination.seek(0)
    assert read_physical_circuit(destination) == circuit


@pytest.mark.parametrize("probe", _DATA["probes"])
def test_source_distinguishes_text_constructor_and_binary_file_grammar(
    probe: dict[str, object],
) -> None:
    text = str(probe["text"])
    source = io.StringIO(text) if probe["text_stream"] else io.BytesIO(text.encode())
    if "error" in probe:
        with pytest.raises(ValueError):
            read_physical_circuit(source)
    else:
        assert read_physical_circuit(source).to_kmx() == probe["value"]


@pytest.mark.parametrize("case", _DATA["cases"])
def test_complete_main_resource_projections(case: dict[str, object]) -> None:
    circuit = read_kmx_circuit(str(case["input"]))
    stats = PhysicalOperationStatistics.from_circuit(circuit)
    assert dict(stats) == case["counts"]
    assert stats.num_instructions == case["instructions"]
    assert stats.num_toffoli_like == case["toffoli"]
    assert stats.num_t_like == case["t"]
    levels = case["levels"]
    assert isinstance(levels, dict)
    assert stats.z_pow_clifford_hierarchy_counts == {
        int(k): v for k, v in levels.items()
    }
    assert stats.items() == list(dict(stats).items())
    assert stats.values() == list(dict(stats).values())
    assert list(stats) == stats.keys()
    assert stats["CZ"] == 0
    assert stats.get("NOT_AN_OP", 99) == 99
    with pytest.raises(KeyError):
        _ = stats["NOT_AN_OP"]
    assert stats == PhysicalOperationStatistics.from_circuit(
        read_kmx_circuit(circuit.to_kmx())
    )


def test_binary_and_text_open_file_objects(tmp_path: Path) -> None:
    circuit = read_kmx_circuit(str(_DATA["cases"][2]["input"]))
    for format, mode in (("kmx", "w"), ("kmx", "wb"), ("kmb", "wb")):
        path = tmp_path / (format + mode)
        with path.open(mode) as output:
            write_physical_circuit(circuit, output, format)
            assert not output.closed
        with path.open("r" if mode == "w" else "rb") as source:
            assert read_physical_circuit(source, format) == circuit
            source.seek(0)
            assert read_physical_circuit(source) == circuit
            assert not source.closed


def test_failures_preserve_format_check_order_and_stream_ownership(
    tmp_path: Path,
) -> None:
    circuit = read_kmx_circuit("X q0")
    absent = tmp_path / "not_created"
    with pytest.raises(ValueError, match="Unrecognized format"):
        write_physical_circuit(circuit, absent, "bmx")
    assert not absent.exists()
    with pytest.raises(ValueError, match="Unrecognized format"):
        read_physical_circuit(absent, "bmx")
    for format in ("auto", "kmx", "kmb"):
        with pytest.raises(TypeError):
            read_physical_circuit(123, format)
    with pytest.raises(TypeError):
        write_physical_circuit(circuit, 123)
    with pytest.raises(ValueError, match="Failed to open file for reading"):
        read_physical_circuit(absent)
    with pytest.raises(ValueError, match="Failed to open file for writing"):
        write_physical_circuit(circuit, absent / "missing")
    with pytest.raises(ValueError, match="binary file"):
        read_physical_circuit(io.StringIO("X q0"), "kmb")
    with pytest.raises((TypeError, ValueError)):
        write_physical_circuit(circuit, io.StringIO(), "kmb")
    binary = io.BytesIO()
    write_physical_circuit(circuit, binary, "kmb")
    binary.seek(0)
    with pytest.raises(ValueError):
        read_physical_circuit(binary, "kmx")
    with pytest.raises(ValueError):
        read_physical_circuit(io.BytesIO(b"X q0"), "kmb")
    with pytest.raises(ValueError):
        read_kmx_circuit("test")


def test_file_roundtrip_reuses_numerical_and_normal_append_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.algorithms import unary_address_traversal as consumer
    from flagquantum.compiler.artifacts.operand_identity import (
        OperandIdentity,
        OperandKind,
    )

    session = PhysicalBasisSession(1, debug_print=False)
    target = OperandIdentity(OperandKind.QUANTUM, 9)
    assert not session.read_within_shot(target, 0)
    stream = io.BytesIO()
    write_physical_circuit(read_kmx_circuit("X q0\nCX q0 q9"), stream, "kmb")
    stream.seek(0)
    session.do(read_physical_circuit(stream))
    assert session.read_within_shot(target, 0)

    monkeypatch.setattr(
        consumer, "bind_reversible_phase_execution", _file_bound_instructions
    )
    consumer.main()


def _file_bound_instructions(
    program: ReversiblePhaseProgram,
) -> tuple[Instruction, ...]:
    stream = io.BytesIO()
    write_physical_circuit(read_kmx_circuit(encode_kmx(program)), stream, "kmb")
    stream.seek(0)
    restored = decode_kmx(read_physical_circuit(stream).to_kmx())
    assert restored == program
    return bind_reversible_phase_execution(restored)


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("park", [False, True])
def test_file_replacement_preserves_every_small_complex_branch(
    n: int, controlled: bool, park: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit import test_kickmix_unary_iteration as accepted

    monkeypatch.setattr(
        accepted, "bind_reversible_phase_execution", _file_bound_instructions
    )
    accepted.test_all_reset_and_erasure_branches_preserve_complex_amplitudes(
        n, controlled, park
    )
