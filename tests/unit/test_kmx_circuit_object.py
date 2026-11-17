"""Map locked raw Circuit equality, ownership and admission to native values."""

import json
from base64 import b64decode
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from struct import pack_into

import pytest

from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128
from flagquantum.compiler.artifacts.reaction_depth import reaction_depth_for_operations
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseOperation
from flagquantum.compiler.formats.kmb import KmxRegister, decode_kmb, encode_kmb
from flagquantum.compiler.formats.kmb_convert import main as convert
from flagquantum.compiler.formats.kmb_operations import OPERATION_NAMES
from flagquantum.compiler.formats.kmx_circuit import inspect_kmx, read_kmx_circuit

_FIXTURES = Path(__file__).parent / "fixtures"
_POSITIVE = json.loads(
    (_FIXTURES / "kickmix_circuit_original_positive.json").read_text()
)
_OWNER = json.loads((_FIXTURES / "kickmix_circuit_ownership.json").read_text())


@pytest.mark.parametrize("name", tuple(_POSITIVE))
def test_original_literal_kmb_bytes(name: str) -> None:
    case = _POSITIVE[name]
    packed = b64decode(case["kmb"])
    assert encode_kmb(case["kmx"]) == packed
    assert decode_kmb(packed) == case["canonical"]
    assert read_kmx_circuit(decode_kmb(packed)) == read_kmx_circuit(case["kmx"])


def test_all_physical_records_are_owned_values() -> None:
    original = read_kmx_circuit(_OWNER["kmx"])
    shallow = copy(original)
    deep = deepcopy(original)
    assert shallow == deep == original
    assert len(original.registers) == _OWNER["registers"]
    assert (original.qubits, original.bits) == (_OWNER["qubits"], _OWNER["bits"])
    assert original.to_kmx() == _OWNER["canonical"]
    assert [record.to_kmx() for record in original] == _OWNER["records"]
    assert set(dict(inspect_kmx(original.to_kmx()).opcode_counts)) == set(
        OPERATION_NAMES.values()
    )
    with pytest.raises(FrozenInstanceError):
        setattr(original.registers[0], "name", "corruption")
    with pytest.raises(FrozenInstanceError):
        setattr(original.records[0], "opcode", "corruption")
    original = read_kmx_circuit("")
    assert (original.records, original.registers, original.qubits, original.bits) == (
        (),
        (),
        0,
        0,
    )
    assert shallow == deep and shallow != original
    assert deep.to_kmx() == _OWNER["canonical"]
    assert read_kmx_circuit(decode_kmb(encode_kmb(deep.to_kmx()))) == deep


@pytest.mark.parametrize(
    "changed",
    [
        'REGISTER r0 "other"\nAPPEND_TO_REGISTER b3 r0\nCX q0 q1 if b2\nZ_POW q0 0.125\nINFO_TAG "note"',
        'REGISTER r0 "r"\nAPPEND_TO_REGISTER b4 r0\nCX q0 q1 if b2\nZ_POW q0 0.125\nINFO_TAG "note"',
        'REGISTER r0 "r"\nAPPEND_TO_REGISTER b3 r0\nCX q1 q0 if b2\nZ_POW q0 0.125\nINFO_TAG "note"',
        'REGISTER r0 "r"\nAPPEND_TO_REGISTER b3 r0\nCX q0 q1 if b1\nZ_POW q0 0.125\nINFO_TAG "note"',
        'REGISTER r0 "r"\nAPPEND_TO_REGISTER b3 r0\nCX q0 q1 if b2\nZ_POW q0 0.25\nINFO_TAG "note"',
        'REGISTER r0 "r"\nAPPEND_TO_REGISTER b3 r0\nCX q0 q1 if b2\nZ_POW q0 0.125\nINFO_TAG "other"',
    ],
)
def test_circuit_equality_checks_order_guards_angles_and_metadata(changed: str) -> None:
    base = 'REGISTER r0 "r"\nAPPEND_TO_REGISTER b3 r0\nCX q0 q1 if b2\nZ_POW q0 0.125\nINFO_TAG "note"'
    assert read_kmx_circuit(changed) != read_kmx_circuit(base)


@pytest.mark.parametrize(
    "kmx",
    [
        'REGISTER r0 "offset',
        "REGISTER r0 offset",
        'REGISTER r0 "offset#" #test',
        r'REGISTER r0 "offset\m"',
        r'REGISTER r0 "offset\\"',
        "REGISTER r0 ''",
        'REGISTER "" r0',
        "INFO_TAG",
        "INFO_TAG # comment",
        'INFO_TAG "unclosed',
        "INFO_TAG unquoted",
        'INFO_TAG "tag#" #test',
        r'INFO_TAG "tag\m"',
        r'INFO_TAG "tag\\"',
        "INFO_TAG ''",
        'INFO_TAG "tag" if b0',
        'INFO_TAG q0 "tag"',
        'INFO_TAG "tag" q0',
    ],
)
def test_original_register_and_tag_refusals(kmx: str) -> None:
    with pytest.raises(ValueError):
        read_kmx_circuit(kmx)


def test_original_named_registers_and_tag_offsets() -> None:
    named = read_kmx_circuit(
        'REGISTER r0 "offset"\nAPPEND_TO_REGISTER q1 r0\n'
        'APPEND_TO_REGISTER b2 r1\nREGISTER r1 "target"'
    )
    assert named.registers == (
        KmxRegister("offset", (("q", 1),)),
        KmxRegister("target", (("b", 2),)),
    )
    assert read_kmx_circuit(
        'REGISTER r0 "offset"\nREGISTER r0 "overwrite"'
    ).registers == (
        KmxRegister("overwrite", ()),
    )
    for text in ('REGISTER r0 ""', "REGISTER r0", 'REGISTER r0 # "offset"'):
        assert read_kmx_circuit(text).registers == (KmxRegister("", ()),)
    text = r'REGISTER r0 "\r\n\P\Q\B"'
    assert read_kmx_circuit(text).registers == (KmxRegister('\r\n#"\\', ()),)
    assert read_kmx_circuit(text).to_kmx() == text

    tags = 'INFO_TAG "first"\nINFO_TAG ""\n' + r'INFO_TAG "\r\n\P\Q\B"'
    circuit = read_kmx_circuit(tags)
    assert [record.tag for record in circuit] == ["first", "", '\r\n#"\\']
    assert (inspect_kmx(tags).tags, inspect_kmx(tags).tag_bytes) == (3, 10)
    assert encode_kmb(tags)[-34:-10] == b"".join(
        offset.to_bytes(8, "little") for offset in (5, 5, 10)
    )
    assert circuit.to_kmx() == tags


def test_original_file_text_and_exact_phase_writes(tmp_path: Path) -> None:
    source, kmb, dest = (
        tmp_path / name for name in ("in.kmx", "bridge.kmb", "out.kmx")
    )
    text = 'REGISTER r0 "literal#hash"\r\nZ_POW q3 0.25 if b5\nZ_POW q5 0.125\n'
    source.write_text(text)
    assert convert(["kmx2kmb", "--in", str(source), "--out", str(kmb)]) == 0
    assert convert(["kmb2kmx", "--in", str(kmb), "--out", str(dest)]) == 0
    canonical = 'REGISTER r0 "literal\\Phash"\nZ_POW q3 0.25 if b5\nZ_POW q5 0.125'
    assert dest.read_bytes() == (canonical + "\n").encode()
    circuit = read_kmx_circuit(canonical)
    assert [record.angle for record in circuit] == [
        PhaseAngle128(0, 1 << 61),
        PhaseAngle128(0, 1 << 60),
    ]
    assert [record.condition_bit for record in circuit] == [5, None]
    assert circuit.to_kmx() == canonical


def test_original_all_operation_file_and_record_streams_agree(tmp_path: Path) -> None:
    original = read_kmx_circuit(_OWNER["kmx"])
    packed, dest = tmp_path / "original.kmb", tmp_path / "restored.kmx"
    packed.write_bytes(encode_kmb(_OWNER["kmx"]))
    assert convert(["kmb2kmx", "--in", str(packed), "--out", str(dest)]) == 0
    assert dest.read_bytes() == (original.to_kmx() + "\n").encode()
    assert read_kmx_circuit(dest.read_text()) == original


def test_original_equivalent_rotation_and_invert_depth() -> None:
    def depth(text: str) -> int:
        return reaction_depth_for_operations(
            ReversiblePhaseOperation(
                op.opcode.lower(), op.qubits, op.result_bit, op.condition_bit, op.angle
            )
            for op in read_kmx_circuit(text)
        )

    text = "HMR q0 b0\nZ q1 if b0\nHMR q1 b1\nX q2 if b1"
    assert depth(text) == depth(text.replace("Z q1", "Z_POW q1 1.0"))
    assert depth("HMR q0 b0\nBIT_INVERT b0") == 1


def test_original_header_and_identifier_bounds() -> None:
    for text in (
        "APPEND_TO_REGISTER q0 r5000000",
        "APPEND_TO_REGISTER q0 r4000000000",
        'REGISTER r5000000 "foo"',
        "REGISTER r1000000",
    ):
        with pytest.raises(ValueError, match="register index"):
            read_kmx_circuit(text)
    header = bytearray(encode_kmb(""))
    pack_into("<Q", header, 32, 2_000_000_001)
    with pytest.raises(
        ValueError, match="num_ops=2000000001 > MAX_CIRCUIT_INSTRUCTIONS="
    ):
        decode_kmb(bytes(header))


def test_all_unsupported_opcode_bytes_are_rejected() -> None:
    payload = bytearray(encode_kmb("NEG"))
    for code in range(256):
        if code not in OPERATION_NAMES:
            payload[120] = code
            with pytest.raises(ValueError, match="Unknown KMB operation"):
                decode_kmb(bytes(payload))
