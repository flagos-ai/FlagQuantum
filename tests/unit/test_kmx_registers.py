"""Locked KMX writer order, register members and execution separation."""

from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.trace import ComponentCall
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_registers import (
    KmxRegister,
    decode_kmx_registers,
    encode_kmx_registers,
)

pytestmark = pytest.mark.unit
_HEADER = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=5 bits=4"\n'
_OPS = "X q0\nCX q0 q1\n"


@pytest.mark.ftqc_reference
def test_sparse_mixed_declarations_preserve_member_order_and_duplicates() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    declarations = (
        'REGISTER r2 "earlier"\n'
        "APPEND_TO_REGISTER b3 r2\n"
        "APPEND_TO_REGISTER q4 r2\n"
        "APPEND_TO_REGISTER b3 r2\n"
        r'REGISTER r2 "item\P\Q\B\r\n"'
        "\n"
    )
    source = km.Circuit(declarations + _HEADER + _OPS)
    program, registers, calls = decode_kmx_registers(declarations + _HEADER + _OPS)
    assert registers == (
        KmxRegister(),
        KmxRegister(),
        KmxRegister('item#"\\\r\n', (("b", 3), ("q", 4), ("b", 3))),
    )
    assert calls == ()
    assert encode_kmx(program) == _HEADER + _OPS
    assert program.n_qubits == source.num_qubits == 5
    assert program.n_bits == source.num_bits == 4
    assert program.input_bits == ()
    assert len(program.operations) == 2
    assert encode_kmx_registers(program, registers).strip() == str(source).strip()


@pytest.mark.ftqc_reference
def test_register_and_block_metadata_share_one_kmx_stream() -> None:
    km = pytest.importorskip("kickmix")
    program = decode_kmx(_HEADER + _OPS)
    registers = (
        KmxRegister("empty"),
        KmxRegister("quantum", (("q", 0), ("q", 1))),
        KmxRegister(members=(("b", 0),)),
    )
    calls = (ComponentCall("outer", 0, 2, children=(ComponentCall("gap", 1, 1),)),)
    encoded = encode_kmx_registers(program, registers, calls=calls)
    circuit = km.Circuit(encoded)
    assert str(circuit).strip() == encoded.strip()
    assert decode_kmx_registers(encoded) == (program, registers, calls)
    sim = km.Simulator(batch_size=1, seed=1)
    sim.do(circuit)
    assert sim.read_across_shots(km.q(0), out=int) == 1
    assert sim.read_across_shots(km.q(1), out=int) == 1
    assert sim.read_shot_phase(0) == 0


@pytest.mark.ftqc_reference
def test_duplicate_names_across_ids_are_refused_by_both_readers() -> None:
    km = pytest.importorskip("kickmix")
    declarations = 'REGISTER r0 "same"\nREGISTER r1 "same"\n'
    for reader in (km.Circuit, decode_kmx_registers):
        with pytest.raises(ValueError, match="Duplicate register name: same"):
            reader(declarations + _HEADER + _OPS)
    with pytest.raises(ValueError, match="Duplicate register name: same"):
        encode_kmx_registers(
            decode_kmx(_HEADER + _OPS), (KmxRegister("same"), KmxRegister("same"))
        )


@pytest.mark.parametrize(
    "declaration,message",
    [
        ("REGISTER r1000000", "ID is out of range"),
        ("APPEND_TO_REGISTER q0 r1000000", "ID is out of range"),
        ("APPEND_TO_REGISTER q5 r0", "member is out of range"),
        ("APPEND_TO_REGISTER b4 r0", "member is out of range"),
        ("APPEND_TO_REGISTER r0 q0", "Invalid KMX register declaration"),
        (r'REGISTER r0 "bad\t"', "Invalid KMX register declaration"),
    ],
)
def test_declaration_reader_rejects_invalid_metadata(
    declaration: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_kmx_registers(declaration + "\n" + _HEADER + _OPS)
