"""Check original Laplacian encodings and reversible binding ownership."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def _shift_matrix(nd):
    size = 64 if nd else 128
    matrix = torch.zeros((size, size), dtype=DTYPE)
    for column in range(size):
        if nd:
            dimension = column & 1
            selector = (column >> 1) & 3
            position = column >> 3
            delta = (selector >> 1) - (1 - (selector & 1))
            shifted = (
                ((((position >> 1) + delta) % 4) << 1) | (position & 1)
                if dimension
                else (position & ~1) | ((position + delta) % 2)
            )
            row = (shifted << 3) | (selector << 1) | dimension
        else:
            position, selector = column >> 2, column & 3
            shifted = (position + (selector >> 1) - (1 - (selector & 1))) % 32
            row = (shifted << 2) | selector
        matrix[row, column] = 1
    return matrix


@pytest.mark.parametrize("coherent_position", [False, True])
@pytest.mark.parametrize(
    ("name", "blob", "nd", "workspace"),
    [
        (
            "1D_Periodic_Laplacian_BE",
            "828f035741e109e98b27338cf6654106e7715da1",
            False,
            3,
        ),
        (
            "ND_Periodic_Laplacian_BE",
            "c6354d224c2609287db33096a48a69e3fe757054",
            True,
            1,
        ),
    ],
)
def test_original_laplacian_against_independent_permutation(
    name, blob, nd, workspace, coherent_position
):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    if coherent_position:
        prepared = (
            "qfunc prepare_position(x:qbit[]) {hadamard_transform(x); S(x[0]);}"
            + raw.decode().replace(
                "allocate(2, l);", "allocate(2, l); prepare_position(j);"
            )
        )
        program = synth.synthesize(prepared, source_format="qmod")
    else:
        program = synth.synthesize(source)
    reference = fq.Circuit(program.circuit.n_wires, dtype=DTYPE)
    if coherent_position:
        width = 3 if nd else 5
        for wire in range(width):
            reference.h(wire)
        reference.s(width - 1)
    if nd:
        reference.h(3).h(4).z(3).z(4).h(5)
        reference.any(*range(6), matrix=_shift_matrix(True))
        reference.h(3).h(4).h(5)
    else:
        reference.h(5).h(6).z(5).z(6)
        reference.any(*range(7), matrix=_shift_matrix(False))
        reference.h(5).h(6).z(5).z(6)
    state = fq.run(program).statevector()
    torch.testing.assert_close(
        state, fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == workspace
    torch.testing.assert_close(
        state.reshape(-1, 1 << workspace)[:, 1:],
        torch.zeros_like(state.reshape(-1, 1 << workspace)[:, 1:]),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("amount", [1, -1])
def test_all_small_integer_inputs_increment_modulo_width(width, amount):
    for value in range(1 << width):
        source = (
            f"qfunc step(a:qnum) {{a += {amount};}}"
            f"qfunc main(output x:qnum<{width}>) {{x={value}; step(x);}}"
        )
        program = synth.synthesize(source, source_format="qmod")
        reference = fq.Circuit(program.circuit.n_wires, dtype=DTYPE)
        expected = (value + amount) % (1 << width)
        for wire in range(width):
            if expected & (1 << (width - 1 - wire)):
                reference.x(wire)
        torch.testing.assert_close(
            fq.run(program).statevector(),
            fq.run(reference).statevector(),
            atol=1e-12,
            rtol=0,
        )


def test_increment_and_inverse_restore_coherent_phase():
    source = """qfunc step(a:qnum) {a += 1;}
      qfunc main(output x:qnum<4>) {
        allocate(x); hadamard_transform(x); phase(x==1,pi);
        step(x); invert {step(x);}
      }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(program.circuit.n_wires, dtype=DTYPE)
    for wire in range(4):
        reference.h(wire)
    phase = torch.ones(16, dtype=DTYPE)
    phase[1] = -1
    reference.any(*range(4), matrix=torch.diag(phase))
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(x); x->{low,high}; H(x);", "source name is suspended"),
        ("allocate(x); x->{low,high}; {high,low}->x;", "original wire order"),
        ("allocate(x); x->{low,high}; {low,low}->x;", "names must be distinct"),
        ("allocate(x); x->{low,high}; low->x;", "widths must cover"),
        ("allocate(x); x->{low,high};", "restore every output"),
        (
            "allocate(x); x->{low,high}; {low,high}->x; H(low);",
            "source name is suspended",
        ),
        ("x->{low,high}; {low,high}->x;", "uninitialized"),
    ],
)
def test_binding_rejects_invalid_ownership(body, diagnostic):
    source = f"qfunc main(output x:qnum<3>) {{low:qnum<1>; high:qnum<2>; {body}}}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


def test_binding_refuses_overwriting_active_owner():
    source = """qfunc main(output x:qnum<2>,output y:qnum<2>) {
      allocate(x); allocate(y); x->y;
    }"""
    with pytest.raises(InteropConversionError, match="original wire order"):
        synth.synthesize(source, source_format="qmod")


def test_helper_in_nested_else_preserves_outer_coherence():
    source = """qfunc step(x:qnum) {x += 1;}
      qfunc main(output x:qnum<2>,output a:qbit,output b:qbit) {
        allocate(x); allocate(a); allocate(b); H(a); S(a); H(b);
        control(a==1) {
          control(b==0) {} else {step(x);}
        }
      }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(program.circuit.n_wires, dtype=DTYPE)
    reference.h(2).s(2).h(3).ccx(2, 3, 1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
