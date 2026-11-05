"""Ordinary QPE preserves complete phase-sensitive vectors and source ownership."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tools.validate_qmod_static_qpe import CASES, validate

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


def source(
    body: str, *, width: int = 3, setup: str = "allocate(data);", suffix: str = ""
) -> str:
    return f"""qfunc main(output counter:qnum<{width},UNSIGNED,{width}>,output data:qbit) {{
        allocate(counter);{setup} qpe(lambda(){{{body}}},counter);{suffix}
    }}"""


@pytest.mark.parametrize("case", CASES)
def test_unchanged_original_against_independent_matrix_powers(case: str) -> None:
    assert validate(case)["validation"] == "passed"


@pytest.mark.parametrize(
    "body,matrix",
    [
        ("phase(pi);", -torch.eye(2, dtype=DTYPE)),
        ("H(data);", torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)),
        (
            "RX(0.43,data);",
            torch.tensor(
                [
                    [math.cos(0.215), -1j * math.sin(0.215)],
                    [-1j * math.sin(0.215), math.cos(0.215)],
                ],
                dtype=DTYPE,
            ),
        ),
        (
            "RY(-0.37,data);",
            torch.tensor(
                [
                    [math.cos(0.185), math.sin(0.185)],
                    [-math.sin(0.185), math.cos(0.185)],
                ],
                dtype=DTYPE,
            ),
        ),
        (
            "RZ(0.31,data);",
            torch.diag(torch.exp(torch.tensor([-0.155j, 0.155j], dtype=DTYPE))),
        ),
        (
            "SX(data);",
            torch.tensor([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=DTYPE) / 2,
        ),
        ("invert{S(data);}", torch.diag(torch.tensor([1, -1j], dtype=DTYPE))),
    ],
)
def test_complex_callbacks_preserve_global_phase(
    body: str, matrix: torch.Tensor
) -> None:
    p = synth.synthesize(
        source(body, setup="allocate(data);H(data);S(data);"), source_format="qmod"
    )
    c = fq.Circuit(4, dtype=DTYPE).h(3).s(3).h(0).h(1).h(2)
    identity = torch.eye(2, dtype=DTYPE)
    for bit in range(3):
        c.any(
            2 - bit,
            3,
            matrix=torch.block_diag(
                identity, torch.linalg.matrix_power(matrix, 1 << bit)
            ),
        )
    axis = torch.arange(8, dtype=torch.float64)
    c.any(
        0,
        1,
        2,
        matrix=torch.exp(-2j * math.pi * torch.outer(axis, axis) / 8) / math.sqrt(8),
    )
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )
    assert p.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "program",
    [
        source("X(counter);"),
        source("allocate(data);"),
        source("drop(data);"),
        source("H(data);", setup=""),
        source("H(data);", suffix="H(counter);"),
        source("H(data);", suffix="qpe(lambda(){X(data);},counter);"),
        source("H(data);", width=9),
        source(""),
        "qfunc main(output p:qnum<2>){allocate(p);qpe(lambda(){phase(pi);},p);}",
        "qfunc main(output p:qnum<2,UNSIGNED,1>){allocate(p);qpe(lambda(){phase(pi);},p);}",
        "qfunc main(output p:qnum<2,UNSIGNED,2>){qpe(lambda(){phase(pi);},p);}",
    ],
)
def test_reject_counter_alias_ownership_escape_and_invalid_phase(program: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(program, source_format="qmod")


@pytest.mark.parametrize("body", ["X(q[0]);", "allocate(q);", "phase(secret,pi);"])
def test_inplace_grover_oracle_still_requires_closed_diagonal_proof(body: str) -> None:
    program = f"""qfunc oracle(q:qbit[]){{{body}}}
    qfunc main(output p:qnum<2,UNSIGNED,2>,output data:qbit[1]){{
    allocate(p);allocate(data);qpe(lambda(){{grover_operator(oracle,hadamard_transform,data);}},p);}}"""
    with pytest.raises(InteropConversionError):
        synth.synthesize(program, source_format="qmod")


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 3},
        {"max_operations": 1},
        {"max_depth": 1},
        {"memory_limit_bytes": 1},
    ],
)
def test_resource_caps_are_preserved(limits: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(source("H(data);"), source_format="qmod", **limits)


def test_native_scalar_qfunc_example() -> None:
    from examples.synthesis_static_qpe import main

    main()


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("typed_allocation", [False, True])
def test_literal_width_binding_and_signed_fixed_point_metadata(
    signed: bool, typed_allocation: bool
) -> None:
    kind = (
        "qnum"
        if typed_allocation
        else "qnum<RESOLUTION," + ("SIGNED" if signed else "UNSIGNED") + ",RESOLUTION>"
    )
    allocation = (
        f"allocate(3,{signed},3,counter);" if typed_allocation else "allocate(counter);"
    )
    program = f"""RESOLUTION:int=3;
    qfunc main(output counter:{kind},output data:qbit){{
    {allocation}allocate(data);qpe(lambda(){{phase(pi);}},counter);}}"""
    p = synth.synthesize(program, source_format="qmod")
    result = fq.run(p)
    expected = torch.zeros(16, dtype=DTYPE)
    expected[8] = 1
    torch.testing.assert_close(
        result.statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )
    assert p.report["qmod_fractional_bits"] == {"counter": 3}
    assert ("counter" in p.report.get("qmod_signed_registers", ())) == signed
    assert dict(p.probabilities(result, registers=("counter",))) == pytest.approx(
        {(4,): 1.0}
    )


def test_qperm_cannot_hide_qpe_inside_a_callback_scope() -> None:
    program = """qperm hidden(p:qnum<2>,q:qbit){qpe(lambda(){X(q);},p);}
    qfunc main(output data:qbit){allocate(data);}"""
    with pytest.raises(InteropConversionError):
        synth.synthesize(program, source_format="qmod")


def test_composed_cx_and_cz_callback_preserves_entangled_amplitudes() -> None:
    text = """qfunc main(output p:qnum<3,UNSIGNED,3>,output a:qbit,output b:qbit){
      allocate(p);allocate(a);allocate(b);H(a);S(a);H(b);
      qpe(lambda(){CX(a,b);CZ(a,b);},p);
    }"""
    p = synth.synthesize(text, source_format="qmod")
    cx = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=DTYPE
    )
    unitary = torch.diag(torch.tensor([1, 1, 1, -1], dtype=DTYPE)) @ cx
    c = fq.Circuit(5, dtype=DTYPE).h(3).s(3).h(4).h(0).h(1).h(2)
    for bit in range(3):
        c.any(
            2 - bit,
            3,
            4,
            matrix=torch.block_diag(
                torch.eye(4, dtype=DTYPE), torch.linalg.matrix_power(unitary, 1 << bit)
            ),
        )
    axis = torch.arange(8, dtype=torch.float64)
    c.any(
        0,
        1,
        2,
        matrix=torch.exp(-2j * math.pi * torch.outer(axis, axis) / 8) / math.sqrt(8),
    )
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("literal", ["[]", "[[1,0],[1]]", "[[1" + ",0" * 16 + "]]"])
def test_matrix_constants_have_closed_storage_bounds(literal: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            f"m:real[][]={literal};qfunc main(output q:qbit){{allocate(q);}}",
            source_format="qmod",
        )
