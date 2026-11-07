"""Static Grover composition, exact record views and coherent inverse acceptance."""

import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification.amplification_iterations.comparisons.circuit import (
    run_example,
)
from flagquantum import synth
from tools.validate_qmod_iterations import CASES, validate

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("k", [0, 1, 2])
def test_unmodified_counting_and_qmci_vectors(case: str, k: int) -> None:
    assert validate(case, k)["validation"] == "passed"


def test_three_form_example() -> None:
    run_example()


def program(
    oracle: str = "Z(q[q.len-1]);",
    prepare: str = "H(q[0]);PHASE(0.37,q[0]);CX(q[0],q[1]);",
    *,
    k: int = 1,
    prefix: str = "",
    extra: str = "",
) -> str:
    return f"qfunc oracle(const q:qbit[]) {{{oracle}}} qfunc prepare(q:qbit[]) {{{prepare}}} {extra} qfunc main(output q:qbit[2]) {{allocate(q);{prefix}amplitude_amplification({k},oracle,prepare,q);}}"


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("initial_flag", [0, 1])
def test_complex_preparation_and_nonzero_marker_xor(k: int, initial_flag: int) -> None:
    p = synth.synthesize(
        program(k=k, prefix="X(q[1]);" if initial_flag else ""), source_format="qmod"
    )
    h = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    v = (
        torch.diag(
            torch.tensor([1, complex(math.cos(0.37), math.sin(0.37))], dtype=DTYPE)
        )
        @ h
    )
    perm = torch.eye(4, dtype=DTYPE)
    perm[1, 1] = perm[3, 3] = 0
    perm[1, 3] = perm[3, 1] = 1
    a = perm @ torch.kron(torch.eye(2, dtype=DTYPE), v)
    s = torch.diag(torch.tensor([1, 1, -1, -1], dtype=DTYPE))
    reflect = torch.diag(torch.tensor([1, -1, -1, -1], dtype=DTYPE))
    grover = a @ reflect @ a.conj().T @ s
    c = fq.Circuit(2, dtype=DTYPE)
    if initial_flag:
        c.x(0)
    c.any(0, 1, matrix=a)
    for _ in range(k):
        c.any(0, 1, matrix=grover)
    actual = fq.run(p).statevector().reshape(4, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(c).statevector().reshape(-1), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("oracle", ["phase(pi);", "Z(q[1]);Z(q[1]);"])
def test_all_and_none_marked_keep_absolute_sign(k: int, oracle: str) -> None:
    p = synth.synthesize(
        program(oracle, "hadamard_transform(q);", k=k), source_format="qmod"
    )
    sign = (-1) ** k if oracle == "phase(pi);" else 1
    state = fq.run(p).statevector().reshape(4, -1)
    torch.testing.assert_close(
        state[:, 0], torch.full((4,), sign / 2, dtype=DTYPE), atol=1e-12, rtol=0
    )
    assert float(state[:, 1:].abs().square().sum()) < 1e-24


@pytest.mark.parametrize(
    "oracle",
    [
        "phase(q[0],0.37);",
        "H(q[0]);",
        "X(q[0]);",
        "phase(0.5*q[0],pi);",
        "Z(other[0]);",
        "Z(q[q.len]);",
        "Z(q[0-1]);",
    ],
)
@pytest.mark.parametrize("k", [0, 1])
def test_invalid_or_captured_sign_oracles_refuse_even_zero(k: int, oracle: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(program(oracle, k=k), source_format="qmod")


@pytest.mark.parametrize("k", [-1, 257])
def test_iteration_bounds_refuse(k: int) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(program(k=k), source_format="qmod")


@pytest.mark.parametrize(
    "prepare",
    ["allocate(q);", "H(foreign[0]);", "extra:qbit;allocate(extra);", "measure(q);"],
)
def test_preparation_cannot_capture_or_change_ownership(prepare: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(program(prepare=prepare, k=0), source_format="qmod")


def test_preparation_callback_requires_inplace_port() -> None:
    source = program().replace(
        "qfunc prepare(q:qbit[])", "qfunc prepare(output q:qbit[2])"
    )
    with pytest.raises(ValueError, match="inplace"):
        synth.synthesize(source, source_format="qmod")


def test_callback_recursion_refuses() -> None:
    source = program("oracle(q);")
    with pytest.raises(ValueError, match="acyclic|recurs"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("types", ["qstruct B {x:qnum<2>;}", "qstruct B {x:qnum<1>;} "])
def test_nominal_record_types_cannot_be_reinterpreted(types: str) -> None:
    source = f"qstruct A {{x:qnum<2>;}} {types} qfunc act(a:A) {{hadamard_transform(a);}} qfunc main(output b:B) {{allocate(b);act(b);}}"
    with pytest.raises(ValueError, match="matching record type"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("width", [1, 3])
def test_untyped_record_view_requires_exact_width(width: int) -> None:
    source = f"qstruct A {{x:qnum<2>;}} qfunc act(a:A) {{hadamard_transform(a);}} qfunc main(output b:qbit[{width}]) {{allocate(b);act(b);}}"
    with pytest.raises(ValueError, match="exact declared width"):
        synth.synthesize(source, source_format="qmod")


def test_record_view_and_marker_cannot_overlap() -> None:
    source = "qstruct A {x:qnum<1>;} qfunc act(a:A,t:qbit) {H(t);} qfunc main(output b:qbit[2]) {allocate(b);act(b[0:1],b[0]);}"
    with pytest.raises(ValueError, match="disjoint"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    "body",
    [
        "CX(q[0],q[1]);",
        "CRX(0.31,q[0],q[1]);",
        "RZZ(0.43,q);",
        "phase(q[0],0.29);",
        "phase(pi);",
    ],
)
def test_pure_helper_inverse_cancels_on_complex_coherent_state(body: str) -> None:
    source = f"qfunc act(q:qbit[]) {{{body}}} qfunc main(output q:qbit[2]) {{allocate(q);hadamard_transform(q);S(q[0]);act(q);invert {{act(q);}}}}"
    p = synth.synthesize(source, source_format="qmod")
    expected = (
        fq.run(fq.Circuit(2, dtype=DTYPE).h(0).h(1).s(1)).statevector().reshape(-1)
    )
    actual = fq.run(p).statevector().reshape(4, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24


@pytest.mark.parametrize(
    "options",
    [
        {"max_qubits": 1},
        {"max_operations": 2},
        {"max_depth": 1},
        {"memory_limit_bytes": 16},
    ],
)
def test_actual_physical_resource_caps_refuse(options: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(program(), source_format="qmod", **options)


def test_iteration_products_keep_effect_expansion_budget() -> None:
    source = program(prepare="H(q[0]);PHASE(0.37,q[0]);CX(q[0],q[1]);" * 8, k=256)
    with pytest.raises(ValueError, match="4096"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("initial_flag", [0, 1])
def test_asymmetric_record_fields_keep_declared_bit_order(initial_flag: int) -> None:
    prefix = "X(q[3]);" if initial_flag else ""
    source = f"qstruct Layout {{low:qnum<1>;high:qnum<2>;}} qperm predicate(const s:Layout,t:qbit) {{t ^= (s.low==1) & (s.high==2);}} qfunc act(s:Layout,t:qbit) {{hadamard_transform(s);predicate(s,t);}} qfunc main(output q:qbit[4]) {{allocate(q);{prefix}act(q[0:3],q[3]);}}"
    program = synth.synthesize(source, source_format="qmod")
    actual = fq.run(program).statevector().reshape(16, -1)
    expected = torch.zeros(16, dtype=DTYPE)
    for word in range(8):
        # The first one-bit field is numeric bit 0; high occupies bits 1 and 2.
        marked = (word & 1) == 1 and (word >> 1) == 2
        expected[(initial_flag ^ marked) * 8 + word] = 1 / math.sqrt(8)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24
