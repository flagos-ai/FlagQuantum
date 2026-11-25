"""Fresh-output swap tests preserve full-state semantics and source refusals."""

import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.state_preparation.swap_test.comparisons.circuit import (
    run_example,
)
from flagquantum import synth
from tools.validate_qmod_swap_test import DTYPE, validate

pytestmark = pytest.mark.integration


def test_original_refusal_and_separate_diagnostic_full_state() -> None:
    report = validate()
    assert report["original_accepted"] is False
    assert report["diagnostic_copy"] is True
    assert report["validation"] == "passed"


def test_three_form_user_example() -> None:
    run_example()


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("kind", ["equal", "orthogonal", "coherent", "complex"])
def test_full_vectors_and_overlap_formula(width: int, kind: str) -> None:
    prep = {
        "equal": "H(a[0]);H(b[0]);",
        "orthogonal": "X(b[0]);",
        "coherent": "hadamard_transform(a);RY(0.37,b[0]);",
        "complex": "hadamard_transform(a);S(a[0]);hadamard_transform(b);RZ(-0.81,b[0]);",
    }[kind]
    source = f"qfunc main(output t:qbit,output a:qbit[{width}],output b:qbit[{width}]) {{allocate(a);allocate(b);{prep}swap_test(a,b,t);}}"
    program = synth.synthesize(source, source_format="qmod")
    state = fq.run(program).statevector().reshape(2, 1 << width, 1 << width, -1)
    a = fq.Circuit(width, dtype=DTYPE)
    b = fq.Circuit(width, dtype=DTYPE)
    if kind == "equal":
        a.h(width - 1)
        b.h(width - 1)
    elif kind == "orthogonal":
        b.x(width - 1)
    else:
        for index in range(width):
            a.h(index)
        if kind == "coherent":
            b.ry(width - 1, theta=0.37)
        else:
            a.s(width - 1)
            for index in range(width):
                b.h(index)
            b.rz(width - 1, theta=-0.81)
    va, vb = fq.run(a).statevector().reshape(-1), fq.run(b).statevector().reshape(-1)
    ab, ba = torch.outer(va, vb), torch.outer(vb, va)
    expected = torch.stack(((ab + ba) / 2, (ab - ba) / 2))
    torch.testing.assert_close(state[..., 0], expected, atol=1e-12, rtol=0)
    assert float(state[..., 1:].abs().square().sum()) < 1e-24
    assert (
        abs(
            float(state[0].abs().square().sum())
            - (1 + float(torch.vdot(va, vb).abs().square())) / 2
        )
        < 1e-12
    )


@pytest.mark.parametrize(
    "call", ["a,a,t", "a,a[0:1],t", "a,b", "a,b,a[0]", "a,b,missing"]
)
def test_bad_signature_or_overlap_refuses(call: str) -> None:
    source = f"qfunc main(output t:qbit,output a:qbit[2],output b:qbit[2]) {{allocate(a);allocate(b);swap_test({call});}}"
    with pytest.raises(ValueError):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("prefix", ["allocate(t);allocate(a);", ""])
def test_reuse_and_uninitialized_inputs_refuse(prefix: str) -> None:
    source = f"qfunc main(output t:qbit,output a:qbit[1],output b:qbit[1]) {{{prefix}allocate(b);swap_test(a,b,t);}}"
    with pytest.raises(ValueError):
        synth.synthesize(source, source_format="qmod")


def test_different_input_widths_refuse() -> None:
    source = "qfunc main(output t:qbit,output a:qbit[1],output b:qbit[2]) {allocate(a);allocate(b);swap_test(a,b,t);}"
    with pytest.raises(ValueError, match="equal nonzero widths"):
        synth.synthesize(source, source_format="qmod")


def test_const_input_cannot_be_mutated() -> None:
    source = "qfunc compare(const a:qbit[1],const b:qbit[1],output t:qbit) {swap_test(a,b,t);} qfunc main(output a:qbit[1],output b:qbit[1],output t:qbit) {allocate(a);allocate(b);compare(a,b,t);}"
    with pytest.raises(ValueError, match="writable|Const|const"):
        synth.synthesize(source, source_format="qmod")


def test_static_helper_can_initialize_fresh_output() -> None:
    source = "qfunc compare(a:qbit[],b:qbit[],output t:qbit) {swap_test(a,b,t);} qfunc main(output t:qbit,output a:qbit[1],output b:qbit[1]) {allocate(a);allocate(b);H(a[0]);compare(a,b,t);}"
    p = synth.synthesize(source, source_format="qmod")
    expected = torch.tensor([2, 1, 1, 0, 0, -1, 1, 0], dtype=DTYPE) / (2 * math.sqrt(2))
    torch.testing.assert_close(
        fq.run(p).statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "options",
    [
        {"max_qubits": 2},
        {"max_operations": 2},
        {"max_depth": 2},
        {"memory_limit_bytes": 16},
    ],
)
def test_actual_resource_caps_refuse(options: dict[str, int]) -> None:
    source = "qfunc main(output t:qbit,output a:qbit[1],output b:qbit[1]) {allocate(a);allocate(b);H(a[0]);swap_test(a,b,t);}"
    with pytest.raises(ValueError):
        synth.synthesize(source, source_format="qmod", **options)


@pytest.mark.parametrize(
    "body",
    [
        "allocate(t);compare(a,b,t);",
        "compare(a,a,t);",
        "control(a[0]) {compare(a,b,t);}",
    ],
)
def test_forwarded_output_freshness_alias_and_control_escape_refuse(body: str) -> None:
    source = f"qfunc compare(a:qbit[],b:qbit[],output t:qbit) {{swap_test(a,b,t);}} qfunc main(output t:qbit,output a:qbit[1],output b:qbit[1]) {{allocate(a);allocate(b);{body}}}"
    with pytest.raises(ValueError):
        synth.synthesize(source, source_format="qmod")
