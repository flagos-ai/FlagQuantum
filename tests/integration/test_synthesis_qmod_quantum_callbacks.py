"""Closed borrowed callback specialization and unchanged original acceptance."""

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from tools.validate_qmod_quantum_callbacks import FIXTURES, validate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("case", ["hidden_shift", "deutsch_jozsa"])
def test_unchanged_original_full_vector(case: str) -> None:
    assert validate(case)["validation"] == "passed"


BASE = """qperm pred(const x:qnum, y:qbit) { y ^= x == 1; }
qfunc wrap(f:qfunc(qnum,qbit), x:qnum, y:qbit) { f(x,y); }
qfunc main(output x:qnum<1>,output y:qbit) {allocate(x);allocate(y);hadamard_transform(x);RY(0.37,y);CALL;}"""


@pytest.mark.parametrize(
    "call", ["wrap(pred,x,y)", "wrap(lambda(a,b){pred(a,b);},x,y)"]
)
def test_named_and_lambda_xor_on_coherent_nonzero_marker(call: str) -> None:
    program = synth.synthesize(BASE.replace("CALL", call), source_format="qmod")
    expected = fq.Circuit(2, dtype=torch.complex128).h(0).ry(1, theta=0.37).cx(0, 1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "call",
    [
        "wrap(lambda(a,b){pred(a,x);},x,y)",
        "wrap(lambda(a,b){allocate(a);},x,y)",
        "wrap(lambda(a){H(a);},x,y)",
        "wrap(lambda(a,a){pred(a,a);},x,y)",
        "wrap(pred,x,x)",
        "wrap(pred,y,x)",
        "wrap(lambda(a,b){pred(a,a);},x,y)",
    ],
)
def test_invalid_callbacks_refuse(call: str) -> None:
    with pytest.raises(ValueError, match="callback|alias|signature|capture|escape"):
        synth.synthesize(BASE.replace("CALL", call), source_format="qmod")


@pytest.mark.parametrize(
    "body",
    [
        "allocate(x);",
        "H(foreign);",
        "f(x,x);",
        "wrap(f,x);",
        "within {X(x);} apply {f(x);}",
        "local:qbit;",
    ],
)
def test_unused_invalid_wrappers_refuse(body: str) -> None:
    source = f"qfunc wrap(f:qfunc(qbit[]),x:qbit[]) {{{body}}} qfunc main(output x:qbit) {{allocate(x);}}"
    with pytest.raises(ValueError, match="callback|borrowed|quantum wrapper"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_qubits", 2),
        ("max_operations", 1),
        ("max_depth", 1),
        ("memory_limit_bytes", 64),
    ],
)
def test_actual_original_resource_budgets(key: str, value: int) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(FIXTURES / "hidden_shift.qmod", **{key: value})


@pytest.mark.parametrize(
    "expression,marked",
    [
        ("max(x ^ 4,x & 3)>=4", [0, 1, 2, 3]),
        ("min(x-5,2)>=0", [5, 6, 7]),
        ("min(x,3)==2", [2]),
        ("max(x-3,0)==0", [0, 1, 2, 3]),
    ],
)
def test_numeric_predicate_extrema_all_basis_amplitudes(
    expression: str, marked: list[int]
) -> None:
    source = f"qperm pred(const x:qnum<3>,y:qbit) {{y ^= {expression};}} qfunc main(output x:qnum<3>) {{allocate(x);hadamard_transform(x);phase_oracle(pred,x);}}"
    p = synth.synthesize(source, source_format="qmod")
    expected = torch.ones(8, dtype=torch.complex128) / (8**0.5)
    expected[marked] *= -1
    torch.testing.assert_close(
        fq.run(p).statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )


def test_output_callback_signature_refuses_before_execution() -> None:
    source = BASE.replace(
        "qperm pred(const x:qnum, y:qbit) { y ^= x == 1; }",
        "qfunc pred(x:qnum,output y:qbit) {allocate(y);}",
    )
    with pytest.raises(ValueError, match="callback|signature|borrowed"):
        synth.synthesize(source.replace("CALL", "wrap(pred,x,y)"), source_format="qmod")


def test_borrowed_callback_requires_initialized_actual() -> None:
    source = BASE.replace(
        "allocate(x);allocate(y);hadamard_transform(x);RY(0.37,y);", "allocate(y);"
    )
    with pytest.raises(ValueError, match="uninitialized"):
        synth.synthesize(source.replace("CALL", "wrap(pred,x,y)"), source_format="qmod")


def test_unused_mutually_recursive_wrappers_refuse() -> None:
    source = """qfunc a(f:qfunc(qbit[]),x:qbit[]) {b(f,x);}
    qfunc b(f:qfunc(qbit[]),x:qbit[]) {a(f,x);}
    qfunc main(output x:qbit) {allocate(x);}"""
    with pytest.raises(ValueError, match="acyclic|recursion"):
        synth.synthesize(source, source_format="qmod")


def test_nested_callback_values_refuse() -> None:
    source = "qfunc wrap(f:qfunc(qbit[]),x:qbit[]) {f(x);} qfunc main(output x:qbit[1]) {allocate(x);wrap(lambda(a){hadamard_transform(a);},x);}"
    for _ in range(200):
        source = source.replace(
            "hadamard_transform(a);", "wrap(lambda(a){hadamard_transform(a);},a);", 1
        )
    with pytest.raises(ValueError, match="callback|depth|budget"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("limit", ["MAX_CALLBACK_STATEMENTS", "MAX_CALLBACK_DEPTH"])
def test_callback_expansion_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    from flagquantum.ecosystem._qmod import quantum_callbacks

    monkeypatch.setattr(quantum_callbacks, limit, 1)
    with pytest.raises(ValueError, match="callback|bounded"):
        synth.synthesize(BASE.replace("CALL", "wrap(pred,x,y)"), source_format="qmod")


def test_empty_callback_signature_is_a_structured_refusal() -> None:
    with pytest.raises(ValueError, match="callback.signature"):
        synth.synthesize(
            "qfunc wrap(f:qfunc()) {} qfunc main(output x:qbit) {allocate(x);}",
            source_format="qmod",
        )


def test_unused_unknown_predicate_refuses() -> None:
    source = "qfunc wrap(f:qfunc(qbit[]),x:qbit[]) {phase_oracle(missing,x);} qfunc main(output x:qbit) {allocate(x);}"
    with pytest.raises(ValueError, match="callback.signature"):
        synth.synthesize(source, source_format="qmod")


def test_unused_wrapper_depth_is_independent_of_definition_order() -> None:
    definitions = ["qfunc w0(f:qfunc(qbit[]),x:qbit[]) {f(x);}"]
    definitions += [
        f"qfunc w{i}(f:qfunc(qbit[]),x:qbit[]) {{w{i-1}(f,x);}}" for i in range(1, 17)
    ]
    for ordered in (definitions, list(reversed(definitions))):
        with pytest.raises(ValueError, match="callback.recursion"):
            synth.synthesize(
                "\n".join(ordered) + " qfunc main(output x:qbit) {allocate(x);}",
                source_format="qmod",
            )
