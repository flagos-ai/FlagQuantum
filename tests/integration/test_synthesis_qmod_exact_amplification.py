"""Exact amplification semantics, strict original refusal and bounded updates."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from tools.validate_qmod_exact_amplification import FIXTURES, corrected_source, validate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "case", ["exact_amplitude_amplification", "partial_exponential_state_preparation"]
)
def test_original_and_labelled_diagnostic_full_vectors(case: str) -> None:
    report = validate(case)
    assert report["original_accepted"] == (case == "exact_amplitude_amplification")


def source(p: float, amplitude: float | None = None, before: str = "") -> str:
    amplitude = math.sqrt(p) if amplitude is None else amplitude
    return f"qfunc prep(q:qbit) {{inplace_prepare_state([{1-p:.17f},{p:.17f}],0,q);}} qfunc main(output q:qbit) {{allocate(q);{before} exact_amplitude_amplification({amplitude:.17f},Z,prep,q);}}"


@pytest.mark.parametrize("p", [0.0006, 0.001, 0.01, 0.07, 0.25, 0.5, 0.9, 1.0])
def test_rotation_round_boundaries_and_positive_phase(p: float) -> None:
    program = synth.synthesize(source(p), source_format="qmod")
    state = fq.run(program).statevector().reshape(2, -1)
    expected = torch.zeros_like(state)
    expected[1, 0] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("amplitude", [0, -0.1, 1.1, 0.07, 0.5])
def test_invalid_or_probability_instead_of_amplitude_refuses(amplitude: float) -> None:
    with pytest.raises(ValueError, match="amplification.amplitude"):
        synth.synthesize(source(0.07, amplitude), source_format="qmod")


def test_inconsistent_original_is_not_rewritten() -> None:
    with pytest.raises(ValueError, match="0.996625424765961.*0.997954578443737"):
        synth.synthesize(FIXTURES / "partial_exponential_state_preparation.qmod")


@pytest.mark.parametrize("before", ["H(q);", "X(q);X(q);", "RY(0.1,q);"])
def test_nonzero_or_unproven_zero_refuses(before: str) -> None:
    with pytest.raises(ValueError, match="proven initialized zero"):
        synth.synthesize(source(0.07, before=before), source_format="qmod")


def test_round_budget_refuses_before_expansion() -> None:
    with pytest.raises(ValueError, match="32 rounds"):
        synth.synthesize(source(0.00000001), source_format="qmod")


@pytest.mark.parametrize(
    "replacement",
    [
        "lambda(a,b){inplace_prepare_state([0.93,0.07],0,a);}",
        "lambda(a){inplace_prepare_state([0.93,0.07],0,other);}",
        "lambda(a){allocate(a);}",
    ],
)
def test_preparation_signature_capture_and_escape_refuse(replacement: str) -> None:
    with pytest.raises(ValueError, match="amplification|signature"):
        synth.synthesize(
            source(0.07).replace(",prep,q)", f",{replacement},q)"), source_format="qmod"
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_qubits", 1),
        ("max_workspace", 0),
        ("max_operations", 5),
        ("max_depth", 5),
        ("memory_limit_bytes", 16),
    ],
)
def test_actual_original_budgets_refuse(key: str, value: int) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            FIXTURES / "exact_amplitude_amplification.qmod", **{key: value}
        )


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("amount", [-256, -18, -2, 0, 2, 18, 256])
def test_constant_offset_all_basis_words(width: int, amount: int) -> None:
    for value in range(1 << width):
        program = synth.synthesize(
            f"qfunc main(output x:qnum<{width}>) {{x={value};x += {amount};}}",
            source_format="qmod",
        )
        actual = fq.run(program).statevector().reshape(1 << width, -1)
        expected = torch.zeros_like(actual)
        expected[(value + amount) % (1 << width), 0] = 1
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_zero_modular_offset_still_reads_initialization() -> None:
    with pytest.raises(ValueError, match="uninitialized"):
        synth.synthesize(
            "qfunc main(output x:qnum<2>) {x += 256;}", source_format="qmod"
        )


def test_offset_and_inverse_restore_all_coherent_phases() -> None:
    program = synth.synthesize(
        "qfunc step(x:qnum) {x += -18;} qfunc main(output x:qnum<4>) {allocate(x);hadamard_transform(x);phase(x==3,pi);step(x);invert {step(x);}}",
        source_format="qmod",
    )
    state = fq.run(program).statevector().reshape(16, -1)
    expected = torch.zeros_like(state)
    expected[:, 0] = 0.25
    expected[3, 0] = -0.25
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_nonproduct_probability_vector_refuses() -> None:
    with pytest.raises(ValueError, match="product distribution"):
        synth.synthesize(
            "qfunc prep(x:qnum) {inplace_prepare_state([0.5,0,0,0.5],0,x);} qfunc main(output x:qnum<2>) {allocate(x);exact_amplitude_amplification(1,lambda(x){phase(1,pi);},prep,x);}",
            source_format="qmod",
        )


def test_corrected_copy_requires_actual_workspace_budget() -> None:
    with pytest.raises(ValueError, match="workspace|candidate|budget"):
        synth.synthesize(corrected_source(), source_format="qmod", max_workspace=0)


def test_asymmetric_array_probability_and_numeric_bit_order() -> None:
    text = f"""qperm pred(const a:qbit[2],m:qbit) {{m ^= (a[0]==1)&(a[1]==0);}}
    qfunc prep(a:qbit[]) {{inplace_prepare_state([0.56,0.24,0.14,0.06],0,a);}}
    qfunc main(output q:qbit[2]) {{allocate(q);exact_amplitude_amplification({math.sqrt(.24):.17f},lambda(a){{phase_oracle(pred,a);}},prep,q);}}"""
    program = synth.synthesize(text, source_format="qmod")
    state = fq.run(program).statevector().reshape(4, -1)
    expected = torch.zeros_like(state)
    expected[1, 0] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_explicit_zero_assignment_is_a_valid_zero_proof() -> None:
    program = synth.synthesize(
        source(0.07).replace("allocate(q);", "q=0;"), source_format="qmod"
    )
    state = fq.run(program).statevector().reshape(2, -1)
    expected = torch.zeros_like(state)
    expected[1, 0] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_near_product_probabilities_do_not_hide_large_amplitude_defect() -> None:
    text = "qfunc main(output q:qnum<2>) {allocate(q);inplace_prepare_state([0.5,0,0.0000000000001,0.4999999999999],0,q);}"
    with pytest.raises(ValueError, match="product distribution"):
        synth.synthesize(text, source_format="qmod")


@pytest.mark.parametrize("amount", [-257, 257])
def test_offset_bound_refuses(amount: int) -> None:
    with pytest.raises(ValueError, match="literal offset"):
        synth.synthesize(
            f"qfunc main(output x:qnum<3>) {{allocate(x);x += {amount};}}",
            source_format="qmod",
        )


def test_zero_round_case_does_not_require_workspace() -> None:
    program = synth.synthesize(
        source(1), source_format="qmod", max_workspace=0, max_qubits=1
    )
    assert program.circuit.n_wires == 1


def test_one_bit_array_z_oracle_uses_numeric_low_bit() -> None:
    text = (
        source(0.07)
        .replace("prep(q:qbit)", "prep(q:qbit[])")
        .replace("output q:qbit)", "output q:qbit[1])")
    )
    state = (
        fq.run(synth.synthesize(text, source_format="qmod"))
        .statevector()
        .reshape(2, -1)
    )
    expected = torch.zeros_like(state)
    expected[1, 0] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "oracle",
    ["lambda(a){phase(a,0.5);}", "lambda(a){phase(foreign,pi);}", "lambda(a){X(a);}"],
)
def test_non_sign_or_captured_or_nondiagonal_oracle_refuses(oracle: str) -> None:
    with pytest.raises(ValueError, match="amplification"):
        synth.synthesize(
            source(0.07).replace(",Z,prep,q)", f",{oracle},prep,q)"),
            source_format="qmod",
        )


def test_output_preparation_callback_refuses() -> None:
    text = source(0.07).replace("prep(q:qbit)", "prep(output q:qbit)")
    with pytest.raises(ValueError, match="amplification.escape"):
        synth.synthesize(text, source_format="qmod")


@pytest.mark.parametrize(
    "probability,amplitude", [(0.999999999999, 1.0), (0.25, 0.500000000001)]
)
def test_close_scalar_does_not_certify_a_large_rotation_remainder(
    probability: float, amplitude: float
) -> None:
    with pytest.raises(ValueError, match="amplitude|cleanup"):
        synth.synthesize(source(probability, amplitude), source_format="qmod")


def test_large_float_pi_ratio_does_not_prove_a_sign_oracle() -> None:
    angle = math.pi * (2**48 + 1)
    assert angle / math.pi == float(2**48 + 1)
    assert abs(math.sin(angle)) > 0.05
    text = source(0.07).replace(
        ",Z,prep,q)", f",lambda(a){{phase(a,{angle:.17f});}},prep,q)"
    )
    with pytest.raises(ValueError, match="signs"):
        synth.synthesize(text, source_format="qmod")
