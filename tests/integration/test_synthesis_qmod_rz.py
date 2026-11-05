"""Real literal migration agrees with native FlagQuantum and analytic phases."""

import cmath
import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("angle", ["0", "1.9", "0.25", "9007199254740992.0"])
def test_entangled_rz_matches_native_circuit_and_absolute_oracle(
    objective: Literal["width", "depth"], angle: str
) -> None:
    theta = float(angle)
    p = synth.synthesize(
        f"qfunc main(output q:qbit,output ref:qbit) {{allocate(q);H(q);ref=q;RZ({angle},q);}}",
        source_format="qmod",
        objective=objective,
    )
    result = fq.run(p)
    expected = {
        (0, 0): cmath.exp(-0.5j * theta) * 2**-0.5,
        (1, 1): cmath.exp(0.5j * theta) * 2**-0.5,
    }
    check_state(p, result, expected)
    native = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1).rz(0, theta=theta)
    torch.testing.assert_close(
        result.statevector(), fq.run(native).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("angle", ["1.5", "0.0", "(1)", "((1))"])
def test_phase_decimal_extension_preserves_old_parenthesized_integers(
    angle: str,
) -> None:
    theta = 1.0 if "(" in angle else float(angle)
    p = synth.synthesize(
        f"qfunc main(output q:qbit) {{allocate(q);H(q);PHASE({angle},q);}}",
        source_format="qmod",
    )
    check_state(p, fq.run(p), {(0,): 2**-0.5, (1,): cmath.exp(1j * theta) * 2**-0.5})


@pytest.mark.parametrize(
    "angle",
    [
        "9007199254740992.1",
        "9007199254740993",
        "1.",
        ".5",
        "1e-3",
        "unknown_angle",
        "q",
        "1 .9",
        "1. 9",
        "1/*x*/.9",
        "1.\n9",
        "0." + "0" * 77,
    ],
)
def test_invalid_or_rounded_out_of_range_literal_is_refused(angle: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            f"qfunc main(output q:qbit) {{allocate(q);RZ({angle},q);}}",
            source_format="qmod",
        )


@pytest.mark.parametrize(
    "source",
    [
        "qfunc main(output q:qbit) {RZ(1.9,q);}",
        "qfunc main(output q:qnum<1>) {allocate(q);RZ(1.9,q);}",
        "qfunc main(output q:qbit) {allocate(q);q=1.9;}",
        "qfunc main(output q:qnum<2>) {allocate(q);q=q+1.9;}",
        "qfunc main(output q:qbit) {allocate(q);RZ(1.9);}",
        "qfunc RZ(output q:qbit) {allocate(q);} qfunc main(output q:qbit) {RZ(q);}",
        "qfunc act(const q:qbit) {RZ(1.9,q);} qfunc main(output q:qbit) {allocate(q);act(q);}",
    ],
)
def test_real_gate_angles_do_not_expand_data_arithmetic_or_roles(source: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(source, source_format="qmod")


def test_tiny_nonzero_literal_is_not_rounded_to_identity() -> None:
    angle = "0." + "0" * 75 + "1"
    p = synth.synthesize(
        f"qfunc main(output q:qbit) {{allocate(q);RZ({angle},q);}}",
        source_format="qmod",
    )
    assert p.circuit.instructions[0].params["theta"] == 1e-76
    assert fq.run(p).statevector().reshape(-1)[0].imag != 0


@pytest.mark.parametrize("input_kind", ["file", "text"])
def test_original_rz_bytes_license_and_global_phase(input_kind: str) -> None:
    m = json.loads((CORPUS / "rz_sources.json").read_text())
    entry = m["fixtures"][0]
    raw = (CORPUS / "RZ.qmod").read_bytes()
    assert (
        entry["sha256"]
        == "5f9c03d460062801ae7d86cb78a20e87fa563ea40824ab8be76a09b0100d801c"
    )
    assert entry["git_blob_sha1"] == "d48bbb3be30677d16f80ceff216521f054cad3de"
    assert (
        m["license_sha256"]
        == "241f68c9f35357f97e8a2a195e447d4f80098c140f0b3b3c770a11eed96f58e0"
    )
    assert m["repository"] == "https://github.com/Classiq/classiq-library"
    assert m["commit"] == "2bfd835e73b6882fcc515a725098b04766c238e7"
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == entry["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((CORPUS / "LICENSE.txt").read_bytes()).hexdigest()
        == m["license_sha256"]
    )
    if input_kind == "file":
        p = synth.synthesize(CORPUS / "RZ.qmod", memory_limit_bytes=1024**2)
    else:
        p = synth.synthesize(
            raw.decode(), source_format="qmod", memory_limit_bytes=1024**2
        )
    check_state(p, fq.run(p), {(0,): cmath.exp(-0.95j)})
    assert p.report["resources"].n_qubits == 1
    assert p.execution.state_bytes == 32
    assert p.report["workspace_qubits"] == 0


@pytest.mark.parametrize("preparation", ["", "X(q);", "H(q);"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_rz_basis_and_coherent_states_keep_absolute_phase(
    preparation: str, objective: Literal["width", "depth"]
) -> None:
    p = synth.synthesize(
        f"qfunc main(output q:qbit) {{allocate(q);{preparation}RZ(1.9,q);}}",
        source_format="qmod",
        objective=objective,
    )
    if preparation == "H(q);":
        expected = {(0,): cmath.exp(-0.95j) / 2**0.5, (1,): cmath.exp(0.95j) / 2**0.5}
    else:
        bit = int(preparation == "X(q);")
        expected = {(bit,): cmath.exp(0.95j if bit else -0.95j)}
    check_state(p, fq.run(p), expected)


def test_rz_compatibility_retains_real_plan_budgets_and_target_refusal() -> None:
    from flagquantum.errors import PlanningError

    source = "qfunc main(output q:qbit) {allocate(q);H(q);RZ(1.9,q);}"
    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(source, source_format="qmod", memory_limit_bytes=16)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(source, source_format="qmod", max_operations=1)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(source, source_format="qmod", basis=("x", "h"))


@pytest.mark.parametrize("expression,theta", [("-1.9", -1.9), ("1.9+1", 2.9)])
def test_rz_angle_expressions_preserve_entangled_absolute_phase(
    expression: str,
    theta: float,
) -> None:
    program = synth.synthesize(
        f"qfunc main(output q:qbit,output ref:qbit) {{allocate(q);H(q);ref=q;RZ({expression},q);}}",
        source_format="qmod",
    )
    result = fq.run(program)
    check_state(
        program,
        result,
        {
            (0, 0): cmath.exp(-0.5j * theta) * 2**-0.5,
            (1, 1): cmath.exp(0.5j * theta) * 2**-0.5,
        },
    )
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1).rz(0, theta=theta)
    torch.testing.assert_close(
        result.statevector(),
        fq.run(circuit).statevector(),
        atol=1e-12,
        rtol=0,
    )
