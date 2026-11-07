"""Exact originals have independently checked, runnable native counterparts."""

import cmath
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import (
    PHASE,
    RZ,
    RZZ,
    Const,
    Output,
    QBit,
    QNum,
    X,
    allocate,
    control,
    hadamard_transform,
    phase,
    prepare_amplitudes,
    prepare_bell_state,
    prepare_state,
    qfunc,
)
from tools.synthesis_catalog import ROOT, Case, load_catalog

pytestmark = pytest.mark.integration
NATIVE_CASES = [case for case in load_catalog()["cases"] if case["native_comparison"]]


@pytest.mark.parametrize("case", NATIVE_CASES, ids=lambda case: case["id"])
def test_original_native_and_independent_circuit(case: Case) -> None:
    assert case["native_comparison"] is not None
    path = Path(case["native_comparison"])
    module = importlib.import_module(".".join(path.with_suffix("").parts))
    module.run_example()


@qfunc
def coherent(q: Output[QBit]) -> None:
    allocate(q)
    hadamard_transform(q)
    RZ(1.9, q)
    PHASE(-0.4, q)
    X(q)


@qfunc
def coherent_pair(q: Output[QNum]) -> None:
    allocate(2, q)
    hadamard_transform(q)
    RZZ(-1.25, q)


@qfunc
def fractional_preparation(q: Output[QBit]) -> None:
    prepare_amplitudes([-3 / 5, 4 / 5], 0, q)


@qfunc
def bell_one(q: Output[QNum]) -> None:
    prepare_bell_state(1, q)


@pytest.mark.parametrize(
    ("definition", "reference"),
    [
        (
            coherent,
            fq.Circuit(1, dtype=torch.complex128)
            .h(0)
            .rz(0, theta=1.9)
            .phase(0, theta=-0.4)
            .x(0),
        ),
        (
            coherent_pair,
            fq.Circuit(2, dtype=torch.complex128)
            .h(0)
            .h(1)
            .cx(0, 1)
            .rz(1, theta=-1.25)
            .cx(0, 1),
        ),
        (
            fractional_preparation,
            fq.Circuit(1, dtype=torch.complex128).any(
                0,
                matrix=torch.tensor([[-0.6, 0.8], [0.8, 0.6]], dtype=torch.complex128),
            ),
        ),
        (bell_one, fq.Circuit(2, dtype=torch.complex128).h(0).z(0).cx(0, 1)),
    ],
)
def test_coherent_gate_phase_and_preparation_semantics(
    definition: _QuantumFunction, reference: fq.Circuit
) -> None:
    program = synth.synthesize(definition)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


def _definition(tmp_path: Path, body: str) -> _QuantumFunction:
    path = tmp_path / "native_case.py"
    path.write_text(
        "from flagquantum.synth import *\nimport math\n@qfunc\ndef main(q: Output[QNum]):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("native_case", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    definition = module.main
    assert isinstance(definition, _QuantumFunction)
    return definition


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("prepare_amplitudes([1, 1], 0, q)", "register.amplitudes"),
        ("prepare_amplitudes([math.inf, 0], 0, q)", "program.prepare"),
        ("prepare_amplitudes([1, 0, 0], 0, q)", "program.prepare"),
        ("prepare_state([-0.5, 1.5], 0, q)", "program.prepare"),
        ("prepare_state([0.4, 0.4], 0, q)", "program.prepare"),
        ("prepare_state([1/0, 0], 0, q)", "program.prepare"),
        ("prepare_state([True, 0], 0, q)", "program.prepare"),
        ("prepare_state([1, 0], -0.1, q)", "program.prepare"),
        ("prepare_state([1, 0], math.nan, q)", "program.prepare"),
        (
            "prepare_state([1, 0], 0, q)\nprepare_state([1, 0], 0, q)",
            "register.initialized",
        ),
        ("allocate(1, q)\nprepare_state([1, 0], 0, q)", "register.initialized"),
        ("allocate(3, q)\nprepare_bell_state(2, q)", "program.width"),
        ("prepare_bell_state(0, q)", "program.prepare"),
        ("prepare_ghz_state(True, q)", "program.prepare"),
        ("prepare_ghz_state(257, q)", "program.prepare"),
        ("allocate(2, q)\nRZ(1, q)", "program.width"),
        ("allocate(1, q)\nRZZ(1, q)", "program.width"),
        ("allocate(1, q)\nRZ(math.nan, q)", "program.angle"),
        ("RZ(1, q)", "program.width"),
        ("allocate(1, q)\nRZ(1, q, q)", "program.signature"),
        ("prepare_ghz_state(3, q, q)", "program.signature"),
        ("prepare_state([1, 0], q)", "program.signature"),
    ],
)
def test_preparation_and_gate_boundaries_refuse(
    tmp_path: Path, body: str, diagnostic: str
) -> None:
    with pytest.raises(HybridVerificationError, match=diagnostic):
        synth.synthesize(_definition(tmp_path, body))


def test_host_calls_are_never_evaluated(tmp_path: Path) -> None:
    marker = tmp_path / "must_not_exist"
    body = f"prepare_state([open({str(marker)!r}, 'w'), 0], 0, q)"
    with pytest.raises(HybridVerificationError, match="program.prepare"):
        synth.synthesize(_definition(tmp_path, body))
    assert not marker.exists()


def test_preparation_respects_resource_budgets() -> None:
    with pytest.raises(ValueError):
        synth.synthesize(bell_one, max_qubits=1)
    with pytest.raises(ValueError):
        synth.synthesize(bell_one, max_operations=1)
    with pytest.raises(ValueError):
        synth.synthesize(bell_one, memory_limit_bytes=1)


def test_additive_exports_follow_the_separate_unreleased_contract() -> None:
    import inspect

    contract = json.loads(
        (ROOT / "contracts/synthesis-native-catalog-candidate.json").read_text()
    )
    assert contract["implementation_authorized"] is True
    assert contract["release_authorized"] is False
    for name, signature in contract["signatures"].items():
        assert name in synth.__all__
        if name == "X":
            from flagquantum.observables import X as OBSERVABLE_X

            assert fq.X is OBSERVABLE_X
            assert fq.X is not synth.X
        else:
            assert name not in fq.__all__
        assert str(inspect.signature(getattr(synth, name))) == signature


def test_preparation_cannot_initialize_a_const_operand() -> None:
    @qfunc
    def overwrite(q: Const[QBit]) -> None:
        prepare_state([1, 0], 0, q)

    @qfunc
    def invalid(q: Output[QBit]) -> None:
        allocate(q)
        overwrite(q)

    with pytest.raises(HybridVerificationError, match="register.const"):
        synth.synthesize(invalid)


def test_rz_phase_equivalence_preserves_relative_phase_under_control() -> None:
    @qfunc
    def main(condition: Output[QBit], target: Output[QBit]) -> None:
        allocate(condition)
        allocate(target)
        hadamard_transform(condition)
        hadamard_transform(target)
        control(condition, lambda: phase(target, 1.5))
        control(condition, lambda: phase(1, -0.95))

    matrix = torch.eye(4, dtype=torch.complex128)
    matrix[2:, 2:] = torch.tensor(
        [[cmath.exp(-0.95j), 0], [0, cmath.exp(0.55j)]], dtype=torch.complex128
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1).any(0, 1, matrix=matrix)
    actual = fq.run(synth.synthesize(main)).statevector().reshape(4, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24


def test_direct_gates_retain_const_mutation_refusal() -> None:
    @qfunc
    def invalid_gate(q: Const[QBit]) -> None:
        RZ(1.9, q)

    @qfunc
    def invalid(q: Output[QBit]) -> None:
        allocate(q)
        invalid_gate(q)

    with pytest.raises(HybridVerificationError, match="register.const"):
        synth.synthesize(invalid)
