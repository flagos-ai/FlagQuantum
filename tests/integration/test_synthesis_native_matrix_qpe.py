"""Matrix QPE preserves numeric labels, global phase and retained environments."""

import importlib
import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CASES = (
    "algorithms/quantum_phase_estimation/qpe_for_matrix",
    "tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix",
)


@pytest.mark.parametrize("directory", CASES)
def test_original_native_and_independent_complete_environment(directory: str) -> None:
    module = "examples.synthesis." + directory.replace("/", ".")
    entry = importlib.import_module(module + ".main")
    reference = importlib.import_module(module + ".comparisons.circuit")
    source = (
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original"
        / directory
        / (Path(directory).name + ".qmod")
    )
    original = synth.synthesize(source)
    native = entry.build_program()
    expected = fq.run(reference.reference_circuit()).statevector().flatten()
    for program in (original, native):
        torch.testing.assert_close(
            fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
        )
        assert program.report["workspace_qubits"] == 0
    assert original.report["resources"] == native.report["resources"]
    assert tuple(native.registers) == ("phase_result",)


@pytest.mark.parametrize("directory", CASES)
@pytest.mark.parametrize("variant", [0, 1])
def test_nonzero_vectors_signed_fractions_and_environment_marginal(
    directory: str, variant: int
) -> None:
    module = "examples.synthesis." + directory.replace("/", ".")
    entry = importlib.import_module(module + ".main")
    reference = importlib.import_module(module + ".comparisons.circuit")
    values = tuple(
        ((-1) ** (i + variant)) * (i + 1 + variant)
        for i in range(1 << reference.DATA_BITS)
    )
    amplitudes = tuple(v / math.hypot(*values) for v in values)
    program = entry.build_program(amplitudes)
    result = fq.run(program)
    expected = fq.run(reference.reference_circuit(amplitudes)).statevector().flatten()
    torch.testing.assert_close(
        result.statevector().flatten(), expected, atol=1e-12, rtol=0
    )
    bits = reference.PHASE_BITS
    expected_marginal = expected.reshape(1 << bits, -1).abs().square().sum(dim=1)
    raw = dict(program.probabilities(result, registers=("phase_result",)))
    numeric = dict(program.numeric_probabilities(result, registers=("phase_result",)))
    signed = bits == 7
    for word in range(1 << bits):
        assert raw.get((word,), 0) == pytest.approx(
            float(expected_marginal[word]), abs=1e-12
        )
        integer = word - (1 << bits) if signed and word >= (1 << (bits - 1)) else word
        assert numeric.get((Fraction(integer, 1 << bits),), 0) == pytest.approx(
            raw.get((word,), 0), abs=1e-12
        )
    assert (
        float(result.statevector().reshape(1 << bits, -1)[:, 1:].abs().square().sum())
        > 0
    )


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("signed", [False, True])
def test_controlled_negative_identity_retains_interfering_global_phase(
    bits: int, signed: bool
) -> None:
    program = synth.synthesize(
        definition(
            f"p:Output[QNum[{bits},{signed},{bits}]]",
            "s:QBit=QBit()\nallocate(s)\nH(s)\nS(s)\nallocate(p)\nqpe(lambda:unitary([[-1,0],[0,-1]],s),p)\ndrop(s)",
        ),
        max_workspace=0,
    )
    state = fq.run(program).statevector().flatten()
    expected = torch.zeros((1 << bits, 2), dtype=torch.complex128)
    expected[1 << (bits - 1)] = torch.tensor(
        [1, 1j], dtype=torch.complex128
    ) / math.sqrt(2)
    torch.testing.assert_close(state, expected.flatten(), atol=1e-12, rtol=0)
    assert dict(
        program.probabilities(fq.run(program), registers=("p",))
    ) == pytest.approx({(1 << (bits - 1),): 1})


@pytest.mark.parametrize(
    "counter,body",
    [
        ("QNum[3]", "qpe(lambda:unitary([[1,0],[0,1]],s),p)"),
        ("QNum[3,False,2]", "qpe(lambda:unitary([[1,0],[0,1]],s),p)"),
        ("QNum[9,False,9]", "qpe(lambda:unitary([[1,0],[0,1]],s),p)"),
        ("QNum[3,False,3]", "H(p)\nqpe(lambda:unitary([[1,0],[0,1]],s),p)"),
        ("QNum[3,False,3]", "qpe(lambda:H(s),p)"),
        ("QNum[3,False,3]", "qpe(lambda:drop(s),p)"),
        ("QNum[3,False,3]", "qpe(lambda:allocate(s),p)"),
        ("QNum[3,False,3]", "qpe(lambda a:unitary([[1,0],[0,1]],s),p)"),
        ("QNum[3,False,3]", "qpe(lambda:unitary([[1,0],[0,1]],s,extra=1),p)"),
        (
            "QNum[3,False,3]",
            "qpe(lambda:unitary([[1,0],[0,1]],s),p)\nqpe(lambda:unitary([[1,0],[0,1]],s),p)",
        ),
        (
            "QNum[3,False,3]",
            "for i in range(1):\n    qpe(lambda:unitary([[1,0],[0,1]],s),p)",
        ),
        ("QNum[3,False,3]", "if True:\n    qpe(lambda:unitary([[1,0],[0,1]],s),p)"),
        ("QNum[1,False,1]", "qpe(lambda:unitary([[1,0],[0,1]],p),p)"),
        ("QNum[3,False,3]", "qpe(lambda:unitary([[1,1],[0,1]],s),p)"),
    ],
)
def test_callback_alias_numeric_freshness_and_region_refusals(
    counter: str, body: str
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                f"p:Output[{counter}]",
                "s:QBit=QBit()\nallocate(s)\nallocate(p)\n" + body + "\ndrop(s)",
            )
        )


@pytest.mark.parametrize("setup", ["allocate(p)", "allocate(s)"])
def test_initialization_is_required_for_both_counter_and_data(setup: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "p:Output[QNum[2,False,2]]",
                "s:QBit=QBit()\n"
                + setup
                + "\nqpe(lambda:unitary([[1,0],[0,1]],s),p)\ndrop(s)",
            )
        )


@pytest.mark.parametrize("budget", ["width", "operations", "depth", "memory"])
def test_actual_resources_are_charged(budget: str) -> None:
    entry = importlib.import_module(
        "examples.synthesis.algorithms.quantum_phase_estimation.qpe_for_matrix.main"
    )
    bindings = {"amplitudes": list(entry.INITIAL_VECTOR)}
    with pytest.raises((HybridVerificationError, PlanningError, synth.SynthesisError)):
        if budget == "width":
            synth.synthesize(entry.main, classical_inputs=bindings, max_qubits=8)
        elif budget == "operations":
            synth.synthesize(entry.main, classical_inputs=bindings, max_operations=1)
        elif budget == "depth":
            synth.synthesize(entry.main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(
                entry.main, classical_inputs=bindings, memory_limit_bytes=16
            )


def test_host_callback_is_not_executed() -> None:
    with pytest.raises(TypeError, match="captured"):
        getattr(synth, "qpe")(
            lambda: (_ for _ in ()).throw(AssertionError("host callback executed")),
            None,
        )
