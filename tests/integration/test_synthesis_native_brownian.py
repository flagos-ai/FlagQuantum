"""Native quantization preserves complete Brownian scope and refuses unsafe use."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.brownian_chebyshev_polynomials.comparisons.circuit import (
    prices,
    reference_circuit,
)
from examples.synthesis.applications.finance.brownian_chebyshev_polynomials.main import (
    build_program,
    main,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.integration.test_synthesis_qmod_brownian_quantized import _original
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/applications/finance/brownian_chebyshev_polynomials/main.py"
)


@pytest.mark.parametrize("coefficient", [0.450158158078553, 0.431573])
def test_complete_original_scope_native_circuit_and_reduced_outputs(
    coefficient: float,
) -> None:
    source = SOURCE.read_text().split('if __name__ == "__main__":')[0]
    source = source.replace("0.450158158078553", repr(coefficient))
    native = synth.synthesize(
        program_definition(source),
        max_qubits=23,
        max_workspace=2,
        max_operations=3000,
        memory_limit_bytes=256 * 1024**2,
    )
    expected = fq.run(reference_circuit(coefficient)).statevector().flatten()
    result = fq.run(native)
    actual = result.statevector().flatten()
    assert native.registers == {"t": (0,), "price": tuple(range(1, 16))}
    assert native._checked.numeric_formats == (("t", False, 0), ("price", True, 8))
    assert actual.numel() == expected.numel() == 1 << 23
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    support = torch.tensor(
        [
            ((row & 1) << 22) | ((value & 32767) << 7) | (row << 2)
            for row, value in enumerate(prices(coefficient))
        ]
    )
    assert torch.equal(
        torch.nonzero(actual.abs() > 1e-12).flatten(), support.sort().values
    )
    torch.testing.assert_close(
        actual[support],
        torch.full((32,), 1 / 32**0.5, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    assert float(actual.reshape(-1, 4)[:, 1:].abs().square().sum()) < 1e-24
    # The complete dense 16-bit reduced density would need 64 GiB. Prove zero
    # off-support blocks above and compare every entry of its active subspace.
    labels = torch.unique(support >> 7)
    a = actual.reshape(65536, 128)[labels]
    b = expected.reshape(65536, 128)[labels]
    torch.testing.assert_close(a @ a.mH, b @ b.mH, atol=1e-12, rtol=0)
    decoded = native.numeric_probabilities(
        result, registers=("t", "price"), max_amplitudes=1 << 23
    )
    expected_probabilities: dict[tuple[Fraction, Fraction], float] = {}
    for row, value in enumerate(prices(coefficient)):
        key = Fraction(row & 1), Fraction(value, 256)
        expected_probabilities[key] = expected_probabilities.get(key, 0) + 1 / 32
    assert dict(decoded) == pytest.approx(expected_probabilities, abs=1e-12)
    if coefficient == 0.450158158078553:
        assert len(expected_probabilities) == 24
        assert (min(prices(coefficient)), max(prices(coefficient))) == (-1874, 2132)
        original = synth.synthesize(_original(), max_qubits=23, max_workspace=2)
        torch.testing.assert_close(
            actual, fq.run(original).statevector().flatten(), atol=1e-12, rtol=0
        )
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    expected_resources = (
        (23, 1403, 1150) if coefficient == 0.450158158078553 else (23, 820, 668)
    )
    assert (
        resources.n_qubits,
        resources.n_operations,
        resources.depth,
    ) == expected_resources
    assert native.report["workspace_qubits"] == 2


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("0.001953125", Fraction(0)),
        ("-0.001953125", Fraction(-1, 256)),
        ("0.1", Fraction(25, 256)),
        ("-0.1", Fraction(-26, 256)),
    ],
)
def test_explicit_rounding_boundaries_and_exact_unmarked_composition(
    expression: str, expected: Fraction
) -> None:
    entry = definition(
        "result:Output[QNum]",
        f"result |= quantize({expression},fractional_bits=8,literal_bits=20)",
    )
    program = synth.synthesize(entry)
    assert dict(program.numeric_probabilities(fq.run(program))) == {(expected,): 1.0}
    exact = definition(
        "result:Output[QNum]",
        "a:QNum=QNum()\na |= quantize(0.125,fractional_bits=8,literal_bits=20)\nresult |= a*0.5+1\ndrop(a)",
    )
    p = synth.synthesize(exact)
    assert dict(p.numeric_probabilities(fq.run(p))) == {(Fraction(17, 16),): 1.0}


@pytest.mark.parametrize(
    "rhs",
    [
        "quantize(0.1)",
        "quantize(0.1,fractional_bits=True,literal_bits=20)",
        "quantize(0.1,fractional_bits=7,literal_bits=20)",
        "quantize(0.1,fractional_bits=8,literal_bits=20.0)",
        "quantize(0.1,8,literal_bits=20)",
        "quantize(quantize(0.1,fractional_bits=8,literal_bits=20),fractional_bits=8,literal_bits=20)",
        "quantize(2**5,fractional_bits=8,literal_bits=20)",
        "quantize(2**1.0,fractional_bits=8,literal_bits=20)",
        "quantize(True,fractional_bits=8,literal_bits=20)",
        "quantize(0.1/2,fractional_bits=8,literal_bits=20)",
    ],
)
def test_marker_rejects_unsupported_precision_and_syntax_at_source(rhs: str) -> None:
    entry = definition("result:Output[QNum]", f"result |= {rhs}")
    with pytest.raises(HybridVerificationError, match="construction.python.*quantized"):
        synth.synthesize(entry)


@pytest.mark.parametrize(
    "body,helper",
    [
        (
            "allocate(1,result)\nresult |= quantize(0.1,fractional_bits=8,literal_bits=20)",
            "",
        ),
        ("result ^= quantize(0.1,fractional_bits=8,literal_bits=20)", ""),
        ("quantize(0.1,fractional_bits=8,literal_bits=20)\nresult |= 0", ""),
        (
            "a:QNum=QNum()\na |= quantize(0.1,fractional_bits=8,literal_bits=20)\ndrop(a)\nresult |= a",
            "",
        ),
        (
            "a:QNum=QNum()\na |= quantize(0.1,fractional_bits=8,literal_bits=20)\nf(a,a,result)\ndrop(a)",
            "@qfunc\ndef f(a:Const[QNum],b:Const[QNum],out:Output[QNum]):\n    out |= a+b\n",
        ),
        (
            "a:QNum=QNum()\na |= quantize(0.1,fractional_bits=8,literal_bits=20)\nf(a)\nresult |= a\ndrop(a)",
            "@qfunc\ndef f(a:Const[QNum]):\n    g(a)\n@qfunc\ndef g(out:Output[QNum]):\n    out |= 0\n",
        ),
        ("result |= quantize(0.1,fractional_bits=8,literal_bits=20)\nresult |= 0", ""),
        (
            "result |= quantize(0.1,fractional_bits=8,literal_bits=20)\nphase(result,0.3)",
            "",
        ),
        (
            "result |= quantize(0.1,fractional_bits=8,literal_bits=20)\nf(result)",
            "@qfunc\ndef f(a:Const[QNum]):\n    f(a)\n",
        ),
        (
            "result |= quantize(0.1,fractional_bits=8,literal_bits=20)\na:QNum=QNum()\na |= 0.1\ndrop(a)",
            "",
        ),
    ],
)
def test_ownership_borrowing_and_unmarked_arithmetic_stay_checked(
    body: str, helper: str
) -> None:
    entry = definition("result:Output[QNum]", body, helper)
    with pytest.raises(HybridVerificationError, match="construction.python.*quantized"):
        synth.synthesize(entry)


def test_incomplete_discard_environment_and_independent_input_budget_refuse() -> None:
    entry = definition(
        "x:Output[QNum[1]],result:Output[QNum]",
        "allocate(x)\nhadamard_transform(x)\na:QNum=QNum()\na |= quantize(x-x,fractional_bits=8,literal_bits=20)\nresult |= 0\ndrop(a)",
    )
    with pytest.raises(
        HybridVerificationError, match="distinguish every prepared branch"
    ):
        synth.synthesize(entry)
    wide = definition(
        "x:Output[QNum[6]],result:Output[QNum]",
        "allocate(x)\nhadamard_transform(x)\nresult |= quantize(x,fractional_bits=8,literal_bits=20)",
    )
    with pytest.raises(HybridVerificationError, match="one bit"):
        synth.synthesize(wide)
    bits = "\n".join(
        f"a{i}:QNum=QNum()\nallocate(1,a{i})\nhadamard_transform(a{i})"
        for i in range(6)
    )
    entry = definition(
        "result:Output[QNum]",
        bits
        + "\nresult |= quantize(0,fractional_bits=8,literal_bits=20)\n"
        + "\n".join(f"drop(a{i})" for i in range(6)),
    )
    with pytest.raises(HybridVerificationError, match="five independent"):
        synth.synthesize(entry)


@pytest.mark.parametrize(
    "max_qubits,max_workspace,max_operations,memory_limit_bytes",
    [
        (22, None, None, None),
        (None, 1, None, None),
        (None, None, 1402, None),
        (None, None, None, 1024**2),
    ],
)
def test_real_emitted_resource_budgets_are_enforced(
    max_qubits: int | None,
    max_workspace: int | None,
    max_operations: int | None,
    memory_limit_bytes: int | None,
) -> None:
    with pytest.raises((PlanningError, HybridVerificationError)):
        synth.synthesize(
            main,
            max_qubits=max_qubits,
            max_workspace=max_workspace,
            max_operations=max_operations,
            memory_limit_bytes=memory_limit_bytes,
        )


def test_default_public_builder_preserves_original_limits() -> None:
    program = build_program()
    assert program.circuit.n_wires == 23


def test_inferred_helper_formats_include_sign_and_scale_without_false_escape() -> None:
    helper = "@qfunc\ndef negate(x:Const[QNum],out:Output[QNum]):\n    out |= -x\n"
    entry = definition(
        "result:Output[QNum]",
        "a:QNum=QNum()\nb:QNum=QNum()\nc:QNum=QNum()\nd:QNum=QNum()\n"
        "a |= quantize(-0.5,fractional_bits=8,literal_bits=20)\nb |= 0.5\n"
        "negate(a,c)\nnegate(b,d)\nresult |= c-d\ndrop(a)\ndrop(b)\ndrop(c)\ndrop(d)",
        helper,
    )
    program = synth.synthesize(entry)
    assert dict(program.numeric_probabilities(fq.run(program))) == {(Fraction(1),): 1.0}


@pytest.mark.parametrize(
    "body",
    [
        "a:QNum[1]=QNum()\nallocate(a)\nhadamard_transform(a)\nwith adjoint():\n    f(a)\nresult |= quantize(0,fractional_bits=8,literal_bits=20)\ndrop(a)",
        "a:QNum[1]=QNum()\nallocate(a)\nhadamard_transform(a)\ncontrol(a,lambda:f(a))\nresult |= quantize(0,fractional_bits=8,literal_bits=20)\ndrop(a)",
        "a:QNum[8]=QNum()\na |= quantize(0.1,fractional_bits=8,literal_bits=20)\nresult |= a\ndrop(a)",
        "a:QNum=QNum()\na |= quantize(0.125,fractional_bits=8,literal_bits=20)\nresult |= a**3\ndrop(a)",
    ],
)
def test_control_inverse_explicit_format_and_exact_precision_refusals(
    body: str,
) -> None:
    helper = "@qfunc\ndef f(a:Const[QNum]):\n    phase(a,0.3)\n"
    with pytest.raises(HybridVerificationError, match="construction.python.*quantized"):
        synth.synthesize(definition("result:Output[QNum]", body, helper))


def test_bounded_ast_expansion_and_word_limits_refuse_before_lowering() -> None:
    bodies = [
        "result |= quantize("
        + "+".join("1" for _ in range(150))
        + ",fractional_bits=8,literal_bits=20)",
        "result |= quantize(1048576,fractional_bits=8,literal_bits=20)",
        "\n".join(f"a{i}:QNum=QNum()\na{i} |= 0\ndrop(a{i})" for i in range(86))
        + "\nresult |= quantize(0,fractional_bits=8,literal_bits=20)",
    ]
    for body in bodies:
        with pytest.raises(HybridVerificationError, match="budget|twenty bits"):
            synth.synthesize(definition("result:Output[QNum]", body))


def test_marker_alias_resolves_by_identity_and_host_body_is_never_executed() -> None:
    entry = definition(
        "result:Output[QNum]",
        "result |= q(0.1,fractional_bits=8,literal_bits=20)",
        "from flagquantum.synth import quantize as q\n",
    )
    assert entry.function is not None

    def bomb() -> None:
        raise AssertionError("host body was executed")

    entry.function.__code__ = bomb.__code__
    program = synth.synthesize(entry)
    assert dict(program.numeric_probabilities(fq.run(program))) == {
        (Fraction(25, 256),): 1.0
    }
