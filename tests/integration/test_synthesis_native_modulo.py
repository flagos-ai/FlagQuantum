"""Power-of-two remainder shares native bitwise lowering and exact ownership."""

from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_expression_values import (
    definition,
    reference,
)
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("divisor", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("a", range(4))
@pytest.mark.parametrize("b", range(4))
def test_every_small_pair_and_remainder_width(
    tmp_path: Path, divisor: int, a: int, b: int
) -> None:
    entry = definition(tmp_path, f"a |= {a}\nb |= {b}\nres |= (a+b)%{divisor}", (2, 2))
    program = synth.synthesize(entry)
    width = max(1, min(3, (divisor - 1).bit_length()))
    assert len(program.registers["res"]) == width
    compare_states(
        program,
        reference((2, 2, width), lambda x, y: (x + y) % divisor, (a, b)),
        ("a", "b", "res"),
    )


@pytest.mark.parametrize("divisor", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_coherent_modulo_and_nested_arithmetic_under_both_schedules(
    tmp_path: Path, divisor: int, objective: Literal["width", "depth"]
) -> None:
    entry = definition(
        tmp_path,
        "allocate(2,a)\nallocate(2,b)\nhadamard_transform(a)\nhadamard_transform(b)\nphase(a,0.37)\n"
        + f"res |= (a+b)%{divisor}",
        (2, 2),
    )
    program = synth.synthesize(entry, objective=objective)
    width = max(1, min(3, (divisor - 1).bit_length()))
    compare_states(
        program,
        reference((2, 2, width), lambda x, y: (x + y) % divisor, phase=0.37),
        ("a", "b", "res"),
    )


@pytest.mark.parametrize(
    "expression,diagnostic",
    [
        ("a%0", "power of two"),
        ("a%3", "power of two"),
        ("a%b", "power of two"),
        ("a%(-2)", r"program\.numeric\.modulo"),
        ("a%2.0", r"program\.numeric\.precision"),
        ("a%True", "power of two"),
        ("a%(2**2)", "power of two"),
        (f"a%{1<<257}", "power of two"),
    ],
)
def test_unsupported_divisors_fail_closed(
    tmp_path: Path, expression: str, diagnostic: str
) -> None:
    entry = definition(tmp_path, "a |= 2\nb |= 1\nres |= " + expression, (2, 2))
    with pytest.raises(HybridVerificationError, match=diagnostic):
        synth.synthesize(entry)


def remainder(a: int, b: int) -> int:
    return (a + b) % 4


def test_pure_expression_frontend_reuses_the_same_modulo_translation() -> None:
    program = synth.synthesize(
        remainder,
        inputs={"a": synth.UInt(2), "b": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
    )
    compare_states(
        program, reference((2, 2, 2), lambda x, y: (x + y) % 4), ("a", "b", "result")
    )


def test_modulo_preserves_unsigned_ownership_and_budget_refusals(
    tmp_path: Path,
) -> None:
    for body in (
        "a |= 2\nb |= 1\nres |= a%4\nres |= b%4",
        "a |= 2\nb |= 1\na |= a%4\nres |= b%4",
        "b |= 1\nres |= a%4",
    ):
        with pytest.raises(HybridVerificationError):
            synth.synthesize(definition(tmp_path, body, (2, 2)))
    entry = definition(
        tmp_path,
        "allocate(2,a)\nallocate(2,b)\nhadamard_transform(a)\nhadamard_transform(b)\nres |= (a+b)%4",
        (2, 2),
    )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(entry, max_qubits=1)


def test_modulo_one_and_constant_folding_keep_a_one_bit_zero_encoding() -> None:
    entry = program_definition(
        "@qfunc\ndef main(a: Output[QNum[2]], res: Output[QNum]):\n    a |= 3\n    res |= a%1\n"
    )
    program = synth.synthesize(entry)
    assert len(program.registers["res"]) == 1


def test_maximum_supported_literal_divisor_reuses_input_width(tmp_path: Path) -> None:
    entry = definition(tmp_path, f"a |= 3\nb |= 1\nres |= a%{1 << 256}", (2, 2))
    program = synth.synthesize(entry)
    assert len(program.registers["res"]) == 2
    compare_states(
        program, reference((2, 2, 2), lambda x, y: x, (3, 1)), ("a", "b", "res")
    )


def folded(a: int) -> int:
    return a + 11 % 4


def test_classical_only_remainder_folds_with_the_same_rules() -> None:
    program = synth.synthesize(
        folded, inputs={"a": synth.UInt(2)}, input_state=synth.InputState.uniform()
    )
    assert len(program.registers["result"]) == 3
    import flagquantum as fq

    result = fq.run(program)
    assert dict(
        program.probabilities(result, registers=("a", "result"))
    ) == pytest.approx({(a, a + 3): 0.25 for a in range(4)})


@pytest.mark.parametrize("annotation", ["Annotated[QNum, 5]", "QNum[5]"])
def test_standard_typing_width_metadata_matches_existing_numeric_domain(
    annotation: str,
) -> None:
    entry = program_definition(
        "from typing import Annotated\n"
        f"@qfunc\ndef main(a: Output[{annotation}], res: Output[QNum]):\n"
        "    a |= 4\n    res |= a%4\n"
    )
    program = synth.synthesize(entry)
    assert len(program.registers["a"]) == 5
    assert len(program.registers["res"]) == 2


@pytest.mark.parametrize(
    "annotation",
    [
        "Annotated[QNum, True]",
        "Annotated[QNum, 0]",
        "Annotated[QNum, 257]",
        "Annotated[QNum, 5, 0]",
        "Annotated[QBit, 5]",
        "Annotated[QArray[QBit], 5]",
        "Annotated[QNum, 2+3]",
    ],
)
def test_width_metadata_rejects_unadmitted_forms(annotation: str) -> None:
    with pytest.raises(HybridVerificationError, match="annotation"):
        entry = program_definition(
            "from typing import Annotated\nfrom flagquantum.synth import QArray\n"
            f"@qfunc\ndef main(a: Output[{annotation}]):\n    a |= 4\n"
        )
        synth.synthesize(entry)


def test_arithmetic_expression_notebook_runs_native_synthesis_and_sampling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.synthesis.functions.arithmetic.arithmetic_expression.arithmetic_expression_example.main import (
        run_example,
    )

    run_example()
    assert capsys.readouterr().out.strip() == "{(2, 1, 5, 0): 1000}"


def test_modulo_notebook_runs_five_bit_native_synthesis_and_sampling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.synthesis.functions.arithmetic.modulo.modulo_example.main import (
        run_example,
    )

    run_example()
    assert capsys.readouterr().out.strip() == "{(4, 7, 3): 1000}"
