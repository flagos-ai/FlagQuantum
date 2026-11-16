"""Native expression values share word kernels without importing foreign syntax."""

import importlib.util
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction

pytestmark = pytest.mark.integration
RELATIONS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def definition(
    tmp_path: Path, body: str, widths: tuple[int, int] | None = None
) -> _QuantumFunction:
    path = tmp_path / "expression_case.py"
    a_type = f"QNum[{widths[0]}]" if widths else "QNum"
    b_type = f"QNum[{widths[1]}]" if widths else "QNum"
    path.write_text(
        "from flagquantum.synth import Output, QNum, allocate, hadamard_transform, phase, qfunc\n"
        f"@qfunc\ndef main(a: Output[{a_type}], b: Output[{b_type}], res: Output[QNum]) -> None:\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("expression_case", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main
    assert isinstance(result, _QuantumFunction)
    return result


def reference(
    widths: tuple[int, int, int],
    calculate: Callable[[int, int], int],
    basis: tuple[int, int] | None = None,
    phase: float = 0.0,
) -> fq.Circuit:
    """Construct an independent full-domain XOR permutation, with complex inputs."""
    wa, wb, wr = widths
    size = 1 << sum(widths)
    circuit = fq.Circuit(sum(widths), dtype=torch.complex128)
    if basis is None:
        for wire in range(wa + wb):
            circuit.h(wire)
        for wire in range(wa):
            circuit.phase(wire, theta=phase * (1 << (wa - 1 - wire)))
    else:
        value = (basis[0] << wb) | basis[1]
        for wire in range(wa + wb):
            if value & (1 << (wa + wb - 1 - wire)):
                circuit.x(wire)
    matrix = torch.zeros((size, size), dtype=torch.complex128)
    for column in range(size):
        ab, result = divmod(column, 1 << wr)
        a, b = divmod(ab, 1 << wb)
        row = (ab << wr) | (result ^ calculate(a, b))
        matrix[row, column] = 1
    return circuit.any(*range(sum(widths)), matrix=matrix)


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("a", range(4))
@pytest.mark.parametrize("b", range(4))
def test_every_two_bit_comparison_input(
    tmp_path: Path, relation: str, a: int, b: int
) -> None:
    root = definition(tmp_path, f"a |= {a}\nb |= {b}\nres |= a {relation} b", (2, 2))
    program = synth.synthesize(root)
    compare_states(
        program, reference((2, 2, 1), RELATIONS[relation], (a, b)), ("a", "b", "res")
    )


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("widths", [(2, 2), (1, 3)])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_comparison_superposition_absolute_phase_and_zero_extension(
    tmp_path: Path,
    relation: str,
    widths: tuple[int, int],
    objective: Literal["width", "depth"],
) -> None:
    wa, wb = widths
    root = definition(
        tmp_path,
        f"allocate({wa}, a)\nallocate({wb}, b)\nhadamard_transform(a)\nhadamard_transform(b)\nphase(a, 0.37)\nres |= a {relation} b",
    )
    compare_states(
        synth.synthesize(root, objective=objective),
        reference((wa, wb, 1), RELATIONS[relation], phase=0.37),
        ("a", "b", "res"),
    )


@pytest.mark.parametrize(
    ("expression", "calculate", "width"),
    [
        ("a & b", operator.and_, 1),
        ("a | b", operator.or_, 3),
        ("a ^ b", operator.xor, 3),
        ("3 & b", lambda a, b: 3 & b, 2),
        ("b & 3", lambda a, b: b & 3, 2),
        ("3 | b", lambda a, b: 3 | b, 3),
        ("b | 3", lambda a, b: b | 3, 3),
        ("3 ^ b", lambda a, b: 3 ^ b, 3),
        ("b ^ 3", lambda a, b: b ^ 3, 3),
        ("~b", lambda a, b: 7 ^ b, 3),
        ("~a", lambda a, b: 1 ^ a, 1),
        ("(a + b) ^ b", lambda a, b: (a + b) ^ b, 4),
        ("~(a & b)", lambda a, b: 1 ^ (a & b), 1),
        ("a | (3 ^ 2)", lambda a, b: a | 1, 1),
        ("a < b <= 5", lambda a, b: int(a < b <= 5), 1),
    ],
)
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_coherent_word_values(
    tmp_path: Path,
    expression: str,
    calculate: Callable[[int, int], int],
    width: int,
    objective: Literal["width", "depth"],
) -> None:
    root = definition(
        tmp_path,
        f"allocate(1, a)\nallocate(3, b)\nhadamard_transform(a)\nhadamard_transform(b)\nphase(a, 0.37)\nres |= {expression}",
    )
    program = synth.synthesize(root, objective=objective)
    assert len(program.registers["res"]) == width
    compare_states(
        program, reference((1, 3, width), calculate, phase=0.37), ("a", "b", "res")
    )


@pytest.mark.parametrize(
    "body",
    [
        "a |= 1\nb |= 2\nres |= a & -1",
        "a |= 1\nb |= 2\nres |= a << 1",
        "a |= 1\nb |= 2\nres |= a in b",
        "a |= 1\nb |= 2\nres |= ~(2 + 1) | a",
        "a |= 1\nb |= 2\na |= a ^ b\nres |= a",
        "a |= 1\nb |= 2\nallocate(1, res)\nres |= a < b",
        "a |= 1\nres |= a < b\nb |= 2",
    ],
)
def test_word_values_preserve_domain_and_ownership_refusals(
    tmp_path: Path, body: str
) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(definition(tmp_path, body))


@pytest.mark.parametrize(
    ("qubits", "workspace", "memory"),
    [(3, None, None), (None, 0, None), (None, None, 1)],
)
def test_comparison_respects_resource_ceilings(
    tmp_path: Path, qubits: int | None, workspace: int | None, memory: int | None
) -> None:
    root = definition(
        tmp_path,
        "allocate(2, a)\nallocate(2, b)\nhadamard_transform(a)\nhadamard_transform(b)\nres |= a < b",
    )
    with pytest.raises((ValueError, MemoryError)):
        synth.synthesize(
            root, max_qubits=qubits, max_workspace=workspace, memory_limit_bytes=memory
        )


def inverted(x: int) -> int:
    return ~x


def classical_inversion(x: int) -> int:
    return x | ~2


def test_pure_entry_reuses_finite_word_inversion() -> None:
    program = synth.synthesize(
        inverted, inputs={"x": synth.UInt(3)}, input_state=synth.InputState.uniform()
    )
    state = fq.run(program)
    assert dict(
        program.probabilities(state, registers=("x", "result"))
    ) == pytest.approx({(x, 7 ^ x): 1 / 8 for x in range(8)})
    with pytest.raises(ValueError, match="quantum width"):
        synth.synthesize(classical_inversion, inputs={"x": synth.UInt(3)})


def test_inversion_notebook_completes_native_synthesis_and_sampling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.synthesis.functions.arithmetic.bitwise_invert.bitwise_invert_example.main import (
        run_example,
    )

    run_example()
    assert capsys.readouterr().out.strip() == "{(6, 1): 1000}"


def test_and_notebook_completes_both_native_programs_and_sampling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.synthesis.functions.arithmetic.bitwise_and.overview.main import (
        run_example,
    )

    run_example()
    assert capsys.readouterr().out.splitlines() == [
        "{(5, 1): 1000}",
        "{(4, 5, 4): 1000}",
    ]
