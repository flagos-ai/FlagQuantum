"""Native modular powers reuse checked permutations with absolute phase."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.modular_exp.modular_exp_example.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.functions.arithmetic.modular_exp.modular_exp_example.main import (
    build_program,
    main,
    run_example,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/fixtures/classiq_qmod/corpus/original/functions/function_usage_examples/arithmetic/modular_exp/modular_exp_example.qmod"
)


def permutation(
    modulus: int, base: int, target_width: int, exponent_width: int
) -> torch.Tensor:
    """Use classical modular arithmetic, independent of emitted gate synthesis."""
    target_size = 1 << target_width
    size = target_size << exponent_width
    matrix = torch.zeros((size, size), dtype=torch.complex128)
    for column in range(size):
        power, x = divmod(column, target_size)
        target = pow(base, power, modulus) * x % modulus if x < modulus else x
        matrix[power * target_size + target, column] = 1
    return matrix


def definition(
    body: str, *, prefix: str = "", width: int = 3, powers: int = 2
) -> synth.Program:
    entry = program_definition(
        "from flagquantum.synth import *\n"
        + prefix
        + "\n"
        + f"@qfunc\ndef main(power: Output[QNum[{powers}]],x: Output[QNum[{width}]]):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    return synth.synthesize(entry)


@pytest.mark.parametrize("basis", range(32))
def test_every_small_basis_column_including_residues_above_the_modulus(
    basis: int,
) -> None:
    exponent, target = divmod(basis, 8)
    program = definition(
        f"power |= {exponent}\nx |= {target}\nmodular_exp(5,2,x,power)"
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        if basis & (1 << (4 - wire)):
            reference.x(wire)
    reference.any(*range(5), matrix=permutation(5, 2, 3, 2))
    compare_states(program, reference, ("power", "x"))


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize(
    ("modulus", "base", "width"),
    [(3, 2, 2), (5, 2, 3), (5, 4, 3), (7, 3, 3), (32, 3, 5)],
)
def test_complex_coherent_inputs_preserve_bit_order_absolute_phase_and_workspace(
    modulus: int, base: int, width: int, objective: Literal["width", "depth"]
) -> None:
    entry = program_definition(
        "from flagquantum.synth import *\n"
        + f"@qfunc\ndef main(power: Output[QNum[2]],x: Output[QNum[{width}]]):\n    allocate(power)\n    allocate(x)\n    hadamard_transform(power)\n    hadamard_transform(x)\n    phase(1,0.37)\n    phase(power,-0.29)\n    phase(x,0.51)\n"
        + f"    modular_exp({modulus},{base},x,power)\n"
    )
    reference = fq.Circuit(width + 2, dtype=torch.complex128)
    for wire in range(width + 2):
        reference.h(wire)
    reference.any(
        0,
        matrix=torch.eye(2, dtype=torch.complex128)
        * torch.exp(torch.tensor(0.37j, dtype=torch.complex128)),
    )
    reference.any(
        0,
        1,
        matrix=torch.diag(torch.exp(-0.29j * torch.arange(4, dtype=torch.float64))),
    )
    reference.any(
        *range(2, width + 2),
        matrix=torch.diag(
            torch.exp(0.51j * torch.arange(1 << width, dtype=torch.float64))
        ),
    )
    reference.any(*range(width + 2), matrix=permutation(modulus, base, width, 2))
    compare_states(
        synth.synthesize(entry, objective=objective), reference, ("power", "x")
    )


def test_original_native_and_notebook_sampling_workflow() -> None:
    for program in (build_program(), synth.synthesize(SOURCE)):
        compare_states(program, reference_circuit(), ("power", "x"))
    run_example()


@pytest.mark.parametrize(
    "call",
    [
        "modular_exp()",
        "modular_exp(5,2,x)",
        "modular_exp(5,2,x,power,x)",
        "modular_exp(n=5,a=2,x=x,power=power)",
        "modular_exp(*[5,2,x,power])",
    ],
)
def test_wrong_call_shapes_are_refused(call: str) -> None:
    with pytest.raises(HybridVerificationError, match="program.signature"):
        definition("allocate(power)\nallocate(x)\n" + call)


@pytest.mark.parametrize(
    "call",
    [
        "modular_exp(1,2,x,power)",
        "modular_exp(9,2,x,power)",
        "modular_exp(5,0,x,power)",
        "modular_exp(6,2,x,power)",
        "modular_exp(True,2,x,power)",
        "modular_exp(5,2.0,x,power)",
        "modular_exp(5,power,x,power)",
        "modular_exp(5,2,x,x)",
        "modular_exp(5,2,x,power[0])",
    ],
)
def test_invalid_domains_aliases_and_quantum_constants_are_refused(call: str) -> None:
    with pytest.raises(HybridVerificationError):
        definition("allocate(power)\nallocate(x)\n" + call)


def test_host_integer_hooks_and_calls_are_not_executed() -> None:
    for value in ("make_base()", "BASE"):
        prefix = "def make_base():\n    raise AssertionError('host execution')\nclass Evil:\n    def __int__(self):\n        raise AssertionError('conversion hook')\nBASE=Evil()"
        with pytest.raises(HybridVerificationError):
            definition(
                f"allocate(power)\nallocate(x)\nmodular_exp(5,{value},x,power)",
                prefix=prefix,
            )


@pytest.mark.parametrize(("width", "powers"), [(6, 2), (3, 11)])
def test_shared_width_boundaries_are_preserved(width: int, powers: int) -> None:
    with pytest.raises(HybridVerificationError, match="register.modular_exp"):
        definition(
            "allocate(power)\nallocate(x)\nmodular_exp(5,2,x,power)",
            width=width,
            powers=powers,
        )


def test_uninitialized_target_and_const_target_are_refused() -> None:
    with pytest.raises(HybridVerificationError):
        definition("allocate(power)\nmodular_exp(5,2,x,power)")
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef helper(x: Const[QNum[3]],power: Const[QNum[2]]):\n    modular_exp(5,2,x,power)\n@qfunc\ndef main(power: Output[QNum[2]],x: Output[QNum[3]]):\n    allocate(power)\n    allocate(x)\n    helper(x,power)\n"
    )
    with pytest.raises(HybridVerificationError, match="register.modular_exp"):
        synth.synthesize(entry)


@pytest.mark.parametrize(
    "build",
    [
        lambda: synth.synthesize(main, max_qubits=1),
        lambda: synth.synthesize(main, max_operations=1),
        lambda: synth.synthesize(main, max_workspace=0),
        lambda: synth.synthesize(main, memory_limit_bytes=16),
    ],
)
def test_modular_resources_use_existing_budget_checks(
    build: Callable[[], synth.Program],
) -> None:
    with pytest.raises((synth.SynthesisError, PlanningError)):
        build()


def test_mutable_numeric_helper_reuses_the_same_complete_register_operation() -> None:
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef helper(x: QNum[3],power: QNum[2]):\n    modular_exp(5,2,x,power)\n@qfunc\ndef main(power: Output[QNum[2]],x: Output[QNum[3]]):\n    allocate(power)\n    hadamard_transform(power)\n    x |= 1\n    helper(x,power)\n"
    )
    reference = (
        fq.Circuit(5, dtype=torch.complex128)
        .h(0)
        .h(1)
        .x(4)
        .any(*range(5), matrix=permutation(5, 2, 3, 2))
    )
    compare_states(synth.synthesize(entry), reference, ("power", "x"))


def test_ten_bit_exponent_boundary_keeps_base_one_identity_and_zero_workspace() -> None:
    program = definition(
        "allocate(power)\nhadamard_transform(power)\nx |= 17\nmodular_exp(31,1,x,power)",
        width=5,
        powers=10,
    )
    reference = fq.Circuit(15, dtype=torch.complex128).x(10).x(14)
    for wire in range(10):
        reference.h(wire)
    compare_states(program, reference, ("power", "x"))
    assert program.report["workspace_qubits"] == 0


def test_qbit_is_a_one_bit_numeric_target() -> None:
    entry = program_definition(
        "from flagquantum.synth import *\n"
        "@qfunc\ndef main(power: Output[QNum[2]],x: Output[QBit]):\n"
        "    allocate(power)\n    hadamard_transform(power)\n    x |= 1\n"
        "    modular_exp(2,3,x,power)\n"
    )
    compare_states(
        synth.synthesize(entry),
        fq.Circuit(3, dtype=torch.complex128).h(0).h(1).x(2),
        ("power", "x"),
    )


def test_full_notebook_provenance_covers_sampling_and_modular_assertion() -> None:
    import json

    inventory = json.loads(
        (ROOT / "tests/fixtures/classiq_qmod/native_library_inventory.json").read_text()
    )
    group = next(
        g
        for g in inventory["groups"]
        if g["id"]
        == "functions/function_usage_examples/arithmetic/modular_exp/modular_exp_example"
    )
    assert group["workflow_status"] == "validated"
    assert [
        c["index"] for c in group["workflow_validation"]["reviewed_code_cells"]
    ] == [2, 3, 5]
