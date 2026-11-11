"""Repeated pure definitions retain every coherent application and specialization."""

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


def _repeated_entry() -> _QuantumFunction:
    return definition(
        "repetitions:int,x:Output[QNum[4]],target:Output[QBit]",
        "allocate(x)\nallocate(target)\nhadamard_transform(x)\n"
        "hadamard_transform(target)\nRZ(0.37,target)\n"
        "for iteration in range(repetitions):\n    mark(x,target)",
        "@qfunc\ndef mark(x:QNum,target:QBit):\n"
        "    control((x==0)|(x==1)|(x==2)|(x==3)|(x==4),lambda:X(target))\n",
    )


@pytest.mark.parametrize("repetitions", [254, 255])
def test_repeated_definitions_fit_source_budget_and_retain_all_applications(
    repetitions: int,
) -> None:
    program, functions = capture_program(
        _repeated_entry(), {"repetitions": repetitions}
    )
    assert len(functions) == 1
    assert (
        sum(
            operation.name == "qreg.control_x"
            for operation in program.body.blocks[0].operations
        )
        == repetitions
    )


@pytest.mark.parametrize("repetitions", [2, 3])
def test_reused_definition_preserves_complete_repeated_interference(
    repetitions: int,
) -> None:
    program = synth.synthesize(
        _repeated_entry(),
        classical_inputs={"repetitions": repetitions},
        max_qubits=24,
        memory_limit_bytes=512 * 1024**2,
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for bit in range(5):
        reference.h(bit)
    reference.rz(4, theta=0.37)
    permutation = torch.eye(32, dtype=torch.complex128)
    destinations = [word ^ int(word // 2 < 5) for word in range(32)]
    permutation = permutation[destinations]
    reference.any(*range(5), matrix=torch.linalg.matrix_power(permutation, repetitions))
    compare_states(program, reference, ("x", "target"))


def test_definition_reuse_does_not_override_operation_budget() -> None:
    with pytest.raises(synth.SynthesisError, match="operations|budget"):
        synth.synthesize(
            _repeated_entry(),
            classical_inputs={"repetitions": 254},
            max_operations=1000,
        )


def test_classical_predicate_bindings_remain_distinct() -> None:
    entry = definition(
        "x:Output[QNum[2]],target:Output[QBit]",
        "allocate(x)\nallocate(target)\nhadamard_transform(x)\n"
        "mark(1,x,target)\nmark(2,x,target)",
        "@qfunc\ndef mark(value:int,x:QNum,target:QBit):\n"
        "    control(x==value,lambda:X(target))\n",
    )
    program = synth.synthesize(entry)
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(1)
    destinations = [word ^ int(word // 2 in (1, 2)) for word in range(8)]
    reference.any(*range(3), matrix=torch.eye(8, dtype=torch.complex128)[destinations])
    compare_states(program, reference, ("x", "target"))


def test_same_predicate_on_different_widths_keeps_input_domains() -> None:
    entry = definition(
        "x:Output[QNum[2]],y:Output[QNum[3]],target:Output[QBit]",
        "allocate(x)\nallocate(y)\nallocate(target)\n"
        "hadamard_transform(x)\nhadamard_transform(y)\n"
        "mark(x,target)\nmark(y,target)",
        "@qfunc\ndef mark(value:QNum,target:QBit):\n"
        "    control(value==1,lambda:X(target))\n",
    )
    program = synth.synthesize(entry)
    reference = fq.Circuit(6, dtype=torch.complex128)
    for bit in range(5):
        reference.h(bit)
    destinations = [
        word ^ int(((word >> 4) == 1) != (((word >> 1) & 7) == 1)) for word in range(64)
    ]
    reference.any(*range(6), matrix=torch.eye(64, dtype=torch.complex128)[destinations])
    compare_states(program, reference, ("x", "y", "target"))
