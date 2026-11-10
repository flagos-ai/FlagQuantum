"""Reversible expression XOR reads arbitrary targets and preserves clean work."""

import cmath
from collections.abc import Callable
from dataclasses import replace
from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.register_program_cases import Registers
from tests.team.compiler.test_construction_register_cleanup import pending
from tools.synthesis_compiler_examples.register_cleanup.main import flag_value, less

pytestmark = pytest.mark.integration
CASES: tuple[tuple[str, int, Callable[[int, int], int]], ...] = (
    ("a+b", 3, lambda a, b: a + b),
    ("a*b", 4, lambda a, b: a * b),
    ("a*a+b", 4, lambda a, b: a * a + b),
    ("a^b", 2, lambda a, b: a ^ b),
    ("a<=b", 1, lambda a, b: int(a <= b)),
)


@pytest.mark.parametrize("expression,width,oracle", CASES)
def test_all_inputs_and_every_nonzero_target_restore_physical_workspace(
    expression: str,
    width: int,
    oracle: Callable[[int, int], int],
) -> None:
    program = synth.synthesize(
        definition(
            f"a:Output[QNum[2]],b:Output[QNum[2]],result:Output[QNum[{width}]]",
            f"allocate(a)\nallocate(b)\nallocate(result)\nresult ^= {expression}",
        )
    )
    wires = tuple(program.registers[name] for name in ("a", "b", "result"))
    for a, b, target in product(range(4), range(4), range(1 << width)):
        initial = encode((a, b, target), wires)
        expected = encode((a, b, target ^ oracle(a, b)), wires)
        assert classical_apply(program.circuit, initial) == expected


@pytest.mark.parametrize("expression,width,oracle", CASES)
def test_complete_phase_weighted_coherence_including_target(
    expression: str,
    width: int,
    oracle: Callable[[int, int], int],
) -> None:
    program = synth.synthesize(
        definition(
            f"a:Output[QNum[2]],b:Output[QNum[2]],result:Output[QNum[{width}]]",
            "allocate(a)\nallocate(b)\nallocate(result)\nhadamard_transform(a)\n"
            "hadamard_transform(b)\nhadamard_transform(result)\n"
            f"phase(0.13*a+0.17*b+0.19*result+0.23,1)\nresult ^= {expression}",
        )
    )
    size = 1 << (4 + width)
    matrix = torch.zeros((size, size), dtype=torch.complex128)
    for a, b, target in product(range(4), range(4), range(1 << width)):
        before = ((4 * a + b) << width) | target
        after = ((4 * a + b) << width) | (target ^ oracle(a, b))
        matrix[after, before] = torch.tensor(
            cmath.exp(1j * (0.13 * a + 0.17 * b + 0.19 * target + 0.23)),
            dtype=torch.complex128,
        )
    reference = fq.Circuit(4 + width, dtype=torch.complex128)
    for wire in range(4 + width):
        reference.h(wire)
    reference.any(*range(4 + width), matrix=matrix)
    compare_states(program, reference, ("a", "b", "result"))


@pytest.mark.parametrize("region", ["repeated", "adjoint", "within"])
def test_nonzero_target_zero_extension_inverse_and_within(region: str) -> None:
    body = "allocate(a)\nallocate(b)\nresult |= 5\nhadamard_transform(a)\nhadamard_transform(b)\n"
    if region == "repeated":
        body += "result ^= a<=b\nresult ^= a<=b"
    elif region == "adjoint":
        body += "result ^= a<=b\nwith adjoint():\n    result ^= a<=b"
    else:
        body += "with within():\n    result ^= a<=b\nwith apply():\n    phase(result-0.5,0.31)"
    program = synth.synthesize(
        definition("a:Output[QNum[2]],b:Output[QNum[2]],result:Output[QNum[3]]", body)
    )
    reference = fq.Circuit(7, dtype=torch.complex128).h(0).h(1).h(2).h(3).x(4).x(6)
    if region == "within":
        values = [
            cmath.exp(0.31j * ((target ^ int(a <= b)) - 0.5))
            for a, b, target in product(range(4), range(4), range(8))
        ]
        reference.any(
            *range(7), matrix=torch.diag(torch.tensor(values, dtype=torch.complex128))
        )
    compare_states(program, reference, ("a", "b", "result"))


@pytest.mark.parametrize(
    "ports,body,diagnostic",
    [
        (
            "a:Output[QNum[2]],result:Output[QNum[3]]",
            "allocate(a)\nresult ^= a+1",
            "uninitialized",
        ),
        (
            "a:Output[QNum[2]],result:Output[QBit]",
            "allocate(a)\nallocate(result)\nresult ^= a+1",
            "truncate",
        ),
        (
            "a:Output[QNum[2]],result:Output[QNum[2]]",
            "allocate(a)\nallocate(result)\nresult ^= result^a",
            "alias",
        ),
        (
            "a:Output[QNum[2,True,0]],result:Output[QNum[3]]",
            "allocate(a)\nallocate(result)\nresult ^= a",
            "numeric",
        ),
        (
            "a:Output[QNum[2]],result:Output[QNum[3,True,0]]",
            "allocate(a)\nallocate(result)\nresult ^= a",
            "unsigned",
        ),
        (
            "a:Output[QNum[2]],result:Output[QArray[QBit]]",
            "allocate(a)\nallocate(3,result)\nresult ^= a",
            "scalar",
        ),
    ],
)
def test_expression_xor_refusals_preserve_format_width_and_storage(
    ports: str,
    body: str,
    diagnostic: str,
) -> None:
    with pytest.raises(
        (synth.SynthesisError, HybridVerificationError), match=diagnostic
    ):
        synth.synthesize(definition(ports, body))


def test_borrowed_helper_expression_xor_is_not_admitted() -> None:
    helper = "@qfunc\ndef xor(a:QNum,result:QNum):\n    result ^= a\n"
    with pytest.raises(HybridVerificationError, match="mutable helpers"):
        synth.synthesize(
            definition(
                "a:Output[QNum[2]],result:Output[QNum[2]]",
                "allocate(a)\nallocate(result)\nxor(a,result)",
                helper,
            )
        )


def test_xor_write_invalidates_a_dependent_cleanup_proof() -> None:
    builder, x, _, flag, graph = pending()
    identity = capture(flag_value, {"flag": 1})
    builder.emit("qreg.xor", x, flag, callee=identity.name)
    builder.emit("qreg.uncompute", flag)
    with pytest.raises(HybridVerificationError, match="register.uncompute"):
        verify_register_program(builder.finish(), (graph, identity))


def test_xor_never_establishes_fresh_zero_cleanup_authority() -> None:
    builder = Registers()
    x, y, flag = (
        builder.declare("x", 2),
        builder.declare("y", 2),
        builder.declare("flag", 1),
    )
    for value in (x, y, flag):
        builder.emit("qreg.allocate", value)
    graph = capture(less, {"x": 2, "y": 2})
    builder.emit("qreg.xor", flag, x, y, callee=graph.name)
    builder.emit("qreg.uncompute", flag)
    with pytest.raises(HybridVerificationError, match="register.uncompute"):
        verify_register_program(builder.finish(), (graph,))


def test_verifier_refuses_formatted_words_without_trusting_capture() -> None:
    root, functions = capture_program(
        definition(
            "a:Output[QNum[2]],result:Output[QNum[2]]",
            "allocate(a)\nallocate(result)\nresult ^= a",
        )
    )
    block = root.body.blocks[0]
    operations = tuple(
        (
            replace(op, attributes={**op.attributes, "numeric_format": (True, 0)})
            if op.name == "qreg.declare" and op.attributes["name"] == "a"
            else op
        )
        for op in block.operations
    )
    forged = replace(
        root, body=replace(root.body, blocks=(replace(block, operations=operations),))
    )
    with pytest.raises(HybridVerificationError, match="unsigned"):
        verify_register_program(forged, functions)


@pytest.mark.parametrize("constraint", ["operations", "qubits", "workspace", "memory"])
def test_xor_budgets_use_actual_lowering(constraint: str) -> None:
    entry = definition(
        "a:Output[QNum[2]],b:Output[QNum[2]],result:Output[QNum[3]]",
        "allocate(a)\nallocate(b)\nallocate(result)\nresult ^= a+b",
    )
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(entry, max_operations=1)
        elif constraint == "qubits":
            synth.synthesize(entry, max_qubits=6)
        elif constraint == "workspace":
            synth.synthesize(entry, max_workspace=0)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)
