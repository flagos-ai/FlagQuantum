"""Checked bound widths preserve native state, ownership and resource contracts."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend.program import capture_program
from flagquantum.compiler._construction.verification.register_profile import (
    verify_register_program,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_bound_width_helper_preserves_phase_and_named_readout(bits: int) -> None:
    function = definition(
        "bits:int,offset:int,q:Output[QNum]",
        "allocate(bits+offset,q)\nprepare(bits+offset,q)",
        "@qfunc\ndef prepare(width:int,q:QNum):\n"
        "    hadamard_transform(q)\n"
        "    phase(0.17*width*q,1)\n",
    )
    for offset in (0, 1):
        width = bits + offset
        program = synth.synthesize(
            function, classical_inputs={"bits": bits, "offset": offset}
        )
        reference = fq.Circuit(width, dtype=torch.complex128)
        for wire in range(width):
            reference.h(wire)
            reference.phase(wire, theta=0.17 * width * 2 ** (width - 1 - wire))
        compare_states(program, reference, ("q",))
        assert dict(program.probabilities(fq.run(program), registers=("q",))) == (
            pytest.approx(dict.fromkeys(((i,) for i in range(1 << width)), 2**-width))
        )


def test_repeated_and_concurrent_specializations_do_not_retain_width() -> None:
    function = definition(
        "bits:int,q:Output[QNum]", "allocate(bits,q)\nhadamard_transform(q)"
    )

    def build(bits: int) -> synth.Program:
        return synth.synthesize(function, classical_inputs={"bits": bits})

    widths = (1, 4, 2, 3)
    sequential = tuple(build(bits) for bits in widths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = tuple(pool.map(build, widths))
    for programs in (sequential, concurrent):
        assert tuple(len(p.registers["q"]) for p in programs) == widths
        for bits, program in zip(widths, programs, strict=True):
            compare_states(
                program,
                _uniform_reference(bits),
                ("q",),
            )
    for left, right in zip(sequential, concurrent, strict=True):
        assert left.report["resources"] == right.report["resources"]


def _uniform_reference(bits: int) -> fq.Circuit:
    circuit = fq.Circuit(bits, dtype=torch.complex128)
    for wire in range(bits):
        circuit.h(wire)
    return circuit


def test_static_constants_array_reads_and_loop_indices_reuse_integer_evaluator() -> (
    None
):
    function = definition(
        "sizes:list[int],q:Output[QNum]",
        "for index in range(1):\n    allocate(+(sizes[index]*2)-WIDTH,q)\n"
        "hadamard_transform(q)",
        "WIDTH=1\n",
    )
    program = synth.synthesize(function, classical_inputs={"sizes": [2]})
    compare_states(program, _uniform_reference(3), ("q",))


def test_closure_integer_size_is_specialized_without_host_execution() -> None:
    width = 3

    @synth.qfunc
    def main(q: synth.Output[synth.QNum]) -> None:
        synth.allocate(width, q)
        synth.hadamard_transform(q)

    compare_states(synth.synthesize(main), _uniform_reference(3), ("q",))


@pytest.mark.parametrize("width", [1, 256])
def test_width_boundary_capture_does_not_allocate_dense_state(width: int) -> None:
    root, helpers = capture_program(
        definition("bits:int,q:Output[QNum]", "allocate(bits,q)"),
        classical_inputs={"bits": width},
    )
    verify_register_program(root, helpers)
    declarations = [
        op.results[0].type.parameters[0]
        for op in root.body.blocks[0].operations
        if op.name == "qreg.declare"
    ]
    assert declarations == [width]


@pytest.mark.parametrize("width", [0, -1, 257, 4097, True, 1.0])
def test_invalid_bound_sizes_are_refused(width: int | float) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition("bits:int,q:Output[QNum]", "allocate(bits,q)"),
            classical_inputs={"bits": width},
        )


@pytest.mark.parametrize(
    "expression",
    [
        "missing",
        "True",
        "1.0",
        "q",
        "q[0]",
        "int(2)",
        "len([1,2])",
        "obj.width",
        "2**2",
        "4096*2",
        "+" * 33 + "1",
    ],
)
def test_nonstatic_or_out_of_contract_sizes_are_refused(expression: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(definition("q:Output[QNum]", f"allocate({expression},q)"))


@pytest.mark.parametrize(
    "ports,body",
    [
        ("bits:int,q:Output[QNum[2]]", "allocate(bits,q)"),
        ("bits:int,q:Output[QNum]", "allocate(bits,q)\nallocate(bits,q)"),
        ("bits:int,q:Output[QNum]", "allocate(bits,q[0])"),
        ("bits:int,q:Const[QNum[3]]", "allocate(bits,q)"),
    ],
)
def test_declared_width_and_storage_ownership_are_not_bypassed(
    ports: str, body: str
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(definition(ports, body), classical_inputs={"bits": 3})


def test_width_and_memory_budgets_apply_after_binding() -> None:
    function = definition("bits:int,q:Output[QNum]", "allocate(bits,q)")
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(function, classical_inputs={"bits": 4}, max_qubits=3)
    with pytest.raises(PlanningError):
        synth.synthesize(function, classical_inputs={"bits": 4}, memory_limit_bytes=16)


def test_bound_width_coherent_control_has_absolute_interference_phase() -> None:
    function = definition(
        "bits:int,q:Output[QNum],flag:Output[QBit]",
        "allocate(bits,q)\nallocate(flag)\nhadamard_transform(q)\nH(flag)\n"
        "control(flag,lambda:phase(q,0.23))",
    )
    for bits in (1, 3):
        program = synth.synthesize(function, classical_inputs={"bits": bits})
        reference = _uniform_reference(bits + 1)
        for wire in range(bits):
            reference.cphase(bits, wire, theta=0.23 * 2 ** (bits - 1 - wire))
        compare_states(program, reference, ("q", "flag"))


def test_bound_width_does_not_admit_bound_numeric_annotation_or_output_helper() -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition("bits:int,q:Output[QNum[bits]]", "allocate(bits,q)"),
            classical_inputs={"bits": 3},
        )
    with pytest.raises(HybridVerificationError, match="plain writable"):
        synth.synthesize(
            definition(
                "bits:int,q:Output[QNum]",
                "prepare(bits,q)",
                "@qfunc\ndef prepare(bits:int,q:Output[QNum]):\n"
                "    allocate(bits,q)\n",
            ),
            classical_inputs={"bits": 3},
        )


def test_single_argument_and_literal_forms_remain_equivalent() -> None:
    programs = (
        synth.synthesize(definition("q:Output[QNum[3]]", "allocate(q)")),
        synth.synthesize(definition("q:Output[QNum]", "allocate(3,q)")),
        synth.synthesize(
            definition("bits:int,q:Output[QNum]", "allocate(bits,q)"),
            classical_inputs={"bits": 3},
        ),
    )
    for program in programs:
        compare_states(program, fq.Circuit(3, dtype=torch.complex128), ("q",))
    assert programs[0].report["resources"] == programs[1].report["resources"]
    assert programs[1].report["resources"] == programs[2].report["resources"]


def test_bound_width_operation_and_workspace_budgets_are_not_bypassed() -> None:
    function = definition(
        "bits:int,q:Output[QNum]", "allocate(bits,q)\nhadamard_transform(q)"
    )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(function, classical_inputs={"bits": 3}, max_operations=2)
    compute = definition(
        "bits:int,x:Output[QNum],y:Output[QNum],result:Output[QNum]",
        "allocate(bits,x)\nallocate(bits,y)\nhadamard_transform(x)\n"
        "hadamard_transform(y)\nresult |= x*x+y",
    )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            compute,
            classical_inputs={"bits": 3},
            max_workspace=0,
        )
