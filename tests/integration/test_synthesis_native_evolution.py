"""Check native Pauli evolution against independent dense Circuit formulas."""

import math
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.hamiltonian_evolution.suzuki_trotter.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
OPERATOR = "1.5*(fq.X(0)@fq.X(1)@fq.Z(2))+0.5*(fq.Y(0)@fq.X(1)@fq.Z(2))"


@pytest.mark.parametrize("time", [0.37, -0.29])
@pytest.mark.parametrize("order,repetitions", [(1, 1), (1, 3), (2, 3)])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_native_evolution_has_declared_order_bit_order_and_absolute_phase(
    time: float, order: int, repetitions: int, objective: Literal["width", "depth"]
) -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        f"OPERATOR={OPERATOR}\n"
        "@qfunc\ndef main(time:float,q:Output[QArray[QBit]]):\n"
        "    allocate(3,q)\n"
        f"    suzuki_trotter(OPERATOR,time,{order},{repetitions},q)\n"
    )
    program = synth.synthesize(
        entry, classical_inputs={"time": time}, objective=objective
    )
    compare_states(program, reference_circuit(time, order, repetitions), ("q",))
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "operator,call,diagnostic",
    [
        ("fq.X(3)", "OPERATOR,0.3,1,1,q", "indices"),
        ("fq.X(0)", "OPERATOR,0.3,3,1,q", "order"),
        ("fq.X(0)", "OPERATOR,0.3,1,257,q", "repetitions"),
        ("fq.X(0)", "OPERATOR,0.3,1,True,q", "integer"),
        ("fq.X(0)", "OPERATOR,q,1,1,q", "classical"),
        ("fq.X(0)", "OPERATOR,0.3,1,1,q[1:]", "complete register"),
        ("fq.X(0)", "fq.X(0),0.3,1,1,q", "static Observable"),
        ("fq.X(0)", "OPERATOR,0.3,1,q,q", "integer"),
    ],
)
def test_invalid_evolution_fails_before_execution(
    operator: str, call: str, diagnostic: str
) -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        f"OPERATOR={operator}\n"
        "@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        "    allocate(3,q)\n"
        f"    suzuki_trotter({call})\n"
    )
    with pytest.raises(
        (HybridVerificationError, synth.SynthesisError), match=diagnostic
    ):
        synth.synthesize(entry)


def test_resource_budget_rejects_before_running() -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        f"OPERATOR={OPERATOR}\n"
        "@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        "    allocate(3,q)\n    suzuki_trotter(OPERATOR,0.3,1,1,q)\n"
    )
    with pytest.raises((PlanningError, synth.SynthesisError), match="qubit|budget"):
        synth.synthesize(entry, max_qubits=2)
    with pytest.raises((PlanningError, synth.SynthesisError)):
        synth.synthesize(entry, max_operations=1)
    with pytest.raises((PlanningError, synth.SynthesisError)):
        synth.synthesize(entry, max_depth=1)
    with pytest.raises((PlanningError, synth.SynthesisError)):
        synth.synthesize(entry, memory_limit_bytes=1)


@pytest.mark.parametrize("time", [0.37, -0.29])
def test_identity_phase_coherent_input_offset_target_and_mutable_helper(
    time: float,
) -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "OPERATOR=-0.5*fq.I()+0.7*fq.Y(0)+0.2*fq.Z(1)\n"
        "@qfunc\ndef evolve(t:float,q:QArray[QBit]):\n"
        "    suzuki_trotter(OPERATOR,t,2,3,q)\n"
        "@qfunc\ndef main(time:float,sentinel:Output[QBit],q:Output[QArray[QBit]]):\n"
        "    allocate(sentinel)\n    H(sentinel)\n    allocate(2,q)\n"
        "    hadamard_transform(q)\n    RZ(0.49,q[0])\n    evolve(time,q)\n"
    )
    program = synth.synthesize(entry, classical_inputs={"time": time})
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2).rz(2, theta=0.49)
    i = torch.eye(2, dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    identity = torch.eye(4, dtype=torch.complex128)
    terms = ((-0.5, identity), (0.7, torch.kron(i, y)), (0.2, torch.kron(z, i)))
    for _ in range(3):
        for weight, pauli in (*terms, *reversed(terms)):
            angle = time / 6 * weight
            reference.any(
                1, 2, matrix=math.cos(angle) * identity - 1j * math.sin(angle) * pauli
            )
    compare_states(program, reference, ("sentinel", "q"))
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "operator",
    [
        "fq.Observable([_PauliTerm(1.0,((0,'x'),))])",
        "fq.Observable((_PauliTerm(True,((0,'x'),)),))",
        "fq.Observable((_PauliTerm(float('nan'),((0,'x'),)),))",
        "fq.Observable((_PauliTerm(10**1000,((0,'x'),)),))",
        "fq.Observable((_PauliTerm(1.0,((0,'x'),(0,'z'))),))",
        "fq.Observable((_PauliTerm(1.0,((0,'a'),)),))",
        "fq.Observable((_PauliTerm(1.0,([0,'x'],)),))",
        "fq.Observable(tuple(_PauliTerm(1.0,((0,'x'),)) for _ in range(1025)))",
    ],
)
def test_malformed_observables_fail_closed(operator: str) -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "from flagquantum.observables import _PauliTerm\n"
        f"OPERATOR={operator}\n"
        "@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        "    allocate(2,q)\n    suzuki_trotter(OPERATOR,0.3,1,1,q)\n"
    )
    with pytest.raises(
        (HybridVerificationError, synth.SynthesisError), match="Pauli|Observable"
    ):
        synth.synthesize(entry)


@pytest.mark.parametrize("allocated", [False, True])
def test_uninitialized_and_const_storage_are_rejected(allocated: bool) -> None:
    prefix = (
        "@qfunc\ndef locked(q:Const[QArray[QBit]]):\n"
        "    suzuki_trotter(OPERATOR,0.3,1,1,q)\n"
        if allocated
        else ""
    )
    body = (
        "allocate(2,q)\n    locked(q)"
        if allocated
        else "suzuki_trotter(OPERATOR,0.3,1,1,q)"
    )
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "OPERATOR=fq.X(0)\n"
        + prefix
        + "@qfunc\ndef main(q:Output[QArray[QBit]]):\n    "
        + body
        + "\n"
    )
    with pytest.raises((HybridVerificationError, synth.SynthesisError)):
        synth.synthesize(entry)


def test_formula_size_is_bounded_before_expansion() -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "OPERATOR=fq.Observable(tuple(fq.X(0).terms[0] for _ in range(256)))\n"
        "@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        "    allocate(1,q)\n    suzuki_trotter(OPERATOR,0.3,2,256,q)\n"
    )
    with pytest.raises((HybridVerificationError, synth.SynthesisError), match="4096"):
        synth.synthesize(entry)


@pytest.mark.parametrize("term_count,repetitions", [(256, 1), (1, 256)])
def test_admitted_term_and_repetition_limits_on_numeric_storage(
    term_count: int, repetitions: int
) -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        f"OPERATOR=fq.Observable(tuple(fq.X(0).terms[0] for _ in range({term_count})))\n"
        "@qfunc\ndef main(q:Output[QNum[1]]):\n    allocate(q)\n"
        f"    suzuki_trotter(OPERATOR,0.37,1,{repetitions},q)\n"
    )
    reference = fq.Circuit(1, dtype=torch.complex128).rx(0, theta=2 * 0.37 * term_count)
    compare_states(synth.synthesize(entry), reference, ("q",))


def test_host_calls_are_not_executed_during_capture() -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "def operator():\n    raise AssertionError('host function was executed')\n"
        "@qfunc\ndef main(q:Output[QNum[1]]):\n    allocate(q)\n"
        "    suzuki_trotter(operator(),0.3,1,1,q)\n"
    )
    with pytest.raises(
        (HybridVerificationError, synth.SynthesisError), match="static Observable"
    ):
        synth.synthesize(entry)


def test_both_migrated_originals_match_native_and_independent_circuits() -> None:
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.hamiltonian_simulation_guide_trotter.comparisons.circuit import (
        reference_circuit as guide_reference,
    )
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.hamiltonian_simulation_guide_trotter.main import (
        build_program as guide_program,
    )
    from examples.synthesis.functions.hamiltonian_evolution.suzuki_trotter.main import (
        build_program,
    )

    root = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"
    for time in (0.37, -0.29):
        original = synth.synthesize(
            root
            / "functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/suzuki_trotter/suzuki_trotter.qmod",
            classical_inputs={"x": time},
        )
        for program in (original, build_program(time)):
            compare_states(program, reference_circuit(time), ("qba",))
    original = synth.synthesize(
        root
        / "algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide_trotter.qmod"
    )
    for program in (original, guide_program()):
        compare_states(program, guide_reference(), ("qba",))


def test_guide_parameter_specializations_match_independent_circuits() -> None:
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.hamiltonian_simulation_guide_trotter.comparisons.circuit import (
        reference_circuit as guide_reference,
    )
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.hamiltonian_simulation_guide_trotter.main import (
        build_program as guide_program,
    )

    for time in (0.37, -0.29):
        compare_states(guide_program(time), guide_reference(time), ("qba",))
