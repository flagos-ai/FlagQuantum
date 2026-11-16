"""Static Pauli views retain local bit order, spectators and absolute phase."""

import importlib
import math
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.pauli_evolution import Term, reference_sequence
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python_program import program_definition
from tools.synthesis_catalog import load_catalog

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
MODULE = "examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d"
OPERATOR = "-0.3*fq.I()+0.7*fq.Y(0)+0.2*fq.Z(1)+0.4*(fq.X(0)@fq.X(1))"
TERMS: tuple[Term, ...] = (
    (-0.3, ()),
    (0.7, ((0, "Y"),)),
    (0.2, ((1, "Z"),)),
    (0.4, ((0, "X"), (1, "X"))),
)


def definition(
    body: str, helper: str = "", ports: str = "time:float,q:Output[QArray[QBit]]"
) -> _QuantumFunction:
    return program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        + "OPERATOR="
        + OPERATOR
        + "\n"
        + helper
        + "@qfunc\ndef main("
        + ports
        + "):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


def reference(
    time: float,
    view: tuple[int, ...],
    order: int,
    repetitions: int,
    samples: int | None = None,
    seed: int = 0,
    width: int = 4,
) -> fq.Circuit:
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    circuit.rz(width - 1, theta=0.31).rz(0, theta=-0.28).cx(width - 1, width - 3)
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    matrices = (
        torch.eye(4, dtype=torch.complex128),
        torch.kron(identity, y),
        torch.kron(z, identity),
        torch.kron(x, x),
    )
    if samples is None:
        indices = (*range(4), *reversed(range(4))) if order == 2 else tuple(range(4))
        sequence = tuple(
            (index, time * TERMS[index][0] / repetitions / order)
            for _ in range(repetitions)
            for index in indices
        )
    else:
        sequence = reference_sequence(TERMS, time, samples, seed)
    matrix = torch.eye(4, dtype=torch.complex128)
    for index, angle in sequence:
        matrix = (
            math.cos(angle) * torch.eye(4, dtype=torch.complex128)
            - 1j * math.sin(angle) * matrices[index]
        ) @ matrix
    circuit.any(*(width - 1 - bit for bit in reversed(view)), matrix=matrix)
    return circuit


@pytest.mark.parametrize("num_iter", [1, 2])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_original_fermi_native_and_independent_circuit(
    num_iter: int, objective: Literal["width", "depth"]
) -> None:
    main = importlib.import_module(MODULE + ".main")
    independent = importlib.import_module(MODULE + ".comparisons.circuit")
    case = next(
        c for c in load_catalog()["cases"] if c["id"].endswith("/fermi_hubbard_1D.qmod")
    )
    programs = (
        synth.synthesize(
            main.main,
            classical_inputs={"num_iter": num_iter},
            objective=objective,
            max_workspace=0,
        ),
        synth.synthesize(
            ROOT / case["original"]["path"],
            classical_inputs={"num_iter": num_iter},
            objective=objective,
            max_workspace=0,
        ),
    )
    for program in programs:
        compare_states(program, independent.reference_circuit(num_iter), ("qba",))
        assert program.report["workspace_qubits"] == 0
        resources = program.report["resources"]
        assert isinstance(resources, ResourceEstimate)
        assert resources.n_qubits == 8


@pytest.mark.parametrize("time", [0.37, -0.29])
@pytest.mark.parametrize("order,repetitions", [(1, 1), (1, 3), (2, 3)])
@pytest.mark.parametrize(
    "target,view",
    [("q[1:3]", (1, 2)), ("[q[3],q[1]]", (3, 1)), ("(q[2],q[0])", (2, 0))],
)
def test_asymmetric_views_coherent_spectators_and_identity_phase(
    time: float, order: int, repetitions: int, target: str, view: tuple[int, ...]
) -> None:
    body = "allocate(4,q)\nhadamard_transform(q)\nRZ(0.31,q[0])\nRZ(-0.28,q[3])\nCX(q[0],q[2])\n"
    entry = definition(
        body + f"suzuki_trotter(OPERATOR,time,{order},{repetitions},{target})"
    )
    program = synth.synthesize(entry, classical_inputs={"time": time}, max_workspace=0)
    compare_states(program, reference(time, view, order, repetitions), ("q",))


@pytest.mark.parametrize("seed", [7, 31])
@pytest.mark.parametrize("time", [0.37, -0.29])
def test_sampled_view_uses_fixed_seed_and_independent_sampling(
    seed: int, time: float
) -> None:
    entry = definition(
        "allocate(4,q)\nhadamard_transform(q)\nRZ(0.31,q[0])\nRZ(-0.28,q[3])\nCX(q[0],q[2])\n"
        + f"qdrift(OPERATOR,time,19,[q[3],q[1]],{seed})"
    )
    program = synth.synthesize(entry, classical_inputs={"time": time}, max_workspace=0)
    compare_states(program, reference(time, (3, 1), 1, 1, 19, seed), ("q",))


@pytest.mark.parametrize("sampled", [False, True])
def test_zero_time_still_requires_initialized_owner(sampled: bool) -> None:
    call = (
        "qdrift(OPERATOR,0.0,19,q[1:3],7)"
        if sampled
        else "suzuki_trotter(OPERATOR,0.0,1,1,q[1:3])"
    )
    entry = definition(call, ports="q:Output[QArray[QBit]]")
    with pytest.raises(HybridVerificationError):
        synth.synthesize(entry)
    for preparation in ("allocate(4,q)", "allocate(4,q)\nhadamard_transform(q)"):
        entry = definition(preparation + "\n" + call, ports="q:Output[QArray[QBit]]")
        program = synth.synthesize(entry, max_workspace=0)
        circuit = fq.Circuit(4, dtype=torch.complex128)
        if "hadamard" in preparation:
            for wire in range(4):
                circuit.h(wire)
        compare_states(program, circuit, ("q",))


@pytest.mark.parametrize(
    "target", ["[]", "[q[0],q[0]]", "[q[4],q[1]]", "q[1:1]", "[q[True],q[1]]"]
)
def test_empty_repeated_and_invalid_views_refuse(target: str) -> None:
    entry = definition(
        "allocate(4,q)\n" + f"suzuki_trotter(OPERATOR,0.3,1,1,{target})",
        ports="q:Output[QArray[QBit]]",
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(entry)


def test_cross_owner_and_readonly_views_refuse() -> None:
    entry = definition(
        "allocate(2,q)\nallocate(2,p)\nsuzuki_trotter(OPERATOR,0.3,1,1,[q[0],p[1]])",
        ports="q:Output[QArray[QBit]],p:Output[QArray[QBit]]",
    )
    with pytest.raises(HybridVerificationError, match="one register"):
        synth.synthesize(entry)
    helper = "@qfunc\ndef locked(q:Const[QArray[QBit]]):\n    suzuki_trotter(OPERATOR,0.3,1,1,q[1:3])\n"
    entry = definition(
        "allocate(4,q)\nlocked(q)", helper, ports="q:Output[QArray[QBit]]"
    )
    with pytest.raises(HybridVerificationError, match="const"):
        synth.synthesize(entry)


def test_mutable_view_helper_preserves_local_order_and_inverse() -> None:
    helper = "@qfunc\ndef evolve(time:float,q:QArray[QBit]):\n    suzuki_trotter(OPERATOR,time,2,3,q)\n"
    preparation = "allocate(4,q)\nhadamard_transform(q)\nRZ(0.31,q[0])\nRZ(-0.28,q[3])\nCX(q[0],q[2])\n"
    entry = definition(preparation + "evolve(time,q[1:3])", helper)
    program = synth.synthesize(entry, classical_inputs={"time": 0.37}, max_workspace=0)
    compare_states(program, reference(0.37, (1, 2), 2, 3), ("q",))
    entry = definition(
        preparation + "evolve(time,q[1:3])\nwith adjoint():\n    evolve(time,q[1:3])",
        helper,
    )
    program = synth.synthesize(entry, classical_inputs={"time": 0.37}, max_workspace=0)
    compare_states(program, reference(0.0, (1, 2), 2, 3), ("q",))


@pytest.mark.parametrize(
    "limit,value",
    [
        ("max_qubits", 7),
        ("max_operations", 1),
        ("max_depth", 1),
        ("memory_limit_bytes", 1),
    ],
)
def test_fermi_actual_resource_limits_refuse(limit: str, value: int) -> None:
    main = importlib.import_module(MODULE + ".main")
    with pytest.raises(ValueError):
        synth.synthesize(
            main.main,
            classical_inputs={"num_iter": 2},
            max_qubits=value if limit == "max_qubits" else 256,
            max_operations=value if limit == "max_operations" else 100000,
            max_depth=value if limit == "max_depth" else None,
            memory_limit_bytes=value if limit == "memory_limit_bytes" else None,
        )


def test_small_owner_compaction_embeds_alternating_views() -> None:
    entry = definition(
        "allocate(3,q)\nhadamard_transform(q)\nRZ(0.31,q[0])\nCX(q[0],q[2])\n"
        "suzuki_trotter(OPERATOR,time,2,256,[q[2],q[0]])\n"
        "suzuki_trotter(OPERATOR,time,2,256,q[1:3])\n"
        "suzuki_trotter(OPERATOR,time,2,256,[q[2],q[0]])\n"
        "suzuki_trotter(OPERATOR,time,2,256,q[1:3])"
    )
    program = synth.synthesize(entry, classical_inputs={"time": 0.37}, max_workspace=0)
    circuit = fq.Circuit(3, dtype=torch.complex128)
    for bit in range(3):
        circuit.h(bit)
    circuit.rz(2, theta=0.31).cx(2, 0)
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    matrices = (
        torch.eye(4, dtype=torch.complex128),
        torch.kron(identity, y),
        torch.kron(z, identity),
        torch.kron(x, x),
    )
    step = torch.eye(4, dtype=torch.complex128)
    for index in (*range(4), *reversed(range(4))):
        angle = 0.37 * TERMS[index][0] / 256 / 2
        step = (
            math.cos(angle) * torch.eye(4, dtype=torch.complex128)
            - 1j * math.sin(angle) * matrices[index]
        ) @ step
    matrix = torch.linalg.matrix_power(step, 256)
    for view in ((2, 0), (1, 2), (2, 0), (1, 2)):
        circuit.any(*(2 - bit for bit in reversed(view)), matrix=matrix)
    compare_states(program, circuit, ("q",))
    assert program.report["workspace_qubits"] == 0
