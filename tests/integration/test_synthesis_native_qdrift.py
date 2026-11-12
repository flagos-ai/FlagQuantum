"""Sampled native evolution preserves independent CDF, complex phase and budgets."""

import bisect
import math
import random

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
COEFFICIENTS = (-0.2, 0.7, -0.3)


def definition(call: str, *, allocated: bool = True) -> _QuantumFunction:
    preparation = (
        "    allocate(2,q)\n    hadamard_transform(q)\n    RZ(0.49,q[0])\n"
        if allocated
        else ""
    )
    return program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "OPERATOR=-0.2*fq.I()+0.7*fq.Y(0)-0.3*(fq.X(0)@fq.Z(1))\n"
        "@qfunc\ndef main(time:float,q:Output[QArray[QBit]]):\n"
        + preparation
        + f"    {call}\n"
    )


def reference(time: float, count: int, seed: int) -> fq.Circuit:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).h(1).rz(1, theta=0.49)
    identity = torch.eye(4, dtype=torch.complex128)
    i = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    matrices = (identity, torch.kron(i, y), torch.kron(z, x))
    cumulative = []
    total = 0.0
    for coefficient in COEFFICIENTS:
        total += abs(coefficient)
        cumulative.append(total)
    rng = random.Random(seed)
    step = time * math.fsum(abs(c) for c in COEFFICIENTS) / count
    product = identity
    for _ in range(count):
        index = bisect.bisect_right(cumulative, rng.random() * total, 0, 2)
        angle = step if COEFFICIENTS[index] > 0 else -step
        product = (
            math.cos(angle) * identity - 1j * math.sin(angle) * matrices[index]
        ) @ product
    circuit.any(0, 1, matrix=product)
    return circuit


@pytest.mark.parametrize("time", (0.37, -0.81))
@pytest.mark.parametrize("count", (1, 17, 2048))
@pytest.mark.parametrize("seed", (0, 23, 2**32 - 1))
def test_independent_cdf_coherence_identity_phase_and_seed(
    time: float, count: int, seed: int
) -> None:
    entry = definition(f"qdrift(OPERATOR,time,{count},q,{seed})")
    state = random.getstate()
    program = synth.synthesize(entry, classical_inputs={"time": time}, max_workspace=0)
    assert random.getstate() == state
    compare_states(program, reference(time, count, seed), ("q",))
    assert program.report["workspace_qubits"] == 0
    if count == 17:
        again = synth.synthesize(entry, classical_inputs={"time": time})
        assert program.circuit.instructions == again.circuit.instructions
        if seed == 0:
            default = synth.synthesize(
                definition(f"qdrift(OPERATOR,time,{count},q)"),
                classical_inputs={"time": time},
            )
            assert program.circuit.instructions == default.circuit.instructions


@pytest.mark.parametrize(
    "call,diagnostic",
    (
        ("qdrift(OPERATOR,time,0,q)", "samples"),
        ("qdrift(OPERATOR,time,2049,q)", "samples"),
        ("qdrift(OPERATOR,time,True,q)", "integer"),
        ("qdrift(OPERATOR,time,2,q,-1)", "seed"),
        ("qdrift(OPERATOR,time,2,q,4294967296)", "seed"),
        ("qdrift(OPERATOR,time,2,q,True)", "integer"),
        ("qdrift(OPERATOR,time,2,q[1:])", "support"),
        ("qdrift(OPERATOR,q,2,q)", "classical"),
        ("qdrift(OPERATOR,1.7e308,2,q)", "finite"),
        ("qdrift(OPERATOR,time,2,q,0,1)", "signature"),
        ("qdrift(OPERATOR,time,2,q,seed=0)", "positional"),
    ),
)
def test_invalid_sampled_calls_fail_before_execution(
    call: str, diagnostic: str
) -> None:
    with pytest.raises(
        (HybridVerificationError, synth.SynthesisError), match=diagnostic
    ):
        synth.synthesize(definition(call), classical_inputs={"time": 0.37})


def test_helper_composition_and_budget_refusals() -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "OPERATOR=-0.2*fq.I()+0.7*fq.Y(0)-0.3*(fq.X(0)@fq.Z(1))\n"
        "@qfunc\ndef evolve(time:float,q:QArray[QBit]):\n"
        "    qdrift(OPERATOR,time,17,q,23)\n"
        "@qfunc\ndef main(time:float,q:Output[QArray[QBit]]):\n"
        "    allocate(2,q)\n    hadamard_transform(q)\n    RZ(0.49,q[0])\n"
        "    evolve(time,q[:])\n"
    )
    program = synth.synthesize(entry, classical_inputs={"time": 0.37})
    compare_states(program, reference(0.37, 17, 23), ("q",))
    for qubits, operations, depth, memory in (
        (1, None, None, None),
        (None, 1, None, None),
        (None, None, 1, None),
        (None, None, None, 1),
    ):
        with pytest.raises(ValueError):
            synth.synthesize(
                entry,
                classical_inputs={"time": 0.37},
                max_qubits=qubits,
                max_operations=operations,
                max_depth=depth,
                memory_limit_bytes=memory,
            )
    with pytest.raises(TypeError, match="captured inside qfunc"):
        getattr(synth, "qdrift")(fq.X(0), 0.3, 17, None)


@pytest.mark.parametrize("zero", ("0.0", "time"))
def test_zero_time_and_operator_keep_storage_checks(zero: str) -> None:
    operator = "0.0*fq.X(0)" if zero == "time" else "fq.X(0)"
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        f"OPERATOR={operator}\n@qfunc\ndef main(time:float,q:Output[QBit]):\n"
        f"    allocate(q)\n    H(q)\n    qdrift(OPERATOR,{zero},17,q)\n"
    )
    program = synth.synthesize(entry, classical_inputs={"time": 0.37})
    compare_states(program, fq.Circuit(1, dtype=torch.complex128).h(0), ("q",))
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition("qdrift(OPERATOR,0.0,17,q)", allocated=False),
            classical_inputs={"time": 0.37},
        )


@pytest.mark.parametrize(
    "helper,body",
    (
        (
            "@qfunc\ndef evolve(q:Const[QNum[2]]):\n    qdrift(OPERATOR,0.3,17,q)\n",
            "evolve(q)",
        ),
        (
            "@qfunc\ndef evolve(a:QArray[QBit],b:QArray[QBit]):\n    qdrift(OPERATOR,0.3,17,a)\n",
            "evolve(q,q)",
        ),
        ("", "control(q[0],lambda:qdrift(OPERATOR,0.3,17,q))"),
    ),
)
def test_const_alias_and_unadmitted_control_are_rejected(
    helper: str, body: str
) -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "OPERATOR=fq.X(0)\n"
        + helper
        + "@qfunc\ndef main(q:Output[QArray[QBit]]):\n    allocate(2,q)\n"
        + f"    {body}\n"
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(entry)


def test_accumulated_effect_budget_is_checked_before_sampling() -> None:
    entry = definition("qdrift(OPERATOR,time,2048,q)\n    qdrift(OPERATOR,time,2048,q)")
    with pytest.raises(HybridVerificationError, match="4096 effect operations"):
        synth.synthesize(entry, classical_inputs={"time": 0.37})
