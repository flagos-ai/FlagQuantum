"""Native RESET is a trace-preserving channel with checked ownership and effects."""

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    CX,
    RESET,
    Const,
    H,
    Output,
    QArray,
    QBit,
    adjoint,
    allocate,
    apply,
    control,
    qfunc,
    within,
)
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration


@qfunc
def bell(q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    H(q[0])
    CX(q[0], q[1])
    RESET(q[0])


@qfunc
def reset_helper(q: QBit) -> None:
    RESET(q)


@qfunc
def repeated(q: Output[QBit]) -> None:
    allocate(q)
    H(q)
    reset_helper(q)
    RESET(q)


def test_reset_bell_keeps_partner_mixed_and_matches_qmod() -> None:
    p = synth.synthesize(bell)
    assert p.execution.state_mode == "density_matrix"
    actual = fq.run(p).state
    assert isinstance(actual, torch.Tensor)
    expected = torch.diag(torch.tensor([0.5, 0, 0.5, 0], dtype=torch.complex128))
    torch.testing.assert_close(actual.reshape(4, 4), expected, atol=1e-12, rtol=0)
    assert float(torch.trace(expected @ expected).real) == 0.5
    original = synth.synthesize(
        "qfunc main(output q:qbit[2]) {allocate(q);H(q[0]);CX(q[0],q[1]);RESET(q[0]);}",
        source_format="qmod",
    )
    torch.testing.assert_close(fq.run(original).state, actual, atol=1e-12, rtol=0)
    assert dict(p.probabilities(fq.run(p), registers=("q",))) == pytest.approx(
        {(0,): 0.5, (2,): 0.5}
    )
    assert (
        sum(bool(gate.metadata.get("is_channel")) for gate in p.circuit.instructions)
        == 1
    )


def test_repeated_reset_and_mutable_helper() -> None:
    p = synth.synthesize(repeated)
    expected = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)
    torch.testing.assert_close(
        fq.run(p).state.reshape(2, 2), expected, atol=1e-12, rtol=0
    )
    assert (
        sum(bool(gate.metadata.get("is_channel")) for gate in p.circuit.instructions)
        == 2
    )


@qfunc
def inverse_reset(q: Output[QBit]) -> None:
    allocate(q)
    with adjoint():
        RESET(q)


@qfunc
def within_reset(q: Output[QBit]) -> None:
    allocate(q)
    with within():
        RESET(q)
    with apply():
        H(q)


@qfunc
def controlled_reset(c: Output[QBit], q: Output[QBit]) -> None:
    allocate(c)
    allocate(q)
    control(c, lambda: reset_helper(q))


@qfunc
def const_reset(q: Const[QBit]) -> None:
    RESET(q)


@qfunc
def root_const(q: Output[QBit]) -> None:
    allocate(q)
    const_reset(q)


@qfunc
def uninitialized(q: Output[QBit]) -> None:
    RESET(q)


array_reset = program_definition(
    "from flagquantum.synth import Output,QArray,QBit,allocate,qfunc,RESET\n"
    "@qfunc\ndef main(q:Output[QArray[QBit]]) -> None:\n"
    "    allocate(2,q)\n    RESET(q)\n"
)


@pytest.mark.parametrize(
    "definition",
    [
        inverse_reset,
        within_reset,
        controlled_reset,
        root_const,
        uninitialized,
        array_reset,
    ],
)
def test_illegal_reset_contexts_refuse(definition: _QuantumFunction) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(definition)


def test_density_and_operation_budgets_refuse() -> None:
    with pytest.raises(PlanningError, match="memory|budget"):
        synth.synthesize(bell, memory_limit_bytes=128)
    with pytest.raises(
        (PlanningError, HybridVerificationError), match="budget|limit|exceed|infeasible"
    ):
        synth.synthesize(bell, max_operations=1)


def test_reset_is_exported_and_not_a_host_operation() -> None:
    import flagquantum.services.synthesis as services

    assert synth.RESET is services.RESET is RESET
    assert "RESET" in synth.__all__
    with pytest.raises(TypeError, match="captured"):
        getattr(RESET, "__call__")(None)
