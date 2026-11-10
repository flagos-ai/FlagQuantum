"""Dropped locals retain entanglement and physical cost without dirty work reuse."""

from typing import Annotated

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QBit, QNum, allocate, drop, qfunc
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("phase", [0.0, 0.73, -1.21])
def test_entangled_environment_marginal_and_absolute_state(phase: float) -> None:
    source = definition(
        "q:Output[QBit]",
        f"s:QBit\nallocate(q)\nallocate(s)\nH(s)\nPHASE({phase},s)\nCX(s,q)\ndrop(s)",
    )
    program = synth.synthesize(source, max_workspace=0)
    assert tuple(program.registers) == ("q",)
    assert program.report["workspace_qubits"] == 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 2
    result = fq.run(program)
    reference = fq.Circuit(2, dtype=torch.complex128).h(1).phase(1, phase).cx(1, 0)
    torch.testing.assert_close(
        result.statevector().flatten(),
        fq.run(reference).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )
    marginal = dict(program.probabilities(result, registers=("q",)))
    assert marginal == pytest.approx({(0,): 0.5, (1,): 0.5})
    state = result.statevector().reshape(2, 2)
    torch.testing.assert_close(
        state @ state.conj().T,
        torch.eye(2, dtype=torch.complex128) / 2,
        atol=1e-12,
        rtol=0,
    )
    assert float(state[:, 1].abs().square().sum()) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kind,initialize,width",
    [
        ("QBit", "allocate(s)\nX(s)", 1),
        ("QNum[3]", "s |= 5", 3),
        ("QArray[QBit]", "prepare_amplitudes([0.6,0,-0.8,0],0,s)", 2),
        ("QNum[3,True,1]", "allocate(s)", 3),
    ],
)
def test_hidden_types_and_mutable_borrow(
    kind: str, initialize: str, width: int
) -> None:
    helper = "@qfunc\ndef gate(q:QBit):\n    PHASE(0.7,q)\n"
    code = f"s:{kind}\nallocate(q)\n{initialize}\ngate(q)\ndrop(s)"
    source = definition("q:Output[QBit]", code, helper)
    program = synth.synthesize(source, max_workspace=0)
    assert fq.run(program).statevector().numel() == 1 << (width + 1)
    root, _ = capture_program(source)
    assert sum(op.name == "qreg.hidden" for op in root.body.blocks[0].operations) == 1
    assert sum(op.name == "qreg.drop" for op in root.body.blocks[0].operations) == 1


@pytest.mark.parametrize(
    "body",
    [
        "s:QBit\nallocate(s)",
        "s:QBit\ndrop(s)",
        "s:QBit\nallocate(s)\ndrop(s)\ndrop(s)",
        "s:QBit\nallocate(s)\ndrop(s)\nH(s)",
        "s:QBit\nallocate(s)\ndrop(s)\nallocate(s)",
        "s:QBit\nallocate(s)\ndrop(s)\nq ^= s",
        "s:QArray[QBit]\nallocate(2,s)\ndrop(s)\nH(s[0])",
        "s:QArray[QBit]\nallocate(2,s)\ndrop(s[0])",
        "s:QBit\nallocate(s)\ndrop(q)",
        "s:QBit\nallocate(s)\nwith adjoint():\n    drop(s)",
        "s:QBit\nallocate(s)\nwith within():\n    H(s)\nwith apply():\n    drop(s)",
        "s:QBit\nallocate(s)\nfor i in range(1):\n    drop(s)",
        "s:QBit\nallocate(s)\nif True:\n    drop(s)",
        "for i in range(1):\n    s:QBit\n    allocate(s)\n    drop(s)",
        "if True:\n    s:QBit\n    allocate(s)\n    drop(s)",
        "s:QBit\ns:QBit\nallocate(s)\ndrop(s)",
        "q:QBit",
        "s:Output[QBit]\nallocate(s)\ndrop(s)",
        "s:QBit=0",
        "s:QBit\nallocate(s)\ncontrol(q,lambda:drop(s))\ndrop(s)",
        "s:QBit\nallocate(s)\nfree(s)",
        "s:QBit\nallocate(s)\ndrop(s)\ngate(s)",
        "s:QBit\nallocate(s)\npair(s,s)\ndrop(s)",
    ],
)
def test_ownership_and_post_drop_refusals(body: str) -> None:
    helper = "@qfunc\ndef gate(a:QBit):\n    H(a)\n@qfunc\ndef pair(a:QBit,b:QBit):\n    CX(a,b)\n"
    with pytest.raises((HybridVerificationError, TypeError)):
        synth.synthesize(definition("q:Output[QBit]", "allocate(q)\n" + body, helper))


def test_helper_cannot_drop_or_allocate_hidden_storage() -> None:
    for body in ("drop(a)", "s:QBit\n    allocate(s)\n    drop(s)"):
        helper = "@qfunc\ndef gate(a:QBit):\n    " + body + "\n"
        with pytest.raises(HybridVerificationError):
            synth.synthesize(
                definition("q:Output[QBit]", "allocate(q)\ngate(q)", helper)
            )


def test_hidden_storage_is_not_recycled_and_physical_budgets_apply() -> None:
    source = definition(
        "q:Output[QBit]",
        "allocate(q)\ns:QBit\nallocate(s)\nX(s)\ndrop(s)\nt:QBit\nallocate(t)\nCX(t,q)\ndrop(t)",
    )
    program = synth.synthesize(source, max_workspace=0)
    assert fq.run(program).statevector().numel() == 8
    with pytest.raises((HybridVerificationError, PlanningError, ValueError)):
        synth.synthesize(source, max_qubits=2)
    with pytest.raises((HybridVerificationError, PlanningError, ValueError)):
        synth.synthesize(source, memory_limit_bytes=16)


def test_annotation_only_local_namespace_is_retained() -> None:
    from flagquantum.synth import QNum as LocalNumber

    @qfunc
    def main(q: Output[QBit]) -> None:
        s: Annotated[LocalNumber, 2] = QNum()
        allocate(q)
        allocate(s)
        drop(s)

    assert fq.run(synth.synthesize(main)).statevector().numel() == 8


def test_drop_is_capture_only() -> None:
    with pytest.raises(TypeError, match="captured"):
        getattr(synth, "drop")(1)


def test_hidden_storage_cannot_escape_through_output_helper() -> None:
    helper = "@qfunc\ndef init(a:Output[QBit]):\n    allocate(a)\n"
    with pytest.raises(HybridVerificationError, match="escape"):
        synth.synthesize(
            definition(
                "q:Output[QBit]", "allocate(q)\ns:QBit\ninit(s)\ndrop(s)", helper
            )
        )


@pytest.mark.parametrize(
    "annotation,initializer",
    [
        ("QBit", "QNum()"),
        ("QBit", "QBit(1)"),
        ("QBit", "QBit(size=1)"),
        ("QBit", "unknown()"),
        ("QArray[QBit]", "QBit()"),
        ("QNum[2]", "[0,0]"),
    ],
)
def test_hidden_handle_constructor_is_checked(
    annotation: str, initializer: str
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "q:Output[QBit]",
                f"allocate(q)\ns:{annotation}={initializer}\nallocate(s)\ndrop(s)",
            )
        )
