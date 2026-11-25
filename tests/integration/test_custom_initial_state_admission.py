"""Planning must not reinterpret live input tensors as default zero states."""

from collections.abc import Callable
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("entry", ["run", "plan", "method"])
@pytest.mark.parametrize("state_kind", ["basis", "entangled", "zero", "batched"])
def test_custom_initial_state_refuses_before_compilation_or_execution(
    entry: Literal["run", "plan", "method"],
    state_kind: Literal["basis", "entangled", "zero", "batched"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = {
        "basis": torch.tensor([0, 1, 0, 0], dtype=torch.complex128),
        "entangled": torch.tensor([1, 0, 0, 1j], dtype=torch.complex128) / 2**0.5,
        "zero": torch.tensor([1, 0, 0, 0], dtype=torch.complex128),
        "batched": torch.tensor([[0, 1, 0, 0], [1, 0, 0, 1j]], dtype=torch.complex128),
    }
    initial = states[state_kind].requires_grad_()
    saved = initial.detach().clone()
    circuit = (
        fq.Circuit(
            2,
            bsz=2 if state_kind == "batched" else 1,
            inputs=initial,
            dtype=torch.complex128,
        )
        .h(0)
        .cx(0, 1)
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unsupported input reached compilation or execution")

    monkeypatch.setattr("flagquantum.runtime.planner.compile_program", forbidden)
    monkeypatch.setattr("flagquantum.runtime.execution.run_native", forbidden)
    calls: dict[str, Callable[[], object]] = {
        "run": lambda: fq.run(circuit, outputs=fq.probabilities()),
        "plan": lambda: fq.plan(circuit, outputs=fq.probabilities()),
        "method": circuit.run,
    }
    with pytest.raises(CapabilityError, match="custom initial states"):
        calls[entry]()
    torch.testing.assert_close(initial, saved, atol=0, rtol=0)
    assert initial.grad is None


@pytest.mark.parametrize(
    "mode",
    ["auto", "statevector", "density_matrix", "mps", "tensor_network", "stabilizer"],
)
def test_input_admission_precedes_representation_selection(mode: str) -> None:
    circuit = fq.Circuit(1, inputs=torch.tensor([0, 1], dtype=torch.complex128))
    with pytest.raises(CapabilityError, match="custom initial states"):
        fq.run(circuit, options=fq.ExecutionOptions(mode=mode))


def test_direct_state_preserves_complex_input_and_input_gradients() -> None:
    initial = (
        torch.tensor([1, 0, 0, 1j], dtype=torch.complex128) / 2**0.5
    ).requires_grad_()
    circuit = fq.Circuit(2, inputs=initial, dtype=torch.complex128).cx(0, 1).s(0)
    observed = circuit.state().reshape(-1)
    # CX maps |11> to |10>; S supplies another factor i on that branch.
    expected = torch.tensor([1, 0, -1, 0], dtype=torch.complex128) / 2**0.5
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    observed[2].real.backward()
    assert initial.grad is not None
    torch.testing.assert_close(
        initial.grad,
        torch.tensor([0, 0, 0, -1j], dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )


def test_gate_preparation_preserves_direct_plan_and_json_replay() -> None:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).s(0).cx(0, 1)
    expected = torch.tensor([[1, 0, 0, 1j]], dtype=torch.complex128) / 2**0.5
    plan = fq.plan(circuit)
    restored = fq.ExecutionPlan.from_json(plan.to_json())
    for program in (circuit, plan, restored):
        result = fq.run(program)
        torch.testing.assert_close(result.statevector(), expected, atol=1e-12, rtol=0)
        if isinstance(program, fq.ExecutionPlan):
            assert result.plan is program
