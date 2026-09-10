"""Invalid numerical results must never become passing capability evidence."""

import pytest
import torch

from flagquantum.runtime import operator_probes
from flagquantum.runtime.capabilities import load_operator_profile

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "fault",
    ["output_nan", "output_inf", "output_shape", "gradient_nan", "gradient_inf"],
)
def test_probe_rejects_invalid_forward_results_and_gradients(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    calls = 0

    def execute(
        operator: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        nonlocal calls
        calls += 1
        leaf = torch.ones(2, dtype=dtype, device=device, requires_grad=requires_grad)
        if calls == 1 and fault.startswith("gradient_"):
            invalid = float("nan") if fault.endswith("nan") else float("inf")
            leaf.register_hook(lambda gradient: torch.full_like(gradient, invalid))
        output = leaf.square()
        if calls == 1:
            if fault == "output_shape":
                output = output.reshape(1, 2)
            elif fault.startswith("output_"):
                invalid = float("nan") if fault.endswith("nan") else float("inf")
                output = torch.full_like(output, invalid)
        return output, (leaf,)

    monkeypatch.setattr(operator_probes, "_execute_probe", execute)
    profile = load_operator_profile("statevector_local_p0")
    requirement = next(
        item for item in profile.requirements if item.operator == "aten::bmm"
    )
    evidence = operator_probes._probe_requirement(
        requirement,
        dtype_name="complex128",
        device=torch.device("cpu"),
        provider="fault_injection",
        profile_hash=profile.profile_hash,
    )
    assert not evidence.passed
    assert not evidence.backward
    assert evidence.forward == fault.startswith("gradient_")
    assert "error" in evidence.details
