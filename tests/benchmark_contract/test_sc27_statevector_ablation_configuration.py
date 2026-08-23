from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.statevector_training_scaling import _validate_ablation_configuration


def configuration(ablation_id: str = "full", **overrides):
    values = {
        "ablation_id": ablation_id,
        "disable_persistent_layout": False,
        "disable_gradient_overlap": False,
        "disable_gradient_bucketing": False,
        "replicated_optimizer": False,
        "checkpoint_strategy": "reversible_adjoint",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("ablation_id", "override"),
    [
        ("full", {}),
        ("persistent_layout_off", {"disable_persistent_layout": True}),
        ("communication_overlap_off", {"disable_gradient_overlap": True}),
        ("gradient_bucketing_off", {"disable_gradient_bucketing": True}),
        ("unique_optimizer_ownership_off", {"replicated_optimizer": True}),
        (
            "reverse_rematerialization",
            {"checkpoint_strategy": "full_rematerialization"},
        ),
    ],
)
def test_accepts_exactly_one_frozen_factor(ablation_id, override) -> None:
    _validate_ablation_configuration(configuration(ablation_id, **override))


def test_rejects_label_only_or_multi_factor_ablation() -> None:
    with pytest.raises(ValueError, match="requires"):
        _validate_ablation_configuration(configuration("persistent_layout_off"))
    with pytest.raises(ValueError, match="requires"):
        _validate_ablation_configuration(
            configuration(
                "communication_overlap_off",
                disable_gradient_overlap=True,
                disable_gradient_bucketing=True,
            )
        )
