"""Contract tests for the cost-aware classical/quantum optimizer benchmark."""

import torch

from benchmarks.hybrid_classical_quantum_optimizer import (
    effective_angles,
    run_experiment,
)


def test_classical_layer_gates_scale_separate_quantum_layers():
    classical = torch.zeros(2)
    quantum = torch.arange(6, dtype=torch.float32)
    torch.testing.assert_close(effective_angles(classical, quantum, depth=2), quantum)


def test_hybrid_optimizer_records_cost_axes_and_separate_assignments():
    common = dict(
        n_wires=2,
        depth=1,
        steps=1,
        seed=7,
        classical_lr=0.03,
        quantum_lr=0.05,
        damping=1e-3,
        dtype=torch.float64,
    )
    adam = run_experiment(method="all_adam", **common)
    hybrid = run_experiment(method="adam_block_qng", **common)

    assert adam["optimizer_assignment"] == {"classical": "adam", "quantum": "adam"}
    assert hybrid["optimizer_assignment"] == {
        "classical": "adam",
        "quantum": "block_qng",
    }
    for experiment in (adam, hybrid):
        assert len(experiment["trace"]) == 2
        assert experiment["trace"][-1]["optimizer_step"] == 1
        assert experiment["trace"][-1]["circuit_evaluations"] > 0
        assert experiment["trace"][-1]["wall_time_seconds"] >= 0
    assert hybrid["total_circuit_evaluations"] > adam["total_circuit_evaluations"]
