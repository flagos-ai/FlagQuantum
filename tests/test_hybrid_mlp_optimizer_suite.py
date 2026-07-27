import torch

from benchmarks.hybrid_mlp_quantum_optimizer_suite import (
    conditioned_angles,
    make_conditioner,
    parameter_features,
    run_method,
)


def test_mlp_conditioner_starts_as_identity_and_aligns_with_hva():
    features = parameter_features(4, 2, torch.float64)
    conditioner = make_conditioner(torch.float64, seed=3, hidden=8)
    quantum = torch.linspace(-0.2, 0.2, features.shape[0], dtype=torch.float64)
    angles, gain, bias = conditioned_angles(conditioner, features, quantum)
    torch.testing.assert_close(angles, quantum)
    torch.testing.assert_close(gain, torch.ones_like(gain))
    torch.testing.assert_close(bias, torch.zeros_like(bias))


def test_all_quantum_methods_emit_cost_aware_traces():
    common = dict(
        n_wires=2,
        depth=1,
        steps=1,
        seed=5,
        hidden=4,
        classical_lr=0.01,
        quantum_lr=0.03,
        damping=1e-3,
        spsa_perturbation=0.08,
        lbfgs_max_iter=2,
        dtype=torch.float64,
    )
    results = [
        run_method(method=method, **common)
        for method in ("adam", "block_qng", "lbfgs", "spsa")
    ]
    assert {result["optimizer_assignment"]["quantum"] for result in results} == {
        "adam",
        "block_qng",
        "lbfgs",
        "spsa",
    }
    for result in results:
        assert len(result["trace"]) == 2
        assert result["total_circuit_evaluations"] >= 3
        assert result["total_wall_time_seconds"] >= 0
