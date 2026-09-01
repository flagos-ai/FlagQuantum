import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

MODULE = (
    Path(__file__).parents[2]
    / "benchmarks/internal/mps_hamiltonian_identification/core.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mps_hamiltonian_identification_core", MODULE
)
core = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def test_system_identification_prediction_is_differentiable_and_mps_native(monkeypatch):
    coupling, field = core.smooth_couplings(4)
    coupling = coupling.clone().requires_grad_(True)
    field = field.clone().requires_grad_(True)
    probe = core.Probe(time_steps=1, flipped_sites=(1,))

    import flagquantum.simulation.mps as mps_module

    monkeypatch.setattr(
        mps_module.MPSState,
        "to_statevector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dense fallback")
        ),
    )
    values = core.predict(
        coupling,
        field,
        probe,
        observation_sites=(0, 2),
        dt=0.08,
        max_bond=8,
        cutoff=0.0,
        device="cpu",
    )
    loss = values.square().sum()
    loss.backward()
    assert values.shape == (4,)
    assert coupling.grad is not None and torch.isfinite(coupling.grad).all()
    assert field.grad is not None and torch.isfinite(field.grad).all()


def test_probe_generation_is_reproducible_and_covers_times():
    first = core.make_probes(16, n_initial_states=3, time_steps=(1, 3), seed=7)
    second = core.make_probes(16, n_initial_states=3, time_steps=(1, 3), seed=7)
    assert first == second
    assert len(first) == 6
    assert {probe.time_steps for probe in first} == {1, 3}


def test_batched_probe_predictions_match_individual_predictions_and_gradients():
    coupling, field = core.smooth_couplings(6)
    probes = (
        core.Probe(time_steps=2, flipped_sites=(1,)),
        core.Probe(time_steps=2, flipped_sites=(3,)),
    )
    batch_j = coupling.clone().requires_grad_(True)
    batch_h = field.clone().requires_grad_(True)
    batched = core.predict_batch(
        batch_j,
        batch_h,
        probes,
        observation_sites=(0, 2, 4),
        dt=0.08,
        max_bond=32,
        cutoff=0.0,
        device="cpu",
    )
    individual = torch.stack(
        [
            core.predict(
                coupling,
                field,
                probe,
                observation_sites=(0, 2, 4),
                dt=0.08,
                max_bond=32,
                cutoff=0.0,
                device="cpu",
            )
            for probe in probes
        ]
    )
    torch.testing.assert_close(batched, individual, atol=1e-5, rtol=1e-5)
    batched.square().mean().backward()
    assert batch_j.grad is not None and torch.isfinite(batch_j.grad).all()
    assert batch_h.grad is not None and torch.isfinite(batch_h.grad).all()


def test_shape_bucketed_brickwork_matches_circuit_interpreter():
    from flagquantum.simulation.mps_brickwork import run_batched_brickwork_mps

    coupling, field = core.smooth_couplings(6)
    probes = (
        core.Probe(time_steps=2, flipped_sites=(1,)),
        core.Probe(time_steps=2, flipped_sites=(3,)),
    )
    reference = core.predict_batch(
        coupling,
        field,
        probes,
        observation_sites=(0, 2, 4),
        dt=0.08,
        max_bond=32,
        cutoff=0.0,
        device="cpu",
    )
    state = run_batched_brickwork_mps(
        coupling,
        field,
        [probe.flipped_sites for probe in probes],
        time_steps=2,
        dt=0.08,
        max_bond=32,
        compiled=False,
    )
    actual = core.local_observables(state, (0, 2, 4))
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)


def test_real_environment_scan_matches_complex_observable_path():
    from flagquantum.simulation.mps_brickwork import (
        compiled_local_z_zz,
        run_batched_brickwork_mps,
    )

    coupling, field = core.smooth_couplings(8)
    probes = core.make_probes(8, n_initial_states=2, time_steps=(2,), seed=3)
    state = run_batched_brickwork_mps(
        coupling,
        field,
        [probe.flipped_sites for probe in probes],
        time_steps=2,
        dt=0.08,
        max_bond=32,
        compiled=False,
    )
    reference = state.expectation_z_and_nearest_neighbor_zz((0, 2, 4, 6))
    actual = compiled_local_z_zz(state, (0, 2, 4, 6), compiled=False)
    torch.testing.assert_close(actual[0], reference[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], reference[1], atol=1e-5, rtol=1e-5)


def test_training_cli_exposes_fail_closed_approximate_gradient_contract():
    script = MODULE.with_name("train.py")
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--gradient-policy {exact,approximate}" in help_result.stdout
    assert "--discarded-weight-tolerance" in help_result.stdout
    assert "--single-step-acceptance" in help_result.stdout

    invalid = subprocess.run(
        [
            sys.executable,
            str(script),
            "--gradient-policy",
            "approximate",
            "--steps",
            "1",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode != 0
    assert "--gradient-policy approximate requires --site-sharded" in invalid.stderr
