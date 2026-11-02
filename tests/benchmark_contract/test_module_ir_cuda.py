"""Real CUDA public-Module conformance, not a GPU performance comparison."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from tools.sanitize_public_evidence import sanitize_text

pytestmark = [pytest.mark.gpu, pytest.mark.benchmark_contract]


def _publish(record):
    output = os.environ.get("FQ_MODULE_CUDA_RECEIPTS")
    if not output:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "schema": "flagquantum.module_ir_cuda.conformance.v1",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, timeout=10
            ).strip(),
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "distribution_semantics": "single_device_fast_path",
            "scalability_claim_allowed": False,
            "scope": "GPU numerical/recovery conformance; no timing, ranking, memory or capacity claim",
            "checkpoint_scope": "fresh Module and optimizer in the same process",
            "cases": [],
        }
    )
    payload["cases"].append(record)
    path.write_text(sanitize_text(json.dumps(payload, indent=2) + "\n"))


def _build(p, x):
    return (
        fq.Circuit(2, bsz=x.shape[0], device=p.device)
        .ry(0, p[0].square() + p[1] * x[:, 0].sin())
        .rx(1, p[0].detach() * x[:, 1])
        .cx(0, 1)
    )


def _model(backend):
    return fq.Module(
        _build,
        2,
        init=torch.tensor([0.23, -0.31], dtype=torch.float64),
        dtype=torch.float64,
        device="cuda:0",
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(backend=backend),
            observable="z_sum",
            observable_qubits=(0, 1),
        ),
    )


def _analytic(p, x):
    return (p[0].square() + p[1] * x[:, 0].sin()).cos() * (
        1 + (p[0].detach() * x[:, 1]).cos()
    )


@pytest.mark.parametrize("backend", ("pytorch", "jax"))
@pytest.mark.parametrize("batch", (1, 32))
def test_cuda_public_module_vjps_adam_validation_and_recovery(
    backend, batch, monkeypatch, tmp_path
):
    if not torch.cuda.is_available():
        pytest.skip("real CUDA is required")
    record = {
        "backend": backend,
        "batch": batch,
        "device": "cuda:0",
        "dtype": "complex128/float64",
        "gpu": torch.cuda.get_device_name(0),
        "updates": 6,
        "status": "running",
    }
    seen_devices = set()
    try:
        if backend == "jax":
            try:
                jax = importlib.import_module("jax")
                devices = jax.devices("gpu")
            except (ImportError, RuntimeError) as error:
                record.update(status="unsupported", reason=type(error).__name__)
                pytest.skip("installed JAX CUDA backend is unavailable")
            assert devices
            kernel = importlib.import_module("flagquantum.runtime.executors.jax.kernel")
            original = kernel._jax_to_torch

            def checked(value, *, like):
                platforms = {device.platform for device in value.devices()}
                assert platforms <= {"gpu", "cuda"} and platforms
                assert like.device.type == "cuda"
                seen_devices.update(platforms)
                return original(value, like=like)

            monkeypatch.setattr(kernel, "_jax_to_torch", checked)
        model = _model(backend)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        reference = model.parameters_tensor.detach().clone().requires_grad_()
        reference_optimizer = torch.optim.Adam([reference], lr=0.03)
        restored = None
        maximum = dict.fromkeys(
            (
                "value",
                "parameter_vjp",
                "input_vjp",
                "parameter",
                "recovery_parameter",
                "recovery_adam",
            ),
            0.0,
        )
        losses = []

        def compare(name, actual, expected):
            maximum[name] = max(
                maximum[name], float((actual.detach() - expected.detach()).abs().max())
            )
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)

        for step in range(6):
            features = (
                torch.linspace(
                    -0.3, 0.5, batch * 4, dtype=torch.float64, device="cuda"
                ).reshape(batch, 4)
                + step * 0.01
            )[:, ::2].requires_grad_()
            reference_features = features.detach().clone().requires_grad_()
            weights = torch.linspace(
                0.4, 1.6, batch, dtype=torch.float64, device="cuda"
            )
            target = torch.linspace(
                -0.2, 0.4, batch, dtype=torch.float64, device="cuda"
            )
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            value, expected = model(features), _analytic(reference, reference_features)
            assert value.device.type == "cuda" and value.dtype == torch.float64
            compare("value", value, expected)
            loss = ((value - target).square() * weights).mean()
            loss.backward()
            (((expected - target).square() * weights).mean()).backward()
            compare("parameter_vjp", model.parameters_tensor.grad, reference.grad)
            compare("input_vjp", features.grad, reference_features.grad)
            optimizer.step()
            reference_optimizer.step()
            compare("parameter", model.parameters_tensor, reference)
            losses.append(float(loss.detach()))
            if restored is not None:
                replay, replay_optimizer = restored
                replay_features = features.detach().clone().requires_grad_()
                replay_optimizer.zero_grad(set_to_none=True)
                (
                    ((replay(replay_features) - target).square() * weights).mean()
                ).backward()
                replay_optimizer.step()
                compare(
                    "recovery_parameter",
                    replay.parameters_tensor,
                    model.parameters_tensor,
                )
                for key, actual in optimizer.state_dict()["state"][0].items():
                    compare(
                        "recovery_adam",
                        actual,
                        replay_optimizer.state_dict()["state"][0][key],
                    )
            if step % 2 == 1:
                validation = features.detach() + 0.005
                with torch.no_grad():
                    torch.testing.assert_close(
                        model(validation),
                        _analytic(model.parameters_tensor, validation),
                        atol=1e-9,
                        rtol=1e-9,
                    )
                    if restored is not None:
                        torch.testing.assert_close(
                            restored[0](validation),
                            model(validation),
                            atol=1e-9,
                            rtol=1e-9,
                        )
            if step == 2:
                checkpoint = model.save_checkpoint(
                    tmp_path / "module.pt", optimizer=optimizer, seed=11, step=3
                )
                record["checkpoint_sha256"] = hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest()
                replay = _model(backend)
                replay_optimizer = torch.optim.Adam(replay.parameters(), lr=0.03)
                replay(features.detach().clone().requires_grad_())
                replay.load_checkpoint(checkpoint, optimizer=replay_optimizer)
                restored = replay, replay_optimizer
        consumed = model._last_ir.content_hash
        with torch.no_grad():
            execution = model.execute(validation)
        assert execution.provenance["ir_hash"] == consumed
        assert model._builder_compile_count == 1
        assert model._jax_kernel_build_count == int(backend == "jax")
        assert model.parameters_tensor.device.type == "cuda"
        if backend == "jax":
            assert seen_devices
        record.update(
            status="passed",
            maximum_absolute_errors=maximum,
            losses=losses,
            builder_captures=model._builder_compile_count,
            jax_kernel_builds=model._jax_kernel_build_count,
            builder_lowering=execution.runtime["builder_lowering"],
            consumed_ir_hash=consumed,
            jax_array_platforms=sorted(seen_devices),
            final_parameters=model.parameters_tensor.detach().cpu().tolist(),
        )
    except BaseException as error:
        if record["status"] != "unsupported":
            record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        _publish(record)
