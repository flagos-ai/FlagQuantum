"""Development-only A800 smoke probe for the Kaiwu sampler data path.

This probe intentionally uses an in-memory remote client. It proves tensor
placement, plugin interoperability, transfers, backward, and optimizer update;
it is not QBoson hardware evidence and cannot pass QDiffusion system acceptance.
"""

from __future__ import annotations

import argparse
import math
import platform
import re
import socket
from pathlib import Path
from typing import Any

import torch

from examples.qdiffusion_kaiwu.private_io import write_private_json_exclusive
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import (
    FrozenIsingMatrix,
    KaiwuTaskMode,
)

HOSTS = {"jp-a800-171", "jp-a800-172"}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")


class _DevelopmentFakeClient:
    """Deterministic fake transport; never imports or contacts Kaiwu SDK."""

    def __init__(self) -> None:
        self.submit_calls = 0

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submit_calls += 1
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id=f"development-fake-{self.submit_calls}",
            provider_target="in-memory-fake",
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = tuple(1 for _ in matrix)
        samples = tuple(sample for _ in range(receipt.requested_samples))
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        return KaiwuTaskResult(
            receipt=receipt,
            samples=samples,
            energies=tuple(energy for _ in samples),
            raw_status="Completed",
            metadata={
                "fallback_occurred": False,
                "transport": "in_memory_fake",
            },
        )


def run_probe(
    *,
    device_name: str,
    execution_host: str,
    expected_hostname: str,
    source_revision: str,
    plugin_revision: str,
) -> dict[str, Any]:
    """Run one bounded plugin/sampler/backward slice and return evidence."""

    from kaiwu.torch_plugin import BoltzmannMachine

    if execution_host not in HOSTS:
        raise ValueError("execution_host must be one of the two declared A800 hosts")
    if FULL_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be a full lowercase Git revision")
    if FULL_REVISION.fullmatch(plugin_revision) is None:
        raise ValueError("plugin_revision must be a full lowercase Git revision")
    observed_hostname = socket.gethostname()
    if observed_hostname != expected_hostname:
        raise RuntimeError(
            f"expected hostname {expected_hostname!r}, observed {observed_hostname!r}"
        )
    device = torch.device(device_name)
    if device != torch.device("cuda:0"):
        raise RuntimeError("A800 sampler smoke requires explicit cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("A800 sampler smoke requires an observed CUDA device")
    torch.cuda.set_device(device)
    observed_gpu = torch.cuda.get_device_name(device)
    if "A800" not in observed_gpu:
        raise RuntimeError(f"expected an NVIDIA A800, observed {observed_gpu!r}")

    torch.manual_seed(17)
    machine = BoltzmannMachine(num_nodes=4, device=device).to(device)
    with torch.no_grad():
        machine.linear_bias.copy_(torch.tensor([0.0, 0.5, -0.25, 0.75], device=device))
        machine.quadratic_coef.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, -0.5],
                    [0.0, 0.0, -0.75, 0.25],
                    [1.0, -0.75, 0.0, 0.0],
                    [-0.5, 0.25, 0.0, 0.0],
                ],
                device=device,
            )
        )

    client = _DevelopmentFakeClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="a800-development-smoke",
        requested_samples=10,
        max_remote_calls=2,
        timeout=5.0,
        poll_interval=0.01,
    )
    visible = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
    negative = machine.condition_sample(sampler, visible)
    positive = torch.randint(0, 2, negative.shape, device=device).to(torch.float32)

    optimizer = torch.optim.SGD(machine.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in machine.parameters()]
    optimizer.zero_grad(set_to_none=True)
    objective = machine.objective(positive, negative)
    if not bool(torch.isfinite(objective)):
        raise RuntimeError("Boltzmann objective is not finite")
    objective.backward()
    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.detach().square().sum().item())
            for parameter in machine.parameters()
            if parameter.grad is not None
        )
    )
    optimizer.step()
    parameter_delta = max(
        float((parameter.detach() - previous).abs().max().item())
        for parameter, previous in zip(machine.parameters(), before, strict=True)
    )
    if gradient_norm <= 0.0 or parameter_delta <= 0.0:
        raise RuntimeError("expected a finite non-zero gradient and parameter update")
    if negative.device != device:
        raise RuntimeError("plugin samples did not return to the requested CUDA device")

    return {
        "schema": "flagquantum.qboson_kaiwu_a800_sampler_probe",
        "version": "1.0",
        "evidence_class": "development_fake_transport",
        "system_acceptance": False,
        "source_revision": source_revision,
        "kaiwu_pytorch_plugin_revision": plugin_revision,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "execution_host": execution_host,
        "observed_hostname": observed_hostname,
        "requested_cuda_device": device_name,
        "observed_tensor_device": str(negative.device),
        "observed_gpu_model": observed_gpu,
        "transport": "in_memory_fake",
        "qboson_hardware_used": False,
        "real_provider_evidence": False,
        "fallback_occurred": False,
        "requested_samples_per_call": 10,
        "returned_samples": int(negative.shape[0]),
        "remote_call_budget": 2,
        "remote_call_count": sampler.remote_call_count,
        "a800_to_cpu_matrix_transfers": sampler.remote_call_count,
        "cpu_to_a800_sample_transfers": sampler.remote_call_count,
        "objective": float(objective.detach().item()),
        "gradient_norm": gradient_norm,
        "parameter_delta_max": parameter_delta,
        "limitations": [
            "The transport is an in-memory fake and no QBoson task was submitted.",
            "This environment is not the frozen composite Python 3.10, Torch 2.7, and Kaiwu 1.3.1 lane.",
            "This is a single-host, single-device probe and not distributed execution.",
            "This does not constitute QDiffusion system or application acceptance.",
        ],
    }


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    write_private_json_exclusive(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution-host", choices=sorted(HOSTS), required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--plugin-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run_probe(
        device_name=arguments.device,
        execution_host=arguments.execution_host,
        expected_hostname=arguments.expected_hostname,
        source_revision=arguments.source_revision,
        plugin_revision=arguments.plugin_revision,
    )
    _write_private_json(arguments.output, payload)
    print(f"Private A800 development record written to {arguments.output}")


if __name__ == "__main__":
    main()
