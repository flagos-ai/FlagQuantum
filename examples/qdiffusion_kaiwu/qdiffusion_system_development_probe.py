"""Development QDiffusion system slice with an explicit fake transport.

The probe exercises the pinned plugin's proposal, conditioned Boltzmann energy,
FlagQuantum sampler, objective, backward, optimizer, and guided generation
paths. It never contacts QBoson and can never produce acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import socket
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from examples.qdiffusion_kaiwu.source_preflight import load_source_preflight
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import FrozenIsingMatrix, KaiwuTaskMode

HOSTS = {"jp-a800-171", "jp-a800-172"}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")


class _DevelopmentFakeClient:
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
            metadata={"fallback_occurred": False, "transport": "in_memory_fake"},
        )


class _ToyProposal(nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return self.output(torch.tanh(self.hidden(self.embedding(input_ids))))


def _execute_qdiffusion_slice(
    device: torch.device,
    *,
    sampler: KaiwuSampler,
    remote_call_budget: int,
) -> dict[str, Any]:
    from kaiwu.torch_plugin import EnergyModel, QDiffusion, QDiffusionConfig
    from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

    class ToyBMEnergy(EnergyModel):
        def __init__(self, sampler: KaiwuSampler) -> None:
            super().__init__(bm_num_visible=4, bm_num_hidden=2, sampler=sampler)
            self.embedding = nn.Embedding(9, 8)
            self.feature_projector = nn.Linear(16, 4)

        @staticmethod
        def _pool(
            tokens: torch.Tensor, mask: torch.Tensor, embedding: nn.Embedding
        ) -> torch.Tensor:
            hidden = embedding(tokens)
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        def score_conditioned(
            self,
            noisy_tokens: torch.Tensor,
            candidate_tokens: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            noisy = self._pool(noisy_tokens, attention_mask, self.embedding)
            candidate = self._pool(candidate_tokens, attention_mask, self.embedding)
            return self.score_visible_logits(
                self.feature_projector(torch.cat((noisy, candidate), dim=-1))
            )

    torch.manual_seed(1701)
    energy_model = ToyBMEnergy(sampler)
    generator = QDiffusion(
        proposal_model=_ToyProposal(vocab_size=9, hidden_size=8),
        energy_model=energy_model,
        token_spec=SequenceTokenSpec(
            pad_id=0,
            bos_id=1,
            eos_id=2,
            mask_id=3,
            x_id=4,
        ),
        config=QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=2,
            proposal_temperature=0.0,
            disable_resample=True,
        ),
        device=device,
        freeze_proposal=True,
    ).to(device)
    targets = torch.tensor(
        [[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]],
        dtype=torch.long,
        device=device,
    )
    optimizer = torch.optim.SGD(energy_model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in energy_model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    outputs = generator.objective({"targets": targets})
    objective = outputs["energy_objective"].mean()
    if not bool(torch.isfinite(objective)):
        raise RuntimeError("QDiffusion energy objective is not finite")
    objective.backward()
    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.detach().square().sum().item())
            for parameter in energy_model.parameters()
            if parameter.grad is not None
        )
    )
    optimizer.step()
    parameter_delta = max(
        float((parameter.detach() - previous).abs().max().item())
        for parameter, previous in zip(energy_model.parameters(), before, strict=True)
    )
    if gradient_norm <= 0.0 or parameter_delta <= 0.0:
        raise RuntimeError("expected a finite non-zero energy gradient and update")

    with torch.no_grad():
        generated = generator.generate(targets, max_steps=1)
    if not isinstance(generated, torch.Tensor):
        raise RuntimeError("QDiffusion generation returned an unexpected state")
    token_constraints_passed = bool(
        generated.dtype == torch.long
        and generated.shape == targets.shape
        and torch.all((generated >= 0) & (generated < 9))
        and torch.all((generated[:, 1:3] >= 5) & (generated[:, 1:3] < 9))
        and torch.equal(generated[:, 0], targets[:, 0])
        and torch.equal(generated[:, 3:], targets[:, 3:])
    )
    if not token_constraints_passed:
        raise RuntimeError("generated tokens violated the declared constraints")
    if sampler.last_result is None:
        raise RuntimeError("QDiffusion did not produce a sampler result")
    if sampler.last_result.metadata.get("fallback_occurred") is not False:
        raise RuntimeError("development sampler did not declare no fallback")

    return {
        "objective": float(objective.detach().item()),
        "gradient_norm": gradient_norm,
        "parameter_delta_max": parameter_delta,
        "generated_tokens": generated.detach().cpu().tolist(),
        "token_constraints_passed": token_constraints_passed,
        "remote_call_budget": remote_call_budget,
        "remote_call_count": sampler.remote_call_count,
        "requested_samples_per_call": 10,
        "task_count": len(sampler.receipts),
        "proposal_device": str(next(generator.proposal_model.parameters()).device),
        "energy_device": str(next(generator.energy_model.parameters()).device),
        "generated_device": str(generated.device),
        "transfer_accounting": {
            "matrix_origin_device": str(
                next(generator.energy_model.parameters()).device
            ),
            "sampler_boundaries": [
                asdict(record) for record in sampler.transfer_records
            ],
            "returned_sample_target_device": str(
                next(generator.energy_model.parameters()).device
            ),
        },
        "fallback_occurred": False,
    }


def _run_qdiffusion_slice(device: torch.device) -> dict[str, Any]:
    client = _DevelopmentFakeClient()
    remote_call_budget = 64
    sampler = KaiwuSampler(
        client=client,
        task_name="qdiffusion-system-development",
        requested_samples=10,
        timeout=5.0,
        poll_interval=0.01,
        max_remote_calls=remote_call_budget,
    )
    return _execute_qdiffusion_slice(
        device,
        sampler=sampler,
        remote_call_budget=remote_call_budget,
    )


def run_probe(
    *,
    device_name: str,
    execution_host: str,
    expected_hostname: str,
    source_revision: str,
    plugin_revision: str,
    source_preflight_sha256: str,
    transfer_manifest_sha256: str,
    validation_image_id: str,
) -> dict[str, Any]:
    if execution_host not in HOSTS:
        raise ValueError("execution_host must be one of the two declared A800 hosts")
    if FULL_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be a full lowercase Git revision")
    if FULL_REVISION.fullmatch(plugin_revision) is None:
        raise ValueError("plugin_revision must be a full lowercase Git revision")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", validation_image_id) is None:
        raise ValueError("validation_image_id must be a full SHA-256 image identity")
    observed_hostname = socket.gethostname()
    if observed_hostname != expected_hostname:
        raise RuntimeError(
            f"expected hostname {expected_hostname!r}, observed {observed_hostname!r}"
        )
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("QDiffusion system probe requires an observed CUDA device")
    torch.cuda.set_device(device)
    observed_gpu = torch.cuda.get_device_name(device)
    if "A800" not in observed_gpu:
        raise RuntimeError(f"expected an NVIDIA A800, observed {observed_gpu!r}")

    slice_record = _run_qdiffusion_slice(device)
    if any(
        slice_record[field] != device_name
        for field in ("proposal_device", "energy_device", "generated_device")
    ):
        raise RuntimeError("QDiffusion tensor work left the requested CUDA device")
    return {
        "schema": "flagquantum.qboson_qdiffusion_system_development",
        "version": "1.0",
        "evidence_class": "development_fake_transport",
        "system_acceptance": False,
        "source_revision": source_revision,
        "kaiwu_pytorch_plugin_revision": plugin_revision,
        "source_preflight_sha256": source_preflight_sha256,
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "validation_image_id": validation_image_id,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "execution_host": execution_host,
        "observed_hostname": observed_hostname,
        "requested_cuda_device": device_name,
        "observed_tensor_device": slice_record["generated_device"],
        "observed_gpu_model": observed_gpu,
        "transport": "in_memory_fake",
        "qboson_hardware_used": False,
        "real_provider_evidence": False,
        **slice_record,
        "limitations": [
            "The transport is an in-memory fake and no QBoson task was submitted.",
            "The validation image is not the frozen Python 3.10 and Torch 2.7 lane.",
            "This is a single-host, single-device probe and not distributed execution.",
            "This does not constitute QDiffusion system or application acceptance.",
        ],
    }


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution-host", choices=sorted(HOSTS), required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--plugin-revision", required=True)
    parser.add_argument("--plugin-root", required=True, type=Path)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--validation-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.plugin_root.is_absolute():
        parser.error("--plugin-root must be an absolute path")
    source_preflight, source_preflight_sha256 = load_source_preflight(
        arguments.source_preflight,
        execution_host=arguments.execution_host,
        source_revision=arguments.source_revision,
        plugin_revision=arguments.plugin_revision,
        source_root=Path(__file__).resolve().parents[2],
        plugin_root=arguments.plugin_root,
    )
    payload = run_probe(
        device_name=arguments.device,
        execution_host=arguments.execution_host,
        expected_hostname=arguments.expected_hostname,
        source_revision=arguments.source_revision,
        plugin_revision=arguments.plugin_revision,
        source_preflight_sha256=source_preflight_sha256,
        transfer_manifest_sha256=source_preflight["manifest_sha256"],
        validation_image_id=arguments.validation_image_id,
    )
    _write_private_json(arguments.output, payload)
    print(f"Private QDiffusion development record written to {arguments.output}")


if __name__ == "__main__":
    main()
