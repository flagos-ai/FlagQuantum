#!/usr/bin/env python
"""Verify one CUDA statevector shard per node across forward, backward and resume.

The pair this runs on is the whole point and the whole limit: two ranks, one
device each, on two hosts. What it establishes is that a single logical
statevector workload is partitioned across the two, that the partitioned
gradient agrees with a single-device reference, and that an owner-sharded
optimizer step survives a checkpoint and a restart -- because a forward-only
probe cannot support a training claim and a training claim that stops at the
first optimizer step is not a training path either.

It makes no scale claim: the scope, the claim flags and the blockers in the
artifact say exactly how narrow the shape is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.executors.statevector.reverse import (
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)
from flagquantum.runtime.executors.statevector.training import (
    train_distributed_statevector,
)

ROOT = Path(__file__).resolve().parents[1]

#: The forward scope. Five wires at two ranks leaves three wires per shard, and
#: the circuit crosses the shard boundary, so the exchange is exercised.
N_WIRES = 5
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128

#: Numerical tolerance for every comparison against a single-device reference.
#: Complex128 with an exact adjoint should agree to round-off, so this is
#: round-off rather than a modelling allowance.
NUMERICAL_TOLERANCE = 1e-10
#: The same, for the multi-step training trajectory: two runs of four steps
#: accumulate more round-off than a single expectation does.
TRAINING_TOLERANCE = 1e-9

TRAINING_OPTIMIZER = "sgd"
TRAINING_LR = 0.05
#: The first leg stops here and the second leg resumes to `RESUMED_STEPS`.
#: `train_distributed_statevector` refuses a run shorter than two steps, because
#: a single step cannot show that the second one continued from the first.
CHECKPOINT_STEPS = 2
RESUMED_STEPS = 4

#: The parameters the trainable circuit binds. Fixed so that the reference and
#: the distributed run are the same function of the same numbers.
TRAINABLE_VALUES = (0.23, -0.37)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _synchronize(device: torch.device) -> None:
    """Wait for the device's work, so a reported failure is the one that happened.

    Every real run is on an accelerator -- the probe refuses to start without
    CUDA -- and this stays device-guarded so that the training legs can be
    exercised on a CPU-only checkout.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_revision() -> str:
    declared = os.environ.get("FLAGQUANTUM_SOURCE_REVISION")
    if declared:
        return declared
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _forward_circuit() -> fq.Circuit:
    return (
        fq.Circuit(N_WIRES, dtype=COMPLEX_DTYPE)
        .h(0)
        .ry(4, 0.31)
        .cx(4, 1)
        .rx(3, -0.27)
        .cx(0, 4)
        .rz(4, 0.19)
        .cx(3, 2)
    )


def _trainable_circuit(
    *,
    device: torch.device,
    dtype: torch.dtype = COMPLEX_DTYPE,
    real_dtype: torch.dtype = REAL_DTYPE,
) -> tuple[fq.Circuit, tuple[torch.Tensor, ...]]:
    """The circuit the training legs differentiate.

    One parameter is bound twice and one gate spans the shard boundary, so a
    gradient that is wrong in its reduction or in its shard ownership cannot
    agree with the reference by accident.
    """

    parameters: list[torch.Tensor] = []
    for value in TRAINABLE_VALUES:
        tensor = torch.empty((), dtype=real_dtype, device=device)
        # Allocation followed by an explicit copy, so an accidental narrowing to
        # float32 is an observable mismatch rather than a silent test of FP32.
        tensor.copy_(torch.tensor(value, dtype=real_dtype))
        parameters.append(tensor.requires_grad_())
    theta, phi = parameters
    circuit = fq.Circuit(N_WIRES, dtype=dtype, device=device)
    circuit.h(0).ry(N_WIRES - 1, theta).rxx(0, N_WIRES - 1, phi).rz(1, theta)
    return circuit, tuple(parameters)


def _observable_wire() -> int:
    # The last logical wire is the least-significant basis bit, which is what
    # `_reference_expectation` below assumes.
    return N_WIRES - 1


def _reference_expectation() -> tuple[float, tuple[float, ...]]:
    """The same expectation and gradient on one CPU device, for the differential."""

    parameters: list[torch.Tensor] = []
    for value in TRAINABLE_VALUES:
        tensor = torch.empty((), dtype=REAL_DTYPE)
        tensor.copy_(torch.tensor(value, dtype=REAL_DTYPE))
        parameters.append(tensor.requires_grad_())
    theta, phi = parameters
    circuit = (
        fq.Circuit(N_WIRES, dtype=COMPLEX_DTYPE)
        .h(0)
        .ry(N_WIRES - 1, theta)
        .rxx(0, N_WIRES - 1, phi)
        .rz(1, theta)
    )
    state = circuit.state(refresh=True)
    probabilities = state.abs().square().reshape(-1)
    expectation = probabilities[::2].sum() - probabilities[1::2].sum()
    expectation.backward()
    return (
        float(expectation.detach()),
        tuple(float(parameter.grad.detach()) for parameter in parameters),
    )


def _validation_state(result: Any) -> torch.Tensor:
    """Materialize a tiny state solely to compare with the CPU reference."""

    local = result.shard_state.amplitudes[0]
    gathered = [torch.empty_like(local) for _ in range(result.plan.world_size)]
    dist.all_gather(gathered, local)
    internal = torch.empty(
        result.plan.total_amplitudes,
        dtype=local.dtype,
        device=local.device,
    )
    for rank, shard in enumerate(gathered):
        internal[rank :: result.plan.world_size] = shard
    mapping = result.logical_to_physical_wires
    canonical = torch.empty_like(internal)
    for logical_basis in range(internal.numel()):
        physical_basis = 0
        for logical_wire, physical_wire in enumerate(mapping):
            bit = (logical_basis >> (len(mapping) - logical_wire - 1)) & 1
            physical_basis |= bit << (len(mapping) - physical_wire - 1)
        canonical[logical_basis] = internal[physical_basis]
    return canonical


def _network_observation(path: Path | None) -> dict[str, Any]:
    configured_interface = os.environ.get("NCCL_SOCKET_IFNAME")
    ib_disabled = os.environ.get("NCCL_IB_DISABLE") == "1"
    observation: dict[str, Any] = {
        "backend": "nccl",
        "configured_interface": configured_interface,
        "ib_disabled": ib_disabled,
        "route": "socket" if ib_disabled else "nccl_auto",
        "evidence_level": "configured_only",
    }
    if path is None:
        return observation
    content = path.read_text(encoding="utf-8", errors="replace")
    socket_observed = "NET/Socket" in content
    interface_observed = bool(configured_interface and configured_interface in content)
    observation.update(
        {
            "debug_log_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "socket_transport_observed": socket_observed,
            "configured_interface_observed": interface_observed,
            "evidence_level": "observed_debug_log",
        }
    )
    if ib_disabled and not (socket_observed and interface_observed):
        raise RuntimeError(
            "NCCL debug log does not confirm the configured socket route"
        )
    return observation


def _prepare_checkpoint_directory(
    directory: Path, *, rank: int, device: torch.device
) -> None:
    """Start from a directory holding no checkpoints, or not at all.

    The resumed leg reads what the first leg wrote, so a checkpoint left by an
    earlier run would be restored instead and the artifact would describe a
    trajectory this run never computed. The caller empties the directory; this
    is the second opinion, because the two must not disagree.

    Rank 0 decides and both ranks act on the same answer. A rank that raised on
    its own would leave its peer waiting in the next collective, which reports
    as a hang rather than as the reason the run was refused.
    """

    verdict = torch.ones((), dtype=torch.int32, device=device)
    if rank == 0:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            existing = sorted(directory.glob("rank-*.pt"))
            if existing:
                verdict.zero_()
        except OSError:
            verdict.zero_()
    dist.broadcast(verdict, src=0)
    if int(verdict.item()) == 0:
        raise RuntimeError(
            f"{directory} could not be used as an empty checkpoint directory; "
            "this probe resumes from what its own first leg wrote"
        )
    dist.barrier()


def _training_observations(
    *,
    device: torch.device,
    checkpoint_directory: Path,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    """Run the legs that make the training claim, and describe them.

    The resumed run and the uninterrupted run are the same computation over the
    same steps. Comparing their losses is what shows the checkpoint restored the
    parameters, the optimizer state and the step counter rather than merely
    reloading a file.

    Each leg gets its own circuit, because a training step writes through to the
    tensors the circuit binds: three legs sharing one circuit would run the
    second and third legs from wherever the first left the parameters, and the
    comparison would then be between two different computations.
    """

    interrupted, _ = _trainable_circuit(device=device)
    first = train_distributed_statevector(
        interrupted,
        steps=CHECKPOINT_STEPS,
        observable_wire=_observable_wire(),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        checkpoint_dir=checkpoint_directory,
        checkpoint_interval=1,
    )
    _synchronize(device)

    uninterrupted, _ = _trainable_circuit(device=device)
    fresh = train_distributed_statevector(
        uninterrupted,
        steps=RESUMED_STEPS,
        observable_wire=_observable_wire(),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        checkpoint_dir=checkpoint_directory / "uninterrupted",
        checkpoint_interval=1,
    )
    _synchronize(device)

    resuming, _ = _trainable_circuit(device=device)
    resumed = train_distributed_statevector(
        resuming,
        steps=RESUMED_STEPS,
        observable_wire=_observable_wire(),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        checkpoint_dir=checkpoint_directory,
        checkpoint_interval=1,
        resume=True,
    )
    _synchronize(device)

    first_summary = first.summary()
    resumed_summary = resumed.summary()
    fresh_summary = fresh.summary()
    expectation = _reference_expectation()[0]
    metrics = {
        "first_leg_initial_expectation_error": abs(first.losses[0] - expectation),
        "resume_prefix_max_abs_error": max(
            (
                abs(actual - expected)
                for actual, expected in zip(
                    first.losses, fresh.losses[:CHECKPOINT_STEPS], strict=True
                )
            ),
            default=0.0,
        ),
        "resume_trajectory_max_abs_error": max(
            (
                abs(actual - expected)
                for actual, expected in zip(
                    resumed.losses, fresh.losses[CHECKPOINT_STEPS:], strict=True
                )
            ),
            default=0.0,
        ),
    }
    observations = {
        "optimizer": TRAINING_OPTIMIZER,
        "learning_rate": TRAINING_LR,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "resumed_steps": RESUMED_STEPS,
        "first_leg_losses": list(first.losses),
        "resumed_leg_losses": list(resumed.losses),
        "uninterrupted_leg_losses": list(fresh.losses),
        "resumed_start_step": resumed.start_step,
        "uninterrupted_start_step": fresh.start_step,
        "checkpoint_files": list(resumed.checkpoint_files),
        "first_leg_summary": first_summary,
        "resumed_leg_summary": resumed_summary,
        "uninterrupted_leg_summary": fresh_summary,
    }
    return observations, metrics, first_summary


def _artifact(
    *,
    rank_records: list[dict[str, Any]],
    metrics: dict[str, float],
    network: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        "schema": "flagquantum.cuda_multinode_statevector_probe.v2",
        "status": "passed",
        "environment": {
            "source_revision": _git_revision(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "nccl": ".".join(str(item) for item in torch.cuda.nccl.version()),
        },
        "scope": {
            "dtype": "complex128",
            "n_wires": N_WIRES,
            "world_size": 2,
            "local_world_size": 1,
            "node_count": 2,
            "distribution_semantics": "sharded_across_ranks",
            "execution": "forward_backward_optimizer_and_checkpoint_resume",
        },
        "observations": {
            "numerical_metrics": metrics,
            "training": training,
            "rank_records": rank_records,
            "network": network,
            "validation_full_state_materialization": True,
            "production_full_state_materialization": False,
        },
        "claim_blockers": [
            "validation_only_tiny_full_state_gather",
            "hidden_host_staging_not_audited",
            "rdma_not_tested",
            "production_performance_not_measured",
            "two_node_pair_only_no_wider_topology",
            "toy_circuit_parameters_only",
        ],
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = _canonical_sha256(evidence)
    return {
        "schema": "flagquantum.cuda_multinode_statevector_artifact.v1",
        "evidence_sha256": digest,
        "evidence": evidence,
    }


def probe(
    *, network_log: Path | None, checkpoint_directory: Path
) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("probe requires exactly two torchrun ranks")
    if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1:
        raise RuntimeError("probe requires exactly one rank per node")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    payload: dict[str, Any] | None = None
    try:
        _prepare_checkpoint_directory(checkpoint_directory, rank=rank, device=device)
        circuit = _forward_circuit()
        result = execute_torch_distributed_statevector(
            circuit,
            device=device,
            dtype=COMPLEX_DTYPE,
            local_world_size=1,
        )
        _synchronize(device)
        if result.shard_state.amplitudes.dtype != COMPLEX_DTYPE:
            raise RuntimeError("distributed shard lost complex128 precision")
        if result.shard_state.amplitudes.device.type != "cuda":
            raise RuntimeError("distributed shard left CUDA")
        if result.inter_node_communication_count < 1:
            raise RuntimeError("workload did not exercise inter-node communication")

        candidate = _validation_state(result)
        reference = _forward_circuit().state()[0].to(device=device)
        metrics = {
            "statevector_max_abs_error": float(
                torch.max(torch.abs(candidate - reference)).item()
            ),
            "statevector_norm_error": float(
                torch.abs(torch.linalg.vector_norm(candidate) - 1).item()
            ),
            "statevector_infidelity": max(
                0.0,
                float(1 - torch.abs(torch.vdot(reference, candidate)).square().item()),
            ),
        }
        if any(value > NUMERICAL_TOLERANCE for value in metrics.values()):
            raise RuntimeError(f"distributed numerical validation failed: {metrics}")

        trainable, parameters = _trainable_circuit(device=device)
        gradient_probe = execute_torch_distributed_statevector_reverse(
            trainable,
            observable_wire=_observable_wire(),
            checkpoint_policy=StatevectorCheckpointPolicy(
                strategy="interval", interval=2
            ),
            device=device,
        )
        gradient_probe.backward()
        _synchronize(device)
        reference_expectation, reference_gradients = _reference_expectation()
        actual_gradients = tuple(
            float(parameter.grad.detach()) for parameter in parameters
        )
        if gradient_probe.local_world_size != 1 or gradient_probe.node_count != 2:
            raise RuntimeError(
                "the backward probe did not resolve the two-node placement: "
                f"local_world_size={gradient_probe.local_world_size} "
                f"node_count={gradient_probe.node_count}"
            )
        metrics.update(
            {
                "expectation_value_error": abs(
                    float(gradient_probe.value.detach()) - reference_expectation
                ),
                "gradient_max_abs_error": max(
                    abs(actual - expected)
                    for actual, expected in zip(
                        actual_gradients, reference_gradients, strict=True
                    )
                ),
            }
        )
        if any(value > NUMERICAL_TOLERANCE for value in metrics.values()):
            raise RuntimeError(f"distributed gradient validation failed: {metrics}")

        training, training_metrics, training_summary = _training_observations(
            device=device, checkpoint_directory=checkpoint_directory
        )
        metrics.update(training_metrics)
        if any(value > TRAINING_TOLERANCE for value in training_metrics.values()):
            raise RuntimeError(
                f"distributed training validation failed: {training_metrics}"
            )
        if training_summary["local_world_size"] != 1:
            raise RuntimeError(
                "the training leg did not resolve the two-node placement: "
                f"local_world_size={training_summary['local_world_size']}"
            )
        if training_summary["node_count"] != 2:
            raise RuntimeError(
                "the training leg did not resolve the two-node placement: "
                f"node_count={training_summary['node_count']}"
            )
        if float(training_summary["losses"][-1]) == float(
            training_summary["losses"][0]
        ):
            # A frozen expectation would make every comparison below agree for
            # the wrong reason. Whether it moved towards or away from the
            # minimum is recorded rather than asserted: SGD on this observable
            # takes a legitimate step in either direction at this learning rate,
            # and the claim under test is that the checkpoint reproduces the
            # trajectory, not that the learning rate was well chosen.
            raise RuntimeError(
                "the optimizer step did not change the expectation: "
                f"{training_summary['losses']}"
            )
        metrics["training_loss_decrease"] = float(
            training_summary["losses"][0] - training_summary["losses"][-1]
        )
        if int(training["resumed_start_step"]) != CHECKPOINT_STEPS:
            raise RuntimeError(
                "the resumed leg did not continue from the checkpoint: "
                f"start_step={training['resumed_start_step']}"
            )

        properties = torch.cuda.get_device_properties(device)
        record = result.summary()
        record.update(
            {
                "hostname_sha256": hashlib.sha256(
                    platform.node().encode("utf-8")
                ).hexdigest(),
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "")),
                "owned_global_basis_indices": (
                    f"{rank} + k*{result.plan.world_size}, "
                    f"0 <= k < {result.shard_state.shard.local_amplitudes}"
                ),
                "backward": gradient_probe.summary(),
                "training": training_summary,
            }
        )
        rank_records: list[dict[str, Any] | None] = [None, None]
        dist.all_gather_object(rank_records, record)
        if rank == 0:
            payload = _artifact(
                rank_records=[item for item in rank_records if item is not None],
                metrics=metrics,
                network={},
                training=training,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0 and payload is not None:
        network = _network_observation(network_log)
        evidence = payload["evidence"]
        evidence["observations"]["network"] = network
        payload["evidence_sha256"] = _canonical_sha256(evidence)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        required=True,
        help=(
            "shared, empty directory the training legs write rank-local "
            "checkpoints into and resume from"
        ),
    )
    parser.add_argument(
        "--network-log",
        type=Path,
        help="rank-zero NCCL debug log used to verify the selected network route",
    )
    args = parser.parse_args()
    if not shutil.which("git") and not os.environ.get("FLAGQUANTUM_SOURCE_REVISION"):
        # The artifact has to name a revision; a checkout without git can still
        # be pinned by the caller, and otherwise it cannot say what it measured.
        raise SystemExit(
            "FLAGQUANTUM_SOURCE_REVISION is required when the staged tree "
            "carries no git checkout"
        )
    payload = probe(
        network_log=args.network_log,
        checkpoint_directory=args.checkpoint_directory,
    )
    if payload is None:
        return 0
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
