"""Run frozen CPU training tasks with independent parity and recovery checks.

Examples:
    python -m benchmarks.training_comparison --smoke --output /tmp/training-smoke
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.training_engines import (
    Engine,
    Expectation,
    build_expectation,
    engine_metadata,
)
from benchmarks.training_workloads import (
    Split,
    Task,
    binary_loss,
    classical_logits,
    dataset,
    dense_loss,
    finite_difference,
    initial_parameters,
    pauli_matrix,
    terms,
)
from flagquantum.benchmarking.contract import runtime_metadata, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/manifests/training_comparison_cpu_v1.json"
JAX_MANIFEST = ROOT / "benchmarks/manifests/training_comparison_jax_cpu_v1.json"
INSTALLED_MANIFEST = (
    ROOT / "benchmarks/manifests/training_comparison_installed_cpu_v1.json"
)
SCHEMA = "flagquantum.training_comparison.v1"
RUNNER = "training_comparison"


@dataclass(frozen=True)
class Trace:
    step: int
    training_loss_pre_update: float
    validation_score: float
    update_seconds: float
    validation_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class Parity:
    value_error: float
    gradient_max_absolute_error: float
    reference_gradient: list[float]
    observed_gradient: list[float]
    passed: bool
    first_value_grad_seconds: float


@dataclass(frozen=True)
class Assessment:
    train_loss: float
    validation_score: float
    test_accuracy: float | None
    final_relative_energy_error: float | None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, timeout=15
    ).strip()


def source_identity(protocol: Path = MANIFEST) -> dict[str, Any]:
    """Bind the frozen protocol and all producing benchmark sources."""
    paths = [
        Path(__file__),
        ROOT / "benchmarks/training_workloads.py",
        ROOT / "benchmarks/training_engines.py",
    ]
    from flagquantum.simulation.native_cpu.adjoint import native_cpu_adjoint_available

    extension = find_spec("flagquantum.simulation.native_cpu._C")
    return {
        "code_version": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=normal")),
        "protocol_sha256": digest(protocol),
        "producer_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "installation": {
            "distribution_version": (
                metadata.version("flagquantum") if _installed("flagquantum") else None
            ),
            "native_cpu_adjoint_available": native_cpu_adjoint_available(),
            "native_cpu_extension_sha256": (
                digest(Path(extension.origin))
                if extension is not None and extension.origin is not None
                else None
            ),
        },
        "versions": {
            p: metadata.version(p)
            for p in (
                "torch",
                "numpy",
                "pennylane",
                "pennylane-lightning",
                "tensorcircuit-ng",
                "jax",
                "jaxlib",
            )
            if _installed(p)
        },
    }


def _installed(package: str) -> bool:
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        return False
    return True


def logits(
    parameters: torch.Tensor, split: Split, expectation: Expectation | None
) -> torch.Tensor:
    if expectation is None:
        return classical_logits(parameters, split.features)
    measured = torch.stack([expectation(parameters, row) for row in split.features])
    return parameters[4] * measured + parameters[5]


def objective(
    task: Task, parameters: torch.Tensor, split: Split, expectation: Expectation | None
) -> torch.Tensor:
    if task == "classifier":
        return binary_loss(logits(parameters, split, expectation), split)
    if expectation is None:
        raise ValueError("Heisenberg objective requires a quantum engine")
    return expectation(parameters, torch.zeros(2, dtype=torch.float64))


def ground_energy() -> float:
    matrix = sum(
        c * pauli_matrix(word, wires, 4) for c, word, wires in terms("heisenberg")
    )
    return float(np.linalg.eigvalsh(matrix)[0])


def score(
    task: Task, parameters: torch.Tensor, split: Split, expectation: Expectation | None
) -> float:
    with torch.no_grad():
        if task == "classifier":
            predictions = logits(parameters, split, expectation) > 0
            return float((predictions == split.labels.bool()).to(torch.float64).mean())
        exact = ground_energy()
        return abs(
            float(objective(task, parameters, split, expectation)) - exact
        ) / abs(exact)


def check_parity(
    task: Task, parameters: torch.Tensor, split: Split, expectation: Expectation | None
) -> Parity:
    probe = Split(split.features[:4], split.labels[:4])
    started = time.perf_counter()
    value = objective(task, parameters, probe, expectation)
    gradient = torch.autograd.grad(value, parameters)[0]
    first_seconds = time.perf_counter() - started
    if expectation is None:
        x, y = probe.features.unbind(1)
        basis = torch.stack(
            (torch.ones_like(x), x, y, x.square(), x * y, y.square()), 1
        )
        reference_gradient = (
            basis.T
            @ (torch.sigmoid(basis @ parameters.detach()) - probe.labels)
            / len(x)
        ).numpy()
        reference_value = float(binary_loss(basis @ parameters.detach(), probe))
    else:
        reference_value = dense_loss(task, parameters.detach().numpy(), probe)
        reference_gradient = finite_difference(task, parameters, probe)
    value_error = abs(float(value.detach()) - reference_value)
    gradient_error = float(
        np.max(np.abs(gradient.detach().numpy() - reference_gradient))
    )
    return Parity(
        value_error,
        gradient_error,
        reference_gradient.tolist(),
        gradient.detach().tolist(),
        value_error <= 1e-9 and gradient_error <= 2e-6,
        first_seconds,
    )


def _optimizer(parameters: torch.Tensor, task: Task) -> torch.optim.Adam:
    return torch.optim.Adam(
        [parameters],
        lr=0.12 if task == "classifier" else 0.03,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )


def train_case(
    task: Task,
    engine: Engine,
    seed: int,
    *,
    smoke: bool,
    directory: Path,
    protocol: Path = MANIFEST,
) -> dict[str, Any]:
    """Run one isolated case; hold out test data until fixed-step training ends."""
    if task == "heisenberg" and engine == "classical":
        raise ValueError("classical control applies only to the classifier")
    train, validation, test = dataset()
    if smoke:
        train, validation, test = (
            Split(s.features[:size], s.labels[:size])
            for s, size in zip((train, validation, test), (8, 4, 4), strict=True)
        )
    steps, checkpoint_step = (3, 1) if smoke else (60, 30)
    torch.set_num_threads(1)
    parameters = initial_parameters(task, seed, classical=engine == "classical")
    setup = time.perf_counter()
    expectation = None if engine == "classical" else build_expectation(engine, task)
    setup_seconds = time.perf_counter() - setup
    parity = check_parity(task, parameters, train, expectation)
    if not parity.passed:
        raise ValueError(f"independent value/gradient parity failed: {asdict(parity)}")
    warmup = time.perf_counter()
    torch.autograd.grad(objective(task, parameters, train, expectation), parameters)
    warmup_seconds = time.perf_counter() - warmup
    optimizer = _optimizer(parameters, task)
    checkpoint: dict[str, Any] | None = None
    trace: list[Trace] = []
    target_time: float | None = None
    initial_score = score(task, parameters, validation, expectation)

    def target(value: float) -> bool:
        return value >= 0.8 if task == "classifier" else value <= 0.01

    if target(initial_score):
        target_time = 0.0
    checkpoint_seconds = 0.0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        update_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(task, parameters, train, expectation)
        loss.backward()
        optimizer.step()
        update_seconds = time.perf_counter() - update_start
        validation_start = time.perf_counter()
        validation_score = score(task, parameters, validation, expectation)
        validation_seconds = time.perf_counter() - validation_start
        elapsed = time.perf_counter() - started
        if target_time is None and target(validation_score):
            target_time = elapsed
        trace.append(
            Trace(
                step,
                float(loss.detach()),
                validation_score,
                update_seconds,
                validation_seconds,
                elapsed,
            )
        )
        if step == checkpoint_step:
            save_start = time.perf_counter()
            checkpoint = {
                "parameters": parameters.detach().clone(),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "step": step,
            }
            torch.save(checkpoint, directory / "checkpoint.pt")
            checkpoint_seconds = time.perf_counter() - save_start
        if step == 1 or step % 10 == 0 or step == steps:
            print(
                json.dumps(
                    {
                        "task": task,
                        "engine": engine,
                        "seed": seed,
                        "step": step,
                        "score": validation_score,
                    }
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    assessment_start = time.perf_counter()
    with torch.no_grad():
        assessment = Assessment(
            float(objective(task, parameters, train, expectation)),
            score(task, parameters, validation, expectation),
            (
                score(task, parameters, test, expectation)
                if task == "classifier"
                else None
            ),
            (
                score(task, parameters, validation, expectation)
                if task == "heisenberg"
                else None
            ),
        )
    assessment_seconds = time.perf_counter() - assessment_start
    # Replay is outside measured training and never selects a checkpoint by quality.
    if checkpoint is None:
        raise RuntimeError("checkpoint was not produced")
    restored = torch.load(directory / "checkpoint.pt", weights_only=True)
    resumed = restored["parameters"].clone().requires_grad_()
    resumed_optimizer = _optimizer(resumed, task)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    for _ in range(restored["step"], steps):
        resumed_optimizer.zero_grad(set_to_none=True)
        objective(task, resumed, train, expectation).backward()
        resumed_optimizer.step()
    checkpoint_error = float((resumed.detach() - parameters.detach()).abs().max())
    checkpoint_gradient_error = float((resumed.grad - parameters.grad).abs().max())
    expected_state = optimizer.state_dict()["state"][0]
    actual_state = resumed_optimizer.state_dict()["state"][0]
    checkpoint_optimizer_error = max(
        float((value - actual_state[key]).abs().max())
        for key, value in expected_state.items()
        if isinstance(value, torch.Tensor)
    )
    if (
        max(checkpoint_error, checkpoint_gradient_error, checkpoint_optimizer_error)
        > 1e-10
    ):
        raise ValueError(
            "checkpoint replay changed parameters, gradients or Adam state"
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return runtime_metadata(
        runner=RUNNER,
        schema=SCHEMA,
        status="completed",
        task=task,
        engine=engine_metadata(engine),
        seed=seed,
        smoke=smoke,
        benchmark_evidence_class="comparison_non_release",
        execution_semantics="single_device_fast_path",
        scalability_claim_allowed=False,
        protocol=json.loads(protocol.read_text()),
        source=source_identity(protocol),
        parity=asdict(parity),
        initial_validation_score=initial_score,
        assessment=asdict(assessment),
        time_to_target_seconds=target_time,
        completed_steps=steps,
        trace=[asdict(point) for point in trace],
        backend_setup_seconds=setup_seconds,
        warmup_seconds=warmup_seconds,
        training_seconds=training_seconds,
        checkpoint_write_seconds=checkpoint_seconds,
        assessment_seconds=assessment_seconds,
        total_measured_seconds=setup_seconds
        + parity.first_value_grad_seconds
        + warmup_seconds
        + training_seconds
        + assessment_seconds,
        checkpoint_replay_max_parameter_error=checkpoint_error,
        checkpoint_replay_max_gradient_error=checkpoint_gradient_error,
        checkpoint_replay_max_optimizer_error=checkpoint_optimizer_error,
        checkpoint_sha256=digest(directory / "checkpoint.pt"),
        process_peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
        final_parameters=parameters.detach().tolist(),
        data_sha256=hashlib.sha256(
            b"".join(
                s.features.numpy().tobytes() + s.labels.numpy().tobytes()
                for s in (train, validation, test)
            )
        ).hexdigest(),
    )


def campaign(output: Path, *, smoke: bool, protocol: Path = MANIFEST) -> int:
    """Give each measured case a fresh process and an explicit wall deadline."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(protocol.read_text())
    if (
        protocol == INSTALLED_MANIFEST
        and not source_identity(protocol)["installation"][
            "native_cpu_adjoint_available"
        ]
    ):
        raise RuntimeError(
            "installed CPU diagnostic requires a loadable native extension"
        )
    frozen_data = dataset()
    freeze = runtime_metadata(
        runner=RUNNER,
        schema=SCHEMA,
        source=source_identity(protocol),
        protocol=manifest,
        smoke=smoke,
        data_sha256=hashlib.sha256(
            b"".join(
                s.features.numpy().tobytes() + s.labels.numpy().tobytes()
                for s in frozen_data
            )
        ).hexdigest(),
        initial_parameters={
            task: {
                str(seed): initial_parameters(task, seed).detach().tolist()
                for seed in manifest["seeds"]
            }
            for task in ("classifier", "heisenberg")
        },
        classical_initial_parameters=initial_parameters(
            "classifier", 11, classical=True
        )
        .detach()
        .tolist(),
        scalability_claim_allowed=False,
    )
    write_json_atomic(output / "freeze.json", freeze)
    seeds = [11] if smoke else manifest["seeds"]
    cases: list[dict[str, Any]] = []
    for task in ("classifier", "heisenberg"):
        engines = manifest["engines"] + (
            ["classical"]
            if task == "classifier" and manifest.get("include_classical_control", True)
            else []
        )
        for engine in engines:
            for seed in seeds:
                directory = output / f"{task}-{engine}-{seed}"
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.training_comparison",
                    "--worker",
                    "--task",
                    task,
                    "--engine",
                    engine,
                    "--seed",
                    str(seed),
                    "--output",
                    str(directory),
                ]
                if smoke:
                    command.append("--smoke")
                if protocol == JAX_MANIFEST:
                    command.append("--jax-diagnostic")
                if protocol == INSTALLED_MANIFEST:
                    command.append("--installed-cpu-diagnostic")
                print(f"Starting {directory.name}", flush=True)
                status = "failed"
                environment = dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    JAX_PLATFORMS="cpu",
                )
                with (output / f"{directory.name}.log").open("w") as log:
                    try:
                        result = subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=manifest["case_timeout_seconds"],
                            check=False,
                            env=environment,
                        )
                        status = "completed" if result.returncode == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                cases.append(
                    {
                        "case": directory.name,
                        "status": status,
                        "result": str(directory.relative_to(output) / "result.json"),
                    }
                )
                write_json_atomic(
                    output / "summary.json",
                    runtime_metadata(
                        runner=RUNNER,
                        schema=SCHEMA,
                        smoke=smoke,
                        source=source_identity(protocol),
                        cases=cases,
                        scalability_claim_allowed=False,
                        benchmark_evidence_class="comparison_non_release",
                    ),
                )
    return int(any(case["status"] != "completed" for case in cases))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    strategy = parser.add_mutually_exclusive_group()
    strategy.add_argument(
        "--jax-diagnostic",
        action="store_true",
        help="run the separately frozen JAX execution-strategy comparison",
    )
    strategy.add_argument(
        "--installed-cpu-diagnostic",
        action="store_true",
        help="compare installed public Module defaults and JAX routes",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--task", choices=("classifier", "heisenberg"), default="classifier"
    )
    parser.add_argument(
        "--engine",
        choices=(
            "flagquantum",
            "flagquantum_torch_module",
            "flagquantum_jax_module",
            "flagquantum_jax_kernel",
            "pennylane_backprop",
            "pennylane_adjoint",
            "tensorcircuit_torch",
            "tensorcircuit_jax",
            "classical",
        ),
        default="flagquantum",
    )
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    protocol = (
        INSTALLED_MANIFEST
        if args.installed_cpu_diagnostic
        else JAX_MANIFEST if args.jax_diagnostic else MANIFEST
    )
    if not args.worker:
        return campaign(args.output, smoke=args.smoke, protocol=protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.float64)
    torch.set_num_interop_threads(1)
    if args.seed not in (11, 29, 47):
        parser.error("the frozen protocol permits seeds 11, 29 and 47 only")
    try:
        payload = train_case(
            args.task,
            args.engine,
            args.seed,
            smoke=args.smoke,
            directory=args.output,
            protocol=protocol,
        )
    except (
        Exception
    ) as exc:  # CLI process boundary: retain failed evidence and traceback.
        traceback.print_exc()
        write_json_atomic(
            args.output / "result.json",
            runtime_metadata(
                runner=RUNNER,
                schema=SCHEMA,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                source=source_identity(protocol),
                scalability_claim_allowed=False,
            ),
        )
        return 1
    write_json_atomic(args.output / "result.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
