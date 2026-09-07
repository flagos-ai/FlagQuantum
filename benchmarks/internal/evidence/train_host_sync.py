#!/usr/bin/env python3
"""Measure host synchronization in the stable ``fq.train`` optimizer loop."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402


class _OrchestrationProbe(fq.Module):
    def __init__(self, *, parameters: int, device: torch.device) -> None:
        super().__init__(
            lambda values: fq.Circuit(1).ry(0, values[0]),
            n_parameters=parameters,
            init="normal",
            seed=17,
            device=device,
        )

    def execute(self, inputs=None, parameters=None):
        del inputs, parameters
        values = self.parameters_tensor
        value = (torch.sin(values) * torch.cos(values * 0.5)).mean().reshape(1)
        return fq.ExecutionResult(value=value)


def _quantum_module(*, wires: int, parameters: int, device: torch.device) -> fq.Module:
    def build(values):
        circuit = fq.Circuit(wires)
        for index in range(parameters):
            wire = index % wires
            circuit.ry(wire, values[index])
            circuit.cx(wire, (wire + 1) % wires)
        return circuit

    return fq.Module(
        build,
        n_parameters=parameters,
        init="normal",
        seed=17,
        device=device,
    )


def _training_pair(args, device: torch.device):
    module = (
        _quantum_module(
            wires=args.wires,
            parameters=args.parameters,
            device=device,
        )
        if args.workload == "quantum_statevector"
        else _OrchestrationProbe(parameters=args.parameters, device=device)
    )
    return module, torch.optim.SGD(module.parameters(), lr=0.01)


def _objective(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean()


def _legacy_train(module, optimizer, steps: int) -> None:
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        execution = module.execute()
        loss = _objective(execution.require_value())
        loss.backward()
        optimizer.step()
        float(loss.detach())
        execution.detach()


def _run(args, device: torch.device, mode: str, steps: int) -> float:
    module, optimizer = _training_pair(args, device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if mode == "silent":
        fq.train(module, optimizer=optimizer, objective=_objective, steps=steps)
    elif mode == "callback":
        fq.train(
            module,
            optimizer=optimizer,
            objective=_objective,
            steps=steps,
            callback=lambda *_: None,
        )
    else:
        _legacy_train(module, optimizer, steps)
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _scalar_materializations(args, device: torch.device, mode: str) -> int:
    module, optimizer = _training_pair(args, device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        if mode == "silent":
            fq.train(module, optimizer=optimizer, objective=_objective, steps=10)
        elif mode == "callback":
            fq.train(
                module,
                optimizer=optimizer,
                objective=_objective,
                steps=10,
                callback=lambda *_: None,
            )
        else:
            _legacy_train(module, optimizer, 10)
    return sum(
        event.count
        for event in p.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=("orchestration", "quantum_statevector"),
        default="orchestration",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wires", type=int, default=12)
    parser.add_argument("--parameters", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--host-label", default=socket.gethostname())
    parser.add_argument("--container-label", default="unknown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.parameters, args.steps, args.warmup_steps, args.repeats) <= 0:
        raise SystemExit(
            "parameters, steps, warmup-steps, and repeats must be positive"
        )
    if args.workload == "quantum_statevector" and args.wires <= 1:
        raise SystemExit("quantum_statevector requires at least two wires")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this benchmark requires a CUDA device")
    modes = ("silent", "callback", "legacy")
    for mode in modes:
        _run(args, device, mode, args.warmup_steps)
    samples = {mode: [] for mode in modes}
    for repeat in range(args.repeats):
        order = modes if repeat % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            samples[mode].append(_run(args, device, mode, args.steps))

    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    result = {
        "schema": "flagquantum.train_host_sync",
        "version": "1.0",
        "host": args.host_label,
        "container": args.container_label,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "workload": args.workload,
        "wires": args.wires if args.workload == "quantum_statevector" else None,
        "parameters": args.parameters,
        "steps": args.steps,
        "repeats": args.repeats,
        "seconds": samples,
        "median_seconds": medians,
        "steps_per_second": {
            mode: args.steps / duration for mode, duration in medians.items()
        },
        "silent_speedup_over_legacy": medians["legacy"] / medians["silent"],
        "silent_speedup_over_callback": medians["callback"] / medians["silent"],
        "scalar_materializations_per_10_steps": {
            mode: _scalar_materializations(args, device, mode) for mode in modes
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
