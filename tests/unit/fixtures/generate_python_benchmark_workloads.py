# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Observe every complete frozen Python workload without replacing its body."""

import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import kickmix as km


def _load(name: str, source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, source / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    reference = Path(__file__).resolve().parents[4] / "ftqc-reference-env"
    source = reference / "kickmix/src/kickmix/py/perf"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        expected = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    blobs = {}
    for path in sorted(source.glob("*.py")):
        relative = str(path.relative_to(reference))
        raw = path.read_bytes()
        digest = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        assert digest == expected[relative]
        blobs[relative] = digest
    assert len(blobs) == 4
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert (
        hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest()
        == "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    suite = _load("benchmark_suite", source)
    array = _load("array_perf", source)
    builder = _load("circuit_builder_perf", source)
    runner = _load("run_all_perf", source)
    observations = []
    for module in (array, builder):
        for method in module.benchmark.benchmarks:
            for repeat in (1, 3):
                observation: dict[str, Any] = {"repeat": repeat}

                def capture(
                    self: Any,
                    body: Any,
                    *,
                    rates: Any = None,
                    repeats: int = repeat,
                    destination: dict[str, Any] = observation,
                ) -> Any:
                    for _ in range(repeats):
                        body()
                    cells = inspect.getclosurevars(body).nonlocals
                    destination["rates"] = rates
                    if "builder" in cells:
                        circuit = cells["builder"].finish_circuit()
                        destination["text"] = str(circuit)
                        destination["allocated"] = cells["builder"].num_allocated_qubits
                    else:
                        destination["items"] = [str(item) for item in cells["items"]]
                        destination["constructed"] = [
                            str(item) for item in km.array(cells["items"])
                        ]
                    return suite._BenchmarkResult(
                        name=None,
                        total_seconds=0.5,
                        total_reps=repeats,
                        rates=tuple((rates or {}).items()),
                        goal_seconds=None,
                    )

                with patch.object(suite.BenchmarkSuite, "go", capture):
                    result = method()
                observation.update(
                    name=result.name, goal=result.goal_seconds, formatted=str(result)
                )
                observations.append(observation)
    discovered = []
    with patch.object(
        suite.BenchmarkSuite,
        "main",
        lambda self: discovered.append([self.name, len(self.benchmarks)]),
    ):
        runner.main()
    executions = []
    for observation in observations:
        if "text" not in observation:
            continue
        circuit = km.Circuit(observation["text"])
        simulator = km.Simulator(
            batch_size=3, seed=0, debug_print=False, counting_operations=True
        )
        simulator.use_same_registers_as(circuit)
        quantum = (
            512
            if "iadd" in observation["name"]
            else (3 if "qqq_1" in observation["name"] else 768)
        )
        for wire in range(quantum):
            simulator.write_across_shots(km.q(wire), (wire * 5 + 3) & 7)
        if "mixed" in observation["name"]:
            for bit in range(256, 512):
                simulator.write_across_shots(km.b(bit), (bit * 3 + 1) & 7)
        simulator.do(circuit)
        executions.append(
            {
                "name": observation["name"],
                "repeat": observation["repeat"],
                "values": simulator.read_across_shots(
                    [km.q(wire) for wire in range(quantum)], out=int
                ),
                "work": (
                    simulator.read_across_shots(
                        [km.q(wire) for wire in range(512, 766)], out=int
                    )
                    if "iadd" in observation["name"]
                    else []
                ),
                "phases": [str(simulator.read_shot_phase(shot)) for shot in range(3)],
                "counts": dict(simulator.operation_statistics()),
            }
        )
    output = {
        "frozen_blobs": blobs,
        "workloads": observations,
        "discovered": discovered,
        "executions": executions,
    }
    destination = (
        Path(__file__).resolve().parents[1] / "data/python_benchmark_workloads.json"
    )
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
