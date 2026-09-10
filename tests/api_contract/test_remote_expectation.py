"""Remote Pauli expectations preserve the root result contract and evidence."""

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.testing import InMemoryRemoteTarget


class _QuafuTestTarget(InMemoryRemoteTarget):
    provider = "quafu"


def test_fq_run_remote_expectation_compiles_once_and_groups_measurements(
    monkeypatch,
) -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    calls = 0
    provider = _QuafuTestTarget()

    def compile_target(program, *, compiler, target, target_qubits=None):
        nonlocal calls
        calls += 1
        assert compiler == "qsteed"
        assert target == "quafu:ScQ-P10"
        assert target_qubits == (3, 4)
        return replace(
            program.to_ir(),
            metadata={
                "execution_target": {
                    "provider": "quafu",
                    "backend": "ScQ-P10",
                    "compiler": None,
                    "target_qubits": (3, 4),
                }
            },
        )

    monkeypatch.setattr("flagquantum._api.compile", compile_target)
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.QuafuProvider", lambda: provider
    )

    result = fq.run(
        circuit,
        outputs=fq.expectation(
            0.5 * (fq.X(0) @ fq.X(1)) - 0.25 * (fq.Y(0) @ fq.Y(1)) + 0.1 * fq.I(),
            name="energy",
        ),
        compiler="qsteed",
        target="quafu:ScQ-P10",
        target_qubits=(3, 4),
        shots=128,
        name="bell energy",
    )

    assert calls == 1
    torch.testing.assert_close(result.expectation("energy"), torch.tensor([0.85]))
    measurement = result.measurement("energy")
    assert measurement.statistics == {
        "standard_error": 0.0,
        "group_count": 2,
        "shots_per_group": (128, 128),
        "total_shots": 256,
    }
    assert result.runtime == {
        "mode": "remote_qpu",
        "shots": 256,
        "shots_per_group": (128, 128),
    }
    assert result.provenance["target_qubits"] == (3, 4)
    assert result.provenance["task_ids"] == ("local-1", "local-2")
    assert len(result.provenance["measurement_groups"]) == 2
    assert len(result.native()) == 2
    assert len(provider._packages) == 2
    for package in provider._packages.values():
        assert package.metadata["provider_options"] == {
            "compiler": None,
            "target_qubits": [3, 4],
        }
        assert package.metadata["deployment_artifact_sha256"]


def test_fq_run_remote_rejects_mixed_outputs_before_compilation(monkeypatch) -> None:
    circuit = fq.Circuit(1)

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("unsupported outputs must fail before compilation")

    monkeypatch.setattr("flagquantum._api.compile", unexpected_compile)

    with pytest.raises(ValueError, match="one full-register.*or one fq.expectation"):
        fq.run(
            circuit,
            outputs=(fq.expectation(fq.Z(0)), fq.counts()),
            compiler="qsteed",
            target="quafu:ScQ-P10",
            shots=1024,
        )
