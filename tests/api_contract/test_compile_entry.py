"""User-facing compiler selection stays explicit and fail-closed."""

import pytest

import flagquantum as fq
from flagquantum.remote import DeploymentResult, ProviderTaskHandle


def test_fq_compile_uses_builtin_compiler_by_default():
    source = fq.Circuit(2).h(0).cx(0, 1)

    compiled = fq.compile(source)

    assert isinstance(compiled, fq.CircuitIR)
    assert compiled.n_wires == 2


def test_fq_compile_resolves_quafu_target_for_named_plugin(monkeypatch):
    source = fq.Circuit(2).h(0).cx(0, 1)
    chip_info = {
        "calibration_time": "2026-09-09 12:00:00",
        "qubits_info": {"Q3": {}, "Q4": {}},
        "couplers_info": {"C0": {"qubits_index": [3, 4], "fidelity": 0.99}},
    }
    captured = {}

    class Provider:
        def fetch_chip_info(self, backend):
            assert backend == "ScQ-P10"
            return chip_info

    def compile_with_extension(program, *, extension, target):
        captured.update(extension=extension, target=target)
        return program

    monkeypatch.setattr("flagquantum.remote.QuafuProvider", Provider)
    monkeypatch.setattr(
        "flagquantum.ecosystem.extensions.compile_with_extension",
        compile_with_extension,
    )

    compiled = fq.compile(source, compiler="qsteed", target="quafu:ScQ-P10")

    assert compiled == source.to_ir()
    assert captured == {
        "extension": "qsteed",
        "target": {
            "provider": "quafu",
            "backend": "ScQ-P10",
            "chip_info": chip_info,
        },
    }

    fq.compile(
        source,
        compiler="qsteed",
        target="quafu:ScQ-P10",
        target_qubits=(4, 3),
    )
    assert captured["target"]["target_qubits"] == (4, 3)


def test_fq_run_compiles_packages_and_executes_one_remote_target(monkeypatch):
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    compiled = circuit.to_ir()
    captured = {}

    class Provider:
        pass

    def compile_target(program, *, compiler, target, target_qubits=None):
        captured.update(
            program=program,
            compiler=compiler,
            target=target,
            target_qubits=target_qubits,
        )
        return compiled

    def deploy_target(program, provider, *, shots, name=None):
        captured.update(compiled=program, provider=provider, shots=shots, name=name)
        handle = ProviderTaskHandle("quafu", "task-17", "ScQ-P10")
        return DeploymentResult(
            handle=handle,
            counts={"00": 513, "11": 511},
            shots=1024,
            metadata={"deployment_artifact_sha256": "artifact-17"},
        )

    monkeypatch.setattr("flagquantum._api.compile", compile_target)
    monkeypatch.setattr("flagquantum.remote.QuafuProvider", Provider)
    monkeypatch.setattr("flagquantum.deployment.deploy_circuit", deploy_target)

    result = fq.run(
        circuit,
        compiler="qsteed",
        target="quafu:ScQ-P10",
        shots=1024,
    )

    assert isinstance(result, fq.ExecutionResult)
    assert result.measurement("counts").value == [{"00": 513, "11": 511}]
    assert result.provenance["task_id"] == "task-17"
    assert result.native().handle.task_id == "task-17"
    assert captured == {
        "program": circuit,
        "compiler": "qsteed",
        "target": "quafu:ScQ-P10",
        "target_qubits": None,
        "compiled": compiled,
        "provider": captured["provider"],
        "shots": 1024,
        "name": None,
    }
    assert isinstance(captured["provider"], Provider)

    fq.run(
        circuit,
        compiler="qsteed",
        target="quafu:ScQ-P10",
        shots=1024,
        name="  bell calibration  ",
        target_qubits=(4, 3),
    )
    assert captured["name"] == "bell calibration"
    assert captured["target_qubits"] == (4, 3)


def test_fq_run_remote_controls_fail_closed_when_incomplete():
    circuit = fq.Circuit(1).x(0)

    with pytest.raises(ValueError, match="requires a compiler target"):
        fq.compile(circuit, compiler="qsteed", target_qubits=(0,))
    with pytest.raises(TypeError, match="both compiler and target"):
        fq.run(circuit, compiler="qsteed", shots=1024)
    with pytest.raises(ValueError, match="positive integer"):
        fq.run(
            circuit,
            compiler="qsteed",
            target="quafu:ScQ-P10",
            shots=0,
        )
    with pytest.raises(ValueError, match="non-empty string"):
        fq.run(
            circuit,
            compiler="qsteed",
            target="quafu:ScQ-P10",
            shots=1024,
            name="  ",
        )


def test_fq_run_uses_resident_jiuding_workspace(monkeypatch):
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    expected = object()
    captured = {}

    def run(program, *, target, outputs=None, shots=None):
        captured.update(program=program, target=target, outputs=outputs, shots=shots)
        return expected

    monkeypatch.setattr("flagquantum.remote.compute.jiuding.run", run)

    output = fq.expectation(fq.Z(0))
    result = fq.run(circuit, target="jiuding:gpu", outputs=output)

    assert result is expected
    assert captured == {
        "program": circuit,
        "target": "jiuding:gpu",
        "outputs": output,
        "shots": None,
    }


def test_fq_run_passes_jiuding_sampling_controls(monkeypatch):
    expected = object()
    captured = {}

    def run(program, *, target, outputs=None, shots=None):
        captured.update(target=target, outputs=outputs, shots=shots)
        return expected

    monkeypatch.setattr("flagquantum.remote.compute.jiuding.run", run)
    output = fq.counts()

    result = fq.run(
        fq.Circuit(2).h(0).cx(0, 1),
        target="jiuding:gpu",
        outputs=output,
        shots=1024,
    )

    assert result is expected
    assert captured == {
        "target": "jiuding:gpu",
        "outputs": output,
        "shots": 1024,
    }


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("compiler", "flagquantum", "does not accept compiler"),
        ("target_qubits", (0,), "does not accept target_qubits"),
        ("name", "bell", "does not accept name"),
    ],
)
def test_fq_run_jiuding_rejects_unsupported_controls(keyword, value, message):
    with pytest.raises(TypeError, match=message):
        fq.run(fq.Circuit(1), target="jiuding:gpu", **{keyword: value})
