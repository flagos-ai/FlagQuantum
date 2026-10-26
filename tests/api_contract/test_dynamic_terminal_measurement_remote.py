"""Stable-facade contract for terminal measurements on static Quafu execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import flagquantum as fq
import flagquantum.errors as fqe
from flagquantum.dynamic import DynamicCircuit
from flagquantum.remote.qpu import _quafu_sdk
from flagquantum.remote.qpu.quafu import QuafuProvider

pytestmark = pytest.mark.unit


class _Client:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, dict[str, object]]] = []
        self.job_handle = SimpleNamespace(
            id="sdk-job-terminal",
            n_tasks=1,
            status=lambda: "Finished",
            result=lambda timeout=0: SimpleNamespace(
                status="Finished",
                task_id="sdk-job-terminal-0",
                target="Baihua",
                backend="Baihua",
                bit_order="c0_rightmost",
                n_clbits=2,
                shots_requested=1024,
                shots_returned=1024,
                counts={"01": 1024},
                measurements=None,
                executed_qasm="service circuit",
                compiler={"name": "qiskit", "version": "2.5"},
                calibration_id="cal-1",
            ),
        )

    def submit(self, circuit: object, **options: object) -> SimpleNamespace:
        self.submissions.append((circuit, options))
        return self.job_handle

    def job(self, job_id: str) -> SimpleNamespace:
        assert job_id == "sdk-job-terminal"
        return self.job_handle


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _Client:
    client = _Client()
    adapter = _quafu_sdk._QuafuSdkAdapter.for_submission(client=client)
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.QuafuProvider",
        lambda **options: QuafuProvider(_sdk_adapter=adapter, **options),
    )
    return client


def _terminal_program(*, permuted: bool = False) -> DynamicCircuit:
    circuit = DynamicCircuit(2).x(0)
    if permuted:
        return circuit.measure(1, classical_bit=0).measure(0, classical_bit=1)
    return circuit.measure(0, classical_bit=0).measure(1, classical_bit=1)


def test_terminal_measurements_submit_once_through_the_static_sdk_path(
    client: _Client,
) -> None:
    result = fq.run(
        _terminal_program(permuted=True),
        target="quafu:Baihua",
        shots=1024,
    )

    assert len(client.submissions) == 1
    qasm, options = client.submissions[0]
    assert isinstance(qasm, str)
    assert "measure q[0] -> c[0];" in qasm
    assert "measure q[1] -> c[1];" in qasm
    assert options["compiler"] == "qiskit"
    assert result.counts == [{"01": 1024}]
    summary = result.provenance["terminal_measurement_normalization"]
    assert summary["measurement_mapping"] == [
        {"qubit": 0, "classical_bit": 1},
        {"qubit": 1, "classical_bit": 0},
    ]
    assert len(summary["normalization_id"]) == 64
    assert (
        result.measurements[0].metadata["terminal_measurement_normalization"] == summary
    )


def test_identity_terminal_mapping_preserves_static_public_counts(
    client: _Client,
) -> None:
    result = fq.run(_terminal_program(), target="quafu:Baihua", shots=1024)

    assert len(client.submissions) == 1
    assert result.counts == [{"10": 1024}]


def test_incompatible_terminal_output_fails_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid output must fail before provider construction")

    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.QuafuProvider",
        reject_provider,
    )

    with pytest.raises(ValueError, match="one full-register fq.counts"):
        fq.run(
            _terminal_program(),
            outputs=fq.samples(),
            target="quafu:Baihua",
            shots=1024,
        )


@pytest.mark.parametrize(
    "program,compiler",
    (
        (DynamicCircuit(2).h(0).measure(0, classical_bit=0), None),
        (DynamicCircuit(2).measure(0).x(1).measure(1), None),
        (_terminal_program(), "qsteed"),
    ),
)
def test_non_v1_or_physical_paths_keep_dynamic_preflight_fail_closed(
    client: _Client,
    program: DynamicCircuit,
    compiler: str | None,
) -> None:
    with pytest.raises(fqe.CapabilityError, match="dynamic_remote_preflight_failed"):
        fq.run(
            program,
            compiler=compiler,
            target="quafu:Baihua",
            shots=1024,
        )

    assert client.submissions == []
