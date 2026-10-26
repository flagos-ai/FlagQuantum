"""Direct Quafu service compilation uses only the maintained SDK."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import flagquantum as fq
from flagquantum.remote.qpu import _quafu_sdk
from flagquantum.remote.qpu.quafu import QuafuProvider

pytestmark = pytest.mark.unit


class Client:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, dict[str, object]]] = []
        self.job_handle = SimpleNamespace(
            id="sdk-job-1",
            n_tasks=1,
            status=lambda: "Finished",
            result=lambda timeout=0: SimpleNamespace(
                status="Finished",
                task_id="sdk-job-1-0",
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
        assert job_id == "sdk-job-1"
        return self.job_handle


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Client:
    client = Client()
    adapter = _quafu_sdk._QuafuSdkAdapter.for_submission(client=client)
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.QuafuProvider",
        lambda **options: QuafuProvider(_sdk_adapter=adapter, **options),
    )
    monkeypatch.setattr(
        "flagquantum._api.compile",
        Mock(side_effect=AssertionError("local compiler called")),
    )
    return client


def test_direct_counts_use_sdk_service_compilation(client: Client) -> None:
    result = fq.run(fq.Circuit(2).x(0), target="quafu:Baihua", shots=1024)

    qasm, options = client.submissions[0]
    assert isinstance(qasm, str) and "qreg q[2];" in qasm
    assert options == {
        "target": "Baihua",
        "shots": 1024,
        "compiler": "qiskit",
        "name": "flagquantum_job",
    }
    assert result.counts == [{"10": 1024}]
    assert result.provenance["compilation_location"] == "service"
    assert result.provenance["service_compiler"] == "qiskit"


def test_service_compilation_rejects_requested_physical_mapping(
    client: Client,
) -> None:
    with pytest.raises(ValueError, match="compiler='qsteed'"):
        fq.run(
            fq.Circuit(2),
            target="quafu:Baihua",
            shots=1024,
            target_qubits=(17, 18),
        )

    assert client.submissions == []


@pytest.mark.parametrize("mapping", [(1,), (1, 1), (-1, 2), (True, 2), (1.5, 2), "12"])
def test_invalid_mapping_never_submits(client: Client, mapping: object) -> None:
    with pytest.raises(ValueError, match="target_qubits"):
        fq.run(fq.Circuit(2), target="quafu:Baihua", shots=1024, target_qubits=mapping)
    assert client.submissions == []
