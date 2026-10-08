from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.algorithms.qubo import QuboProblem
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

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.unit


class _CompletedClient:
    def __init__(self) -> None:
        self.submitted: list[FrozenIsingMatrix] = []

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submitted.append(matrix)
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id=f"fake-{len(self.submitted)}",
            provider_target="SPQC-fake",
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = (1, -1, 1)
        samples = tuple(sample for _ in range(receipt.requested_samples))
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        energies = tuple(energy for _ in samples)
        return KaiwuTaskResult(
            receipt,
            samples,
            energies,
            "Completed",
            {"fallback_occurred": False},
        )


class _RecoveringClient(_CompletedClient):
    def __init__(self) -> None:
        super().__init__()
        self.status_calls = 0

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        self.status_calls += 1
        return "Queued" if self.status_calls == 1 else "Completed"


class _ConcurrentClient(_CompletedClient):
    def __init__(self) -> None:
        super().__init__()
        self.submit_entries = 0
        self._entry_lock = threading.Lock()

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        with self._entry_lock:
            self.submit_entries += 1
        time.sleep(0.05)
        return super().submit(
            matrix,
            task_name=task_name,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
        )


class _IndeterminateSubmitClient(_CompletedClient):
    def __init__(self) -> None:
        super().__init__()
        self.submit_attempts = 0

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        del matrix, task_name, mode, requested_samples, project_no
        self.submit_attempts += 1
        raise RuntimeError("indeterminate provider submission")


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"timeout": True}, "timeout must be finite"),
        ({"timeout": "1"}, "timeout must be finite"),
        ({"poll_interval": False}, "poll_interval must be finite"),
        ({"poll_interval": "1"}, "poll_interval must be finite"),
        ({"project_no": 7}, "project_no must"),
        ({"project_no": "   "}, "project_no must"),
        ({"project_no": "project\nname"}, "project_no must"),
        ({"task_name": "task\tname"}, "task_name must"),
        ({"integer_target_range": (-1, True)}, "values must be integers"),
        ({"integer_target_range": (0, 1)}, "must straddle zero"),
    ),
)
def test_sampler_rejects_invalid_configuration_before_solve(
    options: dict[str, object], message: str
) -> None:
    client = _CompletedClient()
    arguments: dict[str, object] = {
        "client": client,
        "task_name": "invalid-config",
        **options,
    }

    with pytest.raises(ValueError, match=message):
        KaiwuSampler(**arguments)  # type: ignore[arg-type]

    assert client.submitted == []


def test_sampler_returns_plugin_compatible_numpy_spins() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="qdiffusion",
        requested_samples=10,
        max_remote_calls=2,
        poll_interval=0.01,
    )

    result = sampler.solve(
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
    )

    assert result.shape == (10, 3)
    assert result.dtype == np.int8
    assert set(np.unique(result)) == {-1, 1}
    assert sampler.client is client
    assert sampler.remote_call_count == 1
    assert sampler.last_result is not None
    assert sampler.last_result.metadata["fallback_occurred"] is False


def test_sampler_solves_flagquantum_hamiltonian_without_user_matrix() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="hamiltonian",
        requested_samples=10,
        max_remote_calls=1,
        poll_interval=0.01,
    )
    hamiltonian = Hamiltonian(
        (
            pauli_term(-1.0, "Z", (0,)),
            pauli_term(0.5, "ZZ", (0, 1)),
        )
    )

    logical_spins = sampler.solve_hamiltonian(hamiltonian)

    assert logical_spins.shape == (10, 2)
    assert logical_spins.dtype == np.int8
    assert logical_spins[0].tolist() == [1, -1]
    assert client.submitted[0] != ((-1.0, 0.5), (0.5, 0.0))


def test_sampler_solves_flagquantum_qubo_without_user_matrix() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="qubo",
        requested_samples=10,
        max_remote_calls=1,
        poll_interval=0.01,
    )
    problem = QuboProblem(
        n_variables=2,
        linear={0: -1.0, 1: -0.5},
        quadratic={(0, 1): 1.5},
        offset=0.25,
    )

    assignments = sampler.solve_qubo(problem)

    assert assignments.shape == (10, 2)
    assert assignments.dtype == np.int8
    assert assignments[0].tolist() == [1, 0]
    assert sampler.remote_call_count == 1


def test_identical_matrix_is_deduplicated_and_returns_a_copy() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(client=client, task_name="dedupe", max_remote_calls=1)
    matrix = [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]]

    first = sampler.solve(matrix)
    first[0, 0] = -1
    second = sampler.solve(matrix)

    assert second[0, 0] == 1
    assert sampler.remote_call_count == 1
    assert len(client.submitted) == 1
    assert len(sampler.transfer_records) == 2
    assert sampler.transfer_records[0].input_type == "builtins.list"
    assert sampler.transfer_records[0].input_device == "cpu"
    assert sampler.transfer_records[0].canonical_device == "cpu"
    assert sampler.transfer_records[0].canonical_dtype == "torch.float64"
    assert sampler.transfer_records[0].submission_storage == "cpu_python_tuple"
    assert sampler.transfer_records[0].returned_storage == "cpu_numpy"
    assert sampler.transfer_records[0].returned_dtype == "int8"
    assert sampler.transfer_records[0].matrix_shape == (3, 3)
    assert (
        sampler.transfer_records[0].submission_matrix_sha256
        == sampler.receipts[0].matrix_sha256
    )
    assert (
        sampler.transfer_records[0].original_matrix_sha256
        == sampler.transfer_records[1].original_matrix_sha256
    )
    assert sampler.transfer_records[0].returned_shape == (10, 3)
    assert sampler.transfer_records[0].cache_hit is False
    assert sampler.transfer_records[1].cache_hit is True
    assert (
        sampler.transfer_records[1].submission_matrix_sha256
        == sampler.receipts[0].matrix_sha256
    )


def test_concurrent_identical_solves_share_one_budgeted_remote_task() -> None:
    client = _ConcurrentClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="concurrent-dedupe",
        max_remote_calls=1,
    )
    matrix = [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]]
    start = threading.Barrier(2)

    def solve() -> np.ndarray:
        start.wait()
        return sampler.solve(matrix)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(solve)
        second = executor.submit(solve)
        results = (first.result(timeout=5), second.result(timeout=5))

    assert all(result.shape == (10, 3) for result in results)
    assert client.submit_entries == 1
    assert len(client.submitted) == 1
    assert sampler.remote_call_count == 1
    assert len(sampler.receipts) == 1
    assert [record.cache_hit for record in sampler.transfer_records] == [False, True]


def test_remote_call_budget_fails_before_second_unique_submission() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(client=client, task_name="budget", max_remote_calls=1)
    sampler.solve([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])

    with pytest.raises(RuntimeError, match="budget exhausted"):
        sampler.solve([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    assert len(client.submitted) == 1


def test_indeterminate_submission_consumes_budget_before_client_call() -> None:
    client = _IndeterminateSubmitClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="indeterminate-budget",
        max_remote_calls=1,
    )
    matrix = [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]]

    with pytest.raises(RuntimeError, match="indeterminate provider submission"):
        sampler.solve(matrix)

    assert client.submit_attempts == 1
    assert sampler.remote_call_count == 1
    assert sampler.receipts == ()

    with pytest.raises(RuntimeError, match="budget exhausted"):
        sampler.solve(matrix)

    assert client.submit_attempts == 1
    assert sampler.remote_call_count == 1


def test_retry_after_timeout_resumes_same_job_without_resubmission() -> None:
    client = _RecoveringClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="resume",
        timeout=0.0,
        max_remote_calls=1,
    )
    matrix = [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]]

    with pytest.raises(TimeoutError):
        sampler.solve(matrix)
    result = sampler.solve(matrix)

    assert result.shape == (10, 3)
    assert len(client.submitted) == 1
    assert sampler.remote_call_count == 1
    assert len(sampler.receipts) == 1


def test_integer_precision_is_applied_only_when_explicit() -> None:
    plain_client = _CompletedClient()
    plain = KaiwuSampler(client=plain_client, task_name="plain")
    scaled_client = _CompletedClient()
    scaled = KaiwuSampler(
        client=scaled_client,
        task_name="scaled",
        integer_target_range=(-7, 7),
    )
    matrix = [[0.0, 0.25, 0.0], [0.25, 0.0, 0.5], [0.0, 0.5, 0.0]]

    plain.solve(matrix)
    scaled.solve(matrix)

    assert plain_client.submitted[0] == (
        (0.0, 0.25, 0.0),
        (0.25, 0.0, 0.5),
        (0.0, 0.5, 0.0),
    )
    assert scaled_client.submitted[0] == (
        (0.0, 4.0, 0.0),
        (4.0, 0.0, 7.0),
        (0.0, 7.0, 0.0),
    )
    assert plain.last_precision_report is None
    assert scaled.last_precision_report is not None
    assert plain.precision_reports == ()
    assert plain.precision_evidence == ()
    assert len(scaled.precision_reports) == 1
    assert len(scaled.precision_evidence) == 1
    assert (
        scaled.precision_evidence[0].submission_matrix_sha256
        == scaled.receipts[0].matrix_sha256
    )
    assert scaled.precision_evidence[0].normalized_dtype == "torch.float64"
    assert scaled.precision_evidence[0].normalized_min == 0.0
    assert scaled.precision_evidence[0].normalized_max == 0.5
    assert scaled.precision_evidence[0].symmetry_normalization == "arithmetic_mean"
    assert scaled.precision_evidence[0].rounding_policy == "round_half_to_even"


def test_precision_reports_cover_distinct_inputs_that_share_one_remote_matrix() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="precision-collision",
        integer_target_range=(-1, 1),
        max_remote_calls=1,
    )
    first = [[0.0, 0.4, 0.0], [0.4, 0.0, 1.0], [0.0, 1.0, 0.0]]
    second = [[0.0, 0.49, 0.0], [0.49, 0.0, 1.0], [0.0, 1.0, 0.0]]

    sampler.solve(first)
    sampler.solve(second)

    assert sampler.remote_call_count == 1
    assert len(client.submitted) == 1
    assert len(sampler.precision_reports) == 2
    assert [report.max_abs_error for report in sampler.precision_reports] == [
        pytest.approx(0.4),
        pytest.approx(0.49),
    ]
    assert len(sampler.precision_evidence) == 2
    assert (
        len(
            {evidence.original_matrix_sha256 for evidence in sampler.precision_evidence}
        )
        == 2
    )
    assert {
        evidence.submission_matrix_sha256 for evidence in sampler.precision_evidence
    } == {sampler.receipts[0].matrix_sha256}
    assert {record.original_matrix_sha256 for record in sampler.transfer_records} == {
        evidence.original_matrix_sha256 for evidence in sampler.precision_evidence
    }


def test_result_identity_failure_is_not_replaced_by_fallback() -> None:
    class WrongIdentityClient(_CompletedClient):
        def fetch_result(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> KaiwuTaskResult:
            result = super().fetch_result(receipt, matrix)
            return replace(result, receipt=replace(receipt, task_name="wrong"))

    sampler = KaiwuSampler(client=WrongIdentityClient(), task_name="no-fallback")

    with pytest.raises(RuntimeError, match="does not match"):
        sampler.solve([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])

    assert sampler.remote_call_count == 1
    assert sampler.last_result is None
