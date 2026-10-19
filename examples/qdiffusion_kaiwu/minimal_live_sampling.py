"""Submit one bounded QBoson sampling task through FlagQuantum.

This is the shortest live path through the integration introduced by PR 598::

    FlagQuantum QuboProblem or Hamiltonian
        -> flagquantum.ecosystem.kaiwu.KaiwuSampler
        -> flagquantum.remote.kaiwu.KaiwuSDKClient
        -> Kaiwu SDK and the QBoson service

The command consumes exactly ten sampling credits when the matrix is not
already represented by the retained provider checkpoint. It has no local
fallback and requires an explicit cost acknowledgement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flagquantum.algorithms.qubo import QuboProblem
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import KaiwuSDKClient, KaiwuTaskClient

ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_TEN_QBOSON_SAMPLING_CREDITS"
PROBLEM = QuboProblem(
    n_variables=2,
    linear={0: -1.0, 1: -1.5},
    quadratic={(0, 1): 1.5},
    offset=0.25,
)


def run_live_sampling(
    *,
    client: KaiwuTaskClient,
    receipt_output: Path,
    task_name: str,
    project_no: str | None = None,
) -> KaiwuSampler:
    """Run one ten-sample task and retain its credential-free receipt."""

    sampler = KaiwuSampler(
        client=client,
        task_name=task_name,
        project_no=project_no,
        requested_samples=10,
        timeout=3600.0,
        poll_interval=60.0,
        max_remote_calls=1,
        integer_target_range=(-127, 127),
    )
    assignments = sampler.solve_qubo(PROBLEM)
    if sampler.last_job is None or sampler.last_result is None:
        raise RuntimeError("QBoson sampling returned without a retained job and result")
    sampler.last_job.save(receipt_output)

    print("FlagQuantum -> Kaiwu -> QBoson sampling completed")
    print(f"  assignments: {assignments.shape[0]}")
    print(f"  variables per assignment: {assignments.shape[1]}")
    print(f"  first binary assignment: {assignments[0].tolist()}")
    print(f"  remote calls: {sampler.remote_call_count}")
    print(f"  fallback: {sampler.last_result.metadata.get('fallback_occurred')}")
    print(f"  recovery receipt: {receipt_output}")
    return sampler


def main() -> None:
    """Parse the guarded live command and construct the SDK transport."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--task-name", default="flagquantum-minimal-sampling")
    parser.add_argument("--project-no")
    parser.add_argument("--acknowledge-provider-cost", required=True)
    arguments = parser.parse_args()
    if arguments.acknowledge_provider_cost != ACKNOWLEDGEMENT:
        parser.error(
            "--acknowledge-provider-cost must equal "
            f"{ACKNOWLEDGEMENT!r}; no provider client was created"
        )

    client = KaiwuSDKClient(
        checkpoint_dir=arguments.checkpoint_dir,
        expected_version="1.3.1",
    )
    run_live_sampling(
        client=client,
        receipt_output=arguments.receipt_output,
        task_name=arguments.task_name,
        project_no=arguments.project_no,
    )


if __name__ == "__main__":
    main()
