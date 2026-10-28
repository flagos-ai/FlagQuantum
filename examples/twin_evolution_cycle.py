"""Run, resume, and explicitly decide one offline Twin evolution cycle.

The task receipts and counts in this executable are deterministic synthetic
fixtures. Real applications obtain them through their explicit provider
workflow, then use the same validation and cycle APIs shown here.
"""

from __future__ import annotations

from twin_candidate_fit import incumbent_twin, probes, synthetic_report

import flagquantum as fq
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
)


def _counts(probabilities: tuple[float, ...], shots: int) -> dict[str, int]:
    values = [round(value * shots) for value in probabilities]
    values[max(range(len(values)), key=values.__getitem__)] += shots - sum(values)
    width = (len(values) - 1).bit_length()
    return {
        format(index, f"0{width}b"): count
        for index, count in enumerate(values)
        if count
    }


def _holdout_evaluation(incumbent, candidate):
    circuits = (
        fq.Circuit(2).h(0).cx(0, 1),
        fq.Circuit(2).x(0).cx(0, 1),
    )
    suite = fq.twin.prepare_candidate_suite(
        incumbent,
        candidate,
        circuits,
        name="offline-evolution-holdout",
        shots=4096,
    )
    submissions = []
    results = []
    for index, trial in enumerate(suite.trials, start=1):
        experiment = trial.experiment
        receipt = ProviderTaskHandle(
            provider="quafu",
            task_id=f"synthetic-holdout-{index}",
            backend_name=experiment.backend_name,
            payload={
                "deployment_receipt_schema": "flagquantum_submission_receipt_v1",
                "deployment_package_schema": "flagquantum_submitted_qasm_v1",
                "deployment_program_format": "openqasm-2",
                "routing_evidence_sha256": experiment.submitted_qasm_identity,
                "deployment_artifact_sha256": experiment.submitted_qasm_identity,
                "submitted_qasm_sha256": experiment.submitted_qasm_identity,
                "compiler": None,
                "target_qubits": list(experiment.target_qubits),
            },
        )
        submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
        bound_receipt = submission.receipt
        submissions.append(submission)
        results.append(
            DeploymentResult(
                handle=bound_receipt,
                counts=_counts(experiment.prediction.twin_probabilities, 4096),
                shots=4096,
                metadata=build_result_metadata(
                    bound_receipt,
                    {
                        "chip": experiment.backend_name,
                        "transpiled": experiment.submitted_qasm,
                    },
                ),
            )
        )
    return suite.validate_results(
        submissions,
        results,
        circuits=circuits,
        confidence_level=0.95,
    )


def main() -> None:
    incumbent = incumbent_twin()
    probe_circuits = probes()
    bank = fq.twin.prepare_probe_bank(
        target="quafu:Shenglian",
        physical_qubits=(20, 27),
        probes=probe_circuits,
        parameter_family="local_pauli_readout",
    )
    reports = tuple(
        synthetic_report(incumbent, circuit, index)
        for index, circuit in enumerate(probe_circuits, start=1)
    )
    estimate = fq.twin.estimate_probe_parameters(bank, reports)
    candidate = fq.twin.fit_candidate(
        incumbent,
        estimate,
        captured_at="2026-08-15T10:30:00+08:00",
    )

    freshness = fq.twin.TwinFreshnessAssessment(
        status="refresh_due",
        snapshot_identity=incumbent.snapshot.identity,
        evaluated_at="2026-08-15T09:00:00+08:00",
        calibration_age_seconds=81_000.0,
        validation_snapshot_age_seconds=None,
        next_refresh_at="2026-08-15T08:30:00+08:00",
        calibration_drift=None,
        validation_series=None,
        reasons=("maximum_calibration_age_exceeded",),
    )
    circuit_identities = tuple(circuit.to_ir().content_hash for circuit in probes())
    plan = fq.twin.TwinProbeRefreshPlan(
        release_identity="1" * 64,
        candidate_region_identity="2" * 64,
        candidate_snapshot_identity=incumbent.snapshot.identity,
        target="quafu:Shenglian",
        physical_qubits=(20, 27),
        candidate_circuit_identities=circuit_identities,
        selected_circuit_identities=circuit_identities[:2],
        unsupported_candidates=(),
        covered_directed_couplers=((20, 27),),
        coverage_gaps=(),
        calibration_history_identity=None,
        validation_history_identity=None,
        shots=4096,
        maximum_tasks=4,
    )

    cycle = fq.twin.prepare_evolution_cycle(
        incumbent=incumbent,
        freshness=freshness,
        probe_plan=plan,
    ).with_parameter_estimate(estimate)
    fq.twin.dump_evolution_cycle(cycle, "twin-evolution-estimated.json")

    # A later process can resume from identities without resubmitting probes.
    cycle = fq.twin.load_evolution_cycle("twin-evolution-estimated.json")
    cycle = cycle.with_candidate(candidate)
    evaluation = _holdout_evaluation(incumbent, candidate)
    cycle = cycle.with_holdout_evaluation(evaluation)
    fq.twin.dump_evolution_cycle(cycle, "twin-evolution-complete.json")

    # A recommendation has no effect until an application records its decision.
    # This example deliberately retains the incumbent, showing that the operator
    # decision remains distinct from the evidence recommendation.
    decision = fq.twin.record_evolution_decision(
        cycle,
        decision="retain_incumbent",
        decided_at="2026-08-15T11:00:00+08:00",
        decision_authority="example-calibration-operator",
    )
    fq.twin.dump_evolution_decision(decision, "twin-evolution-decision.json")

    print("stage:", cycle.stage)
    print("holdout decision:", cycle.holdout_decision)
    print("recommended action:", cycle.recommended_action)
    print("operator decision:", decision.decision)
    print("followed recommendation:", decision.followed_recommendation)
    print("cycle identity:", cycle.identity)
    print("decision identity:", decision.identity)
    print("record only; no candidate was released or activated")


if __name__ == "__main__":
    main()
