"""Compose one compiled Quafu execution into a FlagQuantum result."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...algorithms import Hamiltonian, HamiltonianTerm
from ...core.ir import ensure_circuit_ir
from ...deployment import create_pauli_measurement_plan, deploy_circuit
from ...observables import OutputRequest
from ...observables import counts as request_counts
from ...runtime.result import ExecutionResult, MeasurementResult
from .quafu import QuafuProvider

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...core.ir import CircuitIR
    from ...runtime.execution_plan import ExecutionPlan


def execute_quafu(
    compiled: CircuitIR,
    *,
    output: OutputRequest,
    compiler: str,
    target: str,
    shots: int,
    name: str | None,
) -> ExecutionResult:
    """Submit one compiled circuit and normalize its Quafu result."""

    provider = QuafuProvider()
    if output.kind == "expectation":
        return _execute_expectation(
            compiled,
            provider=provider,
            output=output,
            compiler=compiler,
            target=target,
            shots=shots,
            name=name,
        )
    return _execute_counts(
        compiled,
        provider=provider,
        output=output,
        compiler=compiler,
        target=target,
        shots=shots,
        name=name,
    )


def validate_quafu_output(
    program: Circuit | CircuitIR | ExecutionPlan,
    outputs: OutputRequest | Sequence[OutputRequest] | None,
) -> OutputRequest:
    """Validate workflow input, rejecting local plans before remote execution."""

    requested_output = request_counts() if outputs is None else outputs
    requested = (
        (requested_output,)
        if isinstance(requested_output, OutputRequest)
        else tuple(requested_output)
    )
    source_ir = ensure_circuit_ir(program)
    full_counts = (
        len(requested) == 1
        and requested[0].kind == "counts"
        and requested[0].observable is None
        and requested[0].wires in {(), tuple(range(source_ir.n_wires))}
    )
    expectation_output = (
        len(requested) == 1
        and requested[0].kind == "expectation"
        and requested[0].observable is not None
    )
    if not full_counts and not expectation_output:
        raise ValueError(
            "remote Quafu execution supports one full-register fq.counts() "
            "or one fq.expectation(...) output"
        )

    observable = requested[0].observable
    if observable is not None and any(
        wire >= source_ir.n_wires
        for term in observable.terms
        for wire, _axis in term.factors
    ):
        raise ValueError("Hamiltonian references wires outside the source circuit")

    return requested[0]


def _execute_expectation(
    compiled: CircuitIR,
    *,
    provider: QuafuProvider,
    output: OutputRequest,
    compiler: str,
    target: str,
    shots: int,
    name: str | None,
) -> ExecutionResult:
    assert output.observable is not None
    hamiltonian = Hamiltonian(
        HamiltonianTerm(
            term.coefficient,
            {wire: axis for wire, axis in term.factors},
        )
        for term in output.observable.terms
    )
    deployment_name = name.strip() if name is not None else "flagquantum_job"
    measurement_plan = create_pauli_measurement_plan(
        compiled,
        hamiltonian,
        name=deployment_name,
        shots=shots,
    )
    native_results = tuple(
        provider.run(package) for package in measurement_plan.packages
    )
    grouped_counts = tuple(result.counts for result in native_results)
    value = measurement_plan.expectation(grouped_counts)
    standard_error = measurement_plan.standard_error(grouped_counts)
    execution_target = dict(compiled.metadata.get("execution_target", {}))
    task_ids = tuple(result.handle.task_id for result in native_results)
    group_evidence = tuple(
        {
            "group_index": index,
            "task_id": result.handle.task_id,
            "deployment_artifact_sha256": package.metadata.get(
                "deployment_artifact_sha256"
            ),
            "routing_evidence_sha256": package.metadata.get("routing_evidence_sha256"),
            "provider_result": dict(result.metadata),
        }
        for index, (package, result) in enumerate(
            zip(measurement_plan.packages, native_results, strict=True)
        )
    )
    observable_wires = tuple(
        sorted(
            {wire for term in output.observable.terms for wire, _axis in term.factors}
        )
    )
    result = ExecutionResult(
        measurements=(
            MeasurementResult(
                kind="expectation",
                wires=observable_wires or (0,),
                value=value,
                shots=sum(measurement_plan.shots_per_group),
                metadata={
                    "fq_output_index": 0,
                    "fq_output_kind": "expectation",
                    "fq_output_name": output.name,
                    **({"name": output.name} if output.name is not None else {}),
                    "provider": native_results[0].handle.provider,
                    "backend": native_results[0].handle.backend_name,
                    "task_ids": task_ids,
                },
                statistics={
                    "standard_error": float(standard_error.item()),
                    "group_count": len(measurement_plan.groups),
                    "shots_per_group": measurement_plan.shots_per_group,
                    "total_shots": sum(measurement_plan.shots_per_group),
                },
            ),
        ),
        provenance={
            "provider": native_results[0].handle.provider,
            "backend": native_results[0].handle.backend_name,
            "task_ids": task_ids,
            "compiler": compiler,
            "target": target,
            "target_qubits": tuple(execution_target.get("target_qubits", ())),
            "name": deployment_name,
            "measurement_plan": measurement_plan.summary(),
            "measurement_groups": group_evidence,
        },
        runtime={
            "mode": "remote_qpu",
            "shots": sum(measurement_plan.shots_per_group),
            "shots_per_group": measurement_plan.shots_per_group,
        },
    )
    object.__setattr__(result, "_native_output", native_results)
    return result


def _execute_counts(
    compiled: CircuitIR,
    *,
    provider: QuafuProvider,
    output: OutputRequest,
    compiler: str,
    target: str,
    shots: int,
    name: str | None,
) -> ExecutionResult:
    if name is None:
        native = deploy_circuit(compiled, provider, shots=shots)
    else:
        native = deploy_circuit(compiled, provider, shots=shots, name=name.strip())
    counts: dict[str | int, int] = {
        str(key): int(value) for key, value in native.counts.items()
    }
    execution_target = dict(compiled.metadata.get("execution_target", {}))
    result = ExecutionResult(
        measurements=(
            MeasurementResult(
                kind="counts",
                wires=tuple(range(compiled.n_wires)),
                value=[counts],
                shots=native.shots,
                metadata={
                    "fq_output_index": 0,
                    "fq_output_kind": "counts",
                    "fq_output_name": output.name,
                    **({"name": output.name} if output.name is not None else {}),
                    "provider": native.handle.provider,
                    "backend": native.handle.backend_name,
                    "task_id": native.handle.task_id,
                },
            ),
        ),
        provenance={
            "provider": native.handle.provider,
            "backend": native.handle.backend_name,
            "task_id": native.handle.task_id,
            "compiler": compiler,
            "target": target,
            "target_qubits": tuple(execution_target.get("target_qubits", ())),
            "name": name.strip() if name is not None else "flagquantum_job",
            "deployment": dict(native.metadata),
        },
        runtime={"mode": "remote_qpu", "shots": native.shots},
    )
    object.__setattr__(result, "_native_output", native)
    return result


__all__: tuple[str, ...] = ()
