"""Implementation of workflows composed by the stable root facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .circuit import Circuit
    from .core.ir import CircuitIR
    from .noise import NoiseModel
    from .observables import OutputRequest
    from .runtime.execution_plan import ExecutionPlan
    from .runtime.options import ExecutionOptions
    from .runtime.result import ExecutionResult


def compile(
    program: Any,
    *,
    compiler: str | None = None,
    target: str | Mapping[str, Any] | None = None,
    target_qubits: Sequence[int] | None = None,
) -> Any:
    """Compile a circuit with FlagQuantum or one named installed compiler.

    Examples:
        >>> import flagquantum as fq
        >>> fq.compile(fq.Circuit(2).h(0).cx(0, 1)).n_wires
        2
    """

    ir = import_module(".core.ir", __package__).ensure_circuit_ir(program)
    if compiler is None or compiler == "flagquantum":
        if target is not None or target_qubits is not None:
            raise ValueError(
                "fq.compile target selection requires a named external compiler; "
                "use flagquantum.compiler.compile for manual topology compilation"
            )
        return import_module(".compiler", __package__).compile(ir)
    if not isinstance(compiler, str) or not compiler.strip():
        raise TypeError("compiler must be a non-empty installed compiler name")

    resolved_target: dict[str, Any] | None
    if isinstance(target, str):
        provider_name, separator, backend = target.partition(":")
        if separator != ":" or not provider_name or not backend:
            raise ValueError("target must use the form 'provider:backend'")
        if provider_name.lower() != "quafu":
            raise ValueError(f"unsupported compiler target provider {provider_name!r}")
        provider = import_module(".remote", __package__).QuafuProvider()
        resolved_target = {
            "provider": "quafu",
            "backend": backend,
            "chip_info": provider.fetch_chip_info(backend),
        }
        if target_qubits is not None:
            resolved_target["target_qubits"] = target_qubits
    elif target is None:
        if target_qubits is not None:
            raise ValueError("target_qubits requires a compiler target")
        resolved_target = None
    elif isinstance(target, Mapping):
        resolved_target = dict(target)
        if target_qubits is not None:
            if "target_qubits" in resolved_target:
                raise ValueError("target_qubits was specified twice")
            resolved_target["target_qubits"] = target_qubits
    else:
        raise TypeError("target must be 'provider:backend', a mapping, or None")

    extensions = import_module(".ecosystem.extensions", __package__)
    return extensions.compile_with_extension(
        ir,
        extension=compiler.strip(),
        target=resolved_target,
    )


def run(
    program_or_plan: Circuit | CircuitIR | ExecutionPlan,
    *,
    options: ExecutionOptions | None = None,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    noise_model: NoiseModel | None = None,
    compiler: str | None = None,
    target: str | None = None,
    target_qubits: Sequence[int] | None = None,
    shots: int | None = None,
    name: str | None = None,
) -> ExecutionResult:
    """Execute locally, or compile and execute on one named remote target.

    Examples:
        >>> import flagquantum as fq
        >>> circuit = fq.Circuit(2).h(0).cx(0, 1)
        >>> fq.run(circuit, outputs=fq.probabilities()).probabilities.shape
        torch.Size([1, 4])
    """

    from .runtime.execution_plan import ExecutionPlan

    remote_requested = (
        compiler is not None or target is not None or target_qubits is not None
    )
    if not remote_requested:
        if name is not None:
            raise TypeError("name is a direct fq.run keyword only for remote execution")
        if isinstance(program_or_plan, ExecutionPlan) and outputs is not None:
            raise TypeError("outputs must be None when executing an ExecutionPlan")
        if shots is not None and options is not None and options.shots is not None:
            raise TypeError("shots was specified both directly and in ExecutionOptions")
        selected_shots = shots if shots is not None else getattr(options, "shots", None)
        selected_seed = getattr(options, "seed", None)
        measurements = None
        if outputs is not None:
            ir = import_module(".core.ir", __package__).ensure_circuit_ir(
                program_or_plan
            )
            measurements = import_module(".observables", __package__).lower_outputs(
                outputs,
                n_wires=ir.n_wires,
                shots=selected_shots,
                seed=selected_seed,
            )
            import_module(".runtime.measurements", __package__).validate_measurements(
                measurements,
                n_wires=ir.n_wires,
            )
        elif shots is not None:
            raise TypeError(
                "local shots requires fq.samples(...) or fq.counts(...) output"
            )
        return import_module(".runtime.execution", __package__).run(
            program_or_plan,
            options=options,
            measurements=measurements,
            noise_model=noise_model,
        )

    provider_name, separator, _ = (target or "").partition(":")
    if separator == ":" and provider_name.lower() == "jiuding":
        if compiler is not None:
            raise TypeError("Jiuding workspace execution does not accept compiler")
        if target_qubits is not None:
            raise TypeError("Jiuding workspace execution does not accept target_qubits")
        if options is not None or noise_model is not None:
            raise TypeError(
                "options and noise_model are not yet supported by Jiuding workspace execution"
            )
        if shots is not None or name is not None:
            raise TypeError("Jiuding workspace execution does not accept shots or name")
        if isinstance(program_or_plan, ExecutionPlan):
            raise TypeError(
                "Jiuding workspace execution requires a Circuit or CircuitIR, not an ExecutionPlan"
            )
        jiuding = import_module(".remote.compute.jiuding", __package__)
        return jiuding.run(program_or_plan, target=target, outputs=outputs)

    if compiler is None or target is None:
        raise TypeError("remote execution requires both compiler and target")
    if options is not None or noise_model is not None:
        raise TypeError(
            "options and noise_model are local execution inputs; "
            "compile the remote circuit before requesting provider-specific behavior"
        )
    if type(shots) is not int or shots <= 0:
        raise ValueError("remote execution shots must be a positive integer")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("remote execution name must be a non-empty string")

    if separator != ":" or provider_name.lower() != "quafu":
        raise ValueError("remote fq.run currently supports target='quafu:<backend>'")

    output_types = import_module(".observables", __package__)
    requested = output_types.counts() if outputs is None else outputs
    requested = (
        (requested,)
        if isinstance(requested, output_types.OutputRequest)
        else tuple(requested)
    )
    source_ir = import_module(".core.ir", __package__).ensure_circuit_ir(
        program_or_plan
    )
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
    compiled = compile(
        program_or_plan,
        compiler=compiler,
        target=target,
        target_qubits=target_qubits,
    )
    remote = import_module(".remote", __package__)
    provider = remote.QuafuProvider()
    if expectation_output:
        assert requested[0].observable is not None
        algorithms = import_module(".algorithms", __package__)
        hamiltonian = algorithms.Hamiltonian(
            algorithms.HamiltonianTerm(
                term.coefficient,
                {wire: axis for wire, axis in term.factors},
            )
            for term in requested[0].observable.terms
        )
        deployment = import_module(".deployment", __package__)
        deployment_name = name.strip() if name is not None else "flagquantum_job"
        measurement_plan = deployment.create_pauli_measurement_plan(
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
                "routing_evidence_sha256": package.metadata.get(
                    "routing_evidence_sha256"
                ),
                "provider_result": dict(result.metadata),
            }
            for index, (package, result) in enumerate(
                zip(measurement_plan.packages, native_results, strict=True)
            )
        )
        observable_wires = tuple(
            sorted(
                {
                    wire
                    for term in requested[0].observable.terms
                    for wire, _axis in term.factors
                }
            )
        )
        contracts = import_module(".runtime.contracts", __package__)
        result = contracts.ExecutionResult(
            measurements=(
                contracts.MeasurementResult(
                    kind="expectation",
                    wires=observable_wires or (0,),
                    value=value,
                    shots=sum(measurement_plan.shots_per_group),
                    metadata={
                        "fq_output_index": 0,
                        "fq_output_kind": "expectation",
                        "fq_output_name": requested[0].name,
                        **(
                            {"name": requested[0].name}
                            if requested[0].name is not None
                            else {}
                        ),
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

    deployment_options: dict[str, Any] = {"shots": shots}
    if name is not None:
        deployment_options["name"] = name.strip()
    native = import_module(".deployment", __package__).deploy_circuit(
        compiled, provider, **deployment_options
    )
    contracts = import_module(".runtime.contracts", __package__)
    counts = {str(key): int(value) for key, value in native.counts.items()}
    execution_target = dict(compiled.metadata.get("execution_target", {}))
    result = contracts.ExecutionResult(
        measurements=(
            contracts.MeasurementResult(
                kind="counts",
                wires=tuple(range(compiled.n_wires)),
                value=[counts],
                shots=native.shots,
                metadata={
                    "fq_output_index": 0,
                    "fq_output_kind": "counts",
                    "fq_output_name": requested[0].name,
                    **(
                        {"name": requested[0].name}
                        if requested[0].name is not None
                        else {}
                    ),
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


def plan(
    program: Circuit | CircuitIR,
    *,
    options: ExecutionOptions | None = None,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    noise_model: NoiseModel | None = None,
) -> ExecutionPlan:
    """Build a backend-neutral execution plan.

    Examples:
        >>> import flagquantum as fq
        >>> fq.plan(fq.Circuit(1).h(0)).state_mode
        'statevector'
    """

    ir = import_module(".core.ir", __package__).ensure_circuit_ir(program)
    measurements = import_module(".observables", __package__).lower_outputs(
        outputs,
        n_wires=ir.n_wires,
        shots=getattr(options, "shots", None),
        seed=getattr(options, "seed", None),
    )
    if measurements is not None:
        import_module(".runtime.measurements", __package__).validate_measurements(
            measurements,
            n_wires=ir.n_wires,
        )
    return import_module(".runtime.planner", __package__).plan(
        program,
        options=options,
        measurements=measurements,
        noise_model=noise_model,
    )


__all__: tuple[str, ...] = ()
