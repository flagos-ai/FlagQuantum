"""Implementation of workflows composed by the stable root facade."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any


def compile_program(
    program: Any,
    *,
    compiler: str | None,
    target: str | Mapping[str, Any] | None,
    target_qubits: Any = None,
) -> Any:
    """Compile through the built-in pipeline or one installed compiler."""

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


def run_program(
    program_or_plan: Any,
    *,
    options: Any,
    measurements: Any,
    noise_model: Any,
    compiler: str | None,
    target: str | None,
    target_qubits: Any,
    shots: int | None,
    name: str | None,
) -> Any:
    """Execute one local plan or one explicit remote QPU workflow."""

    remote_requested = (
        compiler is not None or target is not None or target_qubits is not None
    )
    if not remote_requested:
        if shots is not None or name is not None:
            raise TypeError(
                "shots and name are direct fq.run keywords only for remote "
                "execution; use ExecutionOptions(shots=...) locally"
            )
        return import_module(".runtime.execution", __package__).run(
            program_or_plan,
            options=options,
            measurements=measurements,
            noise_model=noise_model,
        )

    if compiler is None or target is None:
        raise TypeError("remote execution requires both compiler and target")
    if options is not None or measurements is not None or noise_model is not None:
        raise TypeError(
            "options, measurements, and noise_model are local execution inputs; "
            "compile the remote circuit before requesting provider-specific behavior"
        )
    if type(shots) is not int or shots <= 0:
        raise ValueError("remote execution shots must be a positive integer")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("remote execution name must be a non-empty string")

    provider_name, separator, _ = target.partition(":")
    if separator != ":" or provider_name.lower() != "quafu":
        raise ValueError("remote fq.run currently supports target='quafu:<backend>'")

    compiled = compile_program(
        program_or_plan,
        compiler=compiler,
        target=target,
        target_qubits=target_qubits,
    )
    remote = import_module(".remote", __package__)
    deployment_options: dict[str, Any] = {"shots": shots}
    if name is not None:
        deployment_options["name"] = name.strip()
    native = import_module(".deployment", __package__).deploy_circuit(
        compiled, remote.QuafuProvider(), **deployment_options
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
