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
) -> Any:
    """Compile through the built-in pipeline or one installed compiler."""

    ir = import_module(".core.ir", __package__).ensure_circuit_ir(program)
    if compiler is None or compiler == "flagquantum":
        if target is not None:
            raise ValueError(
                "fq.compile target selection requires a named external compiler; "
                "use flagquantum.compiler.compile for manual topology compilation"
            )
        return import_module(".compiler", __package__).compile(ir)
    if not isinstance(compiler, str) or not compiler.strip():
        raise TypeError("compiler must be a non-empty installed compiler name")

    resolved_target: Mapping[str, Any] | None
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
    elif target is None:
        resolved_target = None
    elif isinstance(target, Mapping):
        resolved_target = dict(target)
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
    shots: int | None,
) -> Any:
    """Execute one local plan or one explicit remote QPU workflow."""

    remote_requested = compiler is not None or target is not None
    if not remote_requested:
        if shots is not None:
            raise TypeError(
                "shots is a direct fq.run keyword only for remote execution; "
                "use ExecutionOptions(shots=...) for local execution"
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

    provider_name, separator, _ = target.partition(":")
    if separator != ":" or provider_name.lower() != "quafu":
        raise ValueError("remote fq.run currently supports target='quafu:<backend>'")

    compiled = compile_program(program_or_plan, compiler=compiler, target=target)
    remote = import_module(".remote", __package__)
    native = import_module(".deployment", __package__).deploy_circuit(
        compiled,
        remote.QuafuProvider(),
        shots=shots,
    )
    contracts = import_module(".runtime.contracts", __package__)
    counts = {str(key): int(value) for key, value in native.counts.items()}
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
            "deployment": dict(native.metadata),
        },
        runtime={"mode": "remote_qpu", "shots": native.shots},
    )
    object.__setattr__(result, "_native_output", native)
    return result


__all__: tuple[str, ...] = ()
