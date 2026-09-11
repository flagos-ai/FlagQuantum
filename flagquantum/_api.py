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
        from .runtime.execution import run as run_local

        return run_local(
            program_or_plan,
            options=options,
            measurements=measurements,
            noise_model=noise_model,
        )

    provider_name, separator, _ = (target or "").partition(":")
    if target is not None and separator == ":" and provider_name.lower() == "jiuding":
        from .remote.compute.execution import execute_jiuding

        return execute_jiuding(
            program_or_plan,
            target=target,
            outputs=outputs,
            shots=shots,
            options=options,
            noise_model=noise_model,
            compiler=compiler,
            target_qubits=target_qubits,
            name=name,
        )

    if target is None:
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

    from .remote.qpu.execution import execute_quafu, validate_quafu_output

    output = validate_quafu_output(program_or_plan, outputs)
    if compiler is None:
        if shots % 1024:
            raise ValueError("Quafu shots must be a positive multiple of 1024")
        return execute_quafu(
            import_module(".core.ir", __package__).ensure_circuit_ir(program_or_plan),
            output=output,
            compiler=None,
            target=target,
            shots=shots,
            name=name,
            target_qubits=target_qubits,
        )
    compiled = compile(
        program_or_plan,
        compiler=compiler,
        target=target,
        target_qubits=target_qubits,
    )
    return execute_quafu(
        compiled,
        output=output,
        compiler=compiler,
        target=target,
        shots=shots,
        name=name,
    )


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
    from .runtime.planner import plan as plan_execution

    return plan_execution(
        program,
        options=options,
        measurements=measurements,
        noise_model=noise_model,
    )


__all__: tuple[str, ...] = ()
