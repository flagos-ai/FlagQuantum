"""Implementation of workflows composed by the stable root facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

from torch import Tensor

if TYPE_CHECKING:
    from torch import Generator

    from .circuit import Circuit
    from .core.ir import CircuitIR
    from .gradients import GradientResult
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
                # ``lower_outputs`` took the qubit spelling; ``CircuitIR.n_wires``
                # keeps the old one because it is a frozen payload key.
                n_qubits=ir.n_wires,
                shots=selected_shots,
                seed=selected_seed,
            )
            import_module(".runtime.measurements", __package__).validate_measurements(
                measurements,
                n_qubits=ir.n_wires,
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

    if target is not None and separator == ":" and provider_name.lower() == "azure":
        if compiler is not None:
            raise TypeError(
                "Azure Quantum fq.run does not accept compiler; "
                "the provider route compiles OpenQASM 3 to QIR with QDK"
            )
        if options is not None or noise_model is not None:
            raise TypeError(
                "options and noise_model are local execution inputs; "
                "Azure Quantum provider controls are environment-owned"
            )
        if type(shots) is not int or shots <= 0:
            raise ValueError("remote execution shots must be a positive integer")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("remote execution name must be a non-empty string")
        if target_qubits is not None:
            raise TypeError(
                "Azure Quantum fq.run does not accept target_qubits; "
                "the QIR target compiler owns physical mapping"
            )
        from .remote.qpu.execution import execute_azure, validate_azure_output

        output = validate_azure_output(program_or_plan, outputs)
        source = import_module(".core.ir", __package__).ensure_circuit_ir(
            program_or_plan
        )
        return execute_azure(
            source,
            output=output,
            target=target,
            shots=shots,
            name=name,
        )

    if target is None or (
        compiler is None and not (separator == ":" and provider_name.lower() == "quafu")
    ):
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
        raise ValueError(
            "remote fq.run supports target='quafu:<backend>' or "
            "target='azure:<target-id>'"
        )

    from .remote.qpu.execution import execute_quafu, validate_quafu_output
    from .remote.qpu.quafu import _validate_quafu_shots

    output = validate_quafu_output(program_or_plan, outputs)
    _validate_quafu_shots(target, shots)
    if compiler is None:
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


def gradient(
    program: Callable[[Tensor], Any],
    parameters: Tensor,
    loss: Callable[[Any], Tensor] | None = None,
    *,
    method: str = "auto",
    step: float | None = None,
    directions: int = 1,
    generator: Generator | None = None,
) -> GradientResult:
    """Differentiate one scalar loss with respect to circuit parameters.

    ``method="auto"`` measures the program rather than trusting a declaration: it
    uses reverse-mode PyTorch autograd when the loss at ``parameters`` carries a
    graph, the exact per-opcode parameter-shift rule when a circuit is available,
    and central finite differences otherwise. The resolved method is reported on
    the result, so a fallback is never silent. ``method="adjoint"`` is refused,
    because FlagQuantum has no standalone adjoint entry point.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> theta = torch.tensor([0.3], dtype=torch.float64)
        >>> result = fq.gradient(
        ...     lambda p: fq.Circuit(1).rx(0, theta=p[0]).expectation_z(0), theta
        ... )
        >>> result.method
        'autograd'
    """

    from .gradients import gradient as differentiate

    return differentiate(
        program,
        parameters,
        loss,
        method=method,
        step=step,
        directions=directions,
        generator=generator,
    )


def jacobian(
    program: Callable[[Tensor], Tensor],
    parameters: Tensor,
) -> Tensor:
    """Differentiate every element of a vector-valued program.

    Use this when a program returns several values -- the probabilities of a
    register, one expectation per qubit, a batch of losses -- so that no single
    gradient exists. The result is shaped
    ``(*program_output.shape, *parameters.shape)``. For one scalar loss use
    :func:`gradient`, which can also choose an approximate method.

    Args:
        program: Maps a parameter tensor to a real tensor.
        parameters: Real, finite, non-empty parameter tensor.

    Returns:
        A detached tensor of shape ``(*program_output.shape, *parameters.shape)``
        in the dtype and on the device of ``parameters``.

    Raises:
        CapabilityError: If the program's output does not depend on
            ``parameters`` through PyTorch autograd.
        ValidationError: If ``parameters`` is empty, complex, or non-finite, or
            if the program returns a non-tensor, an empty tensor, or a complex
            tensor.
        TypeError: If ``parameters`` is not a tensor.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> theta = torch.tensor([0.3], dtype=torch.float64)
        >>> fq.jacobian(
        ...     lambda p: fq.Circuit(1).ry(0, theta=p[0]).probabilities(), theta
        ... ).shape
        torch.Size([1, 2, 1])
    """

    from .gradients import jacobian as differentiate_each

    return differentiate_each(program, parameters)


def jvp(
    program: Callable[[Tensor], Tensor],
    parameters: Tensor,
    tangents: Tensor,
) -> Tensor:
    """Apply the derivative of a vector-valued program to a parameter direction.

    This is the forward-mode action ``J v``: how the whole output moves when the
    ``parameters`` tensor moves along ``tangents``. It costs two reverse sweeps
    whatever the output size, so it stays affordable where :func:`jacobian` would
    need one sweep per output element. ``tangents`` must be shaped exactly like
    ``parameters``; a direction that merely broadcasts is refused, because
    broadcasting would silently differentiate along a different parameterization.

    Args:
        program: Maps a parameter tensor to a real tensor.
        parameters: Real, finite, non-empty parameter tensor.
        tangents: Real floating-point tensor shaped exactly like ``parameters``.

    Returns:
        A detached tensor shaped like the program's output, in the dtype and on
        the device of ``parameters``.

    Raises:
        CapabilityError: If the program's output does not depend on
            ``parameters``, or if the program cannot be differentiated twice.
        ValidationError: If ``parameters`` or ``tangents`` is empty, complex, or
            non-finite, if ``tangents`` is not shaped like ``parameters``, or if
            the program returns a non-tensor, an empty tensor, or a complex
            tensor.
        TypeError: If ``parameters`` or ``tangents`` is not a tensor.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> theta = torch.tensor([0.3], dtype=torch.float64)
        >>> one = torch.ones(1, dtype=torch.float64)
        >>> fq.jvp(
        ...     lambda p: fq.Circuit(1).ry(0, theta=p[0]).probabilities(), theta, one
        ... ).shape
        torch.Size([1, 2])
    """

    from .gradients import jvp as push_forward

    return push_forward(program, parameters, tangents)


def vjp(
    program: Callable[[Tensor], Tensor],
    parameters: Tensor,
    cotangents: Tensor,
) -> Tensor:
    """Apply the derivative of a vector-valued program to an output direction.

    This is the reverse-mode action ``c^T J``: how one scalar built from the
    program's output, ``<cotangents, program(parameters)>``, moves with the
    ``parameters`` tensor. It costs one reverse sweep whatever the parameter
    count, which is the shape a training step wants when a scalar objective is
    assembled from several measured values. ``cotangents`` must be shaped exactly
    like the program's output; weight every value equally with
    ``torch.ones_like(output)``.

    Args:
        program: Maps a parameter tensor to a real tensor.
        parameters: Real, finite, non-empty parameter tensor.
        cotangents: Real floating-point tensor shaped exactly like the program's
            output.

    Returns:
        A detached tensor shaped like ``parameters``, in the dtype and on the
        device of ``parameters``.

    Raises:
        CapabilityError: If the program's output does not depend on
            ``parameters`` through PyTorch autograd.
        ValidationError: If ``parameters`` or ``cotangents`` is empty, complex,
            or non-finite, if ``cotangents`` is not shaped like the program's
            output, or if the program returns a non-tensor, an empty tensor, or a
            complex tensor.
        TypeError: If ``parameters`` or ``cotangents`` is not a tensor.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> theta = torch.tensor([0.3], dtype=torch.float64)
        >>> weights = torch.ones(1, 2, dtype=torch.float64)
        >>> fq.vjp(
        ...     lambda p: fq.Circuit(1).ry(0, theta=p[0]).probabilities(),
        ...     theta,
        ...     weights,
        ... ).shape
        torch.Size([1])
    """

    from .gradients import vjp as pull_back

    return pull_back(program, parameters, cotangents)


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
        n_qubits=ir.n_wires,
        shots=getattr(options, "shots", None),
        seed=getattr(options, "seed", None),
    )
    if measurements is not None:
        import_module(".runtime.measurements", __package__).validate_measurements(
            measurements,
            n_qubits=ir.n_wires,
        )
    from .runtime.planner import plan as plan_execution

    return plan_execution(
        program,
        options=options,
        measurements=measurements,
        noise_model=noise_model,
    )


def from_openqasm(source: str) -> Any:
    """Import OpenQASM 2 or OpenQASM 3 text as a FlagQuantum program.

    This is the inverse of ``emit_openqasm``: it reads the canonical subset
    FlagQuantum writes and refuses everything else with an issue code, so an
    imported program is never an approximation of the text.

    Args:
        source: The complete text of one OpenQASM program.

    Returns:
        An :class:`~flagquantum.compiler.openqasm_import.OpenQASMImport`, which
        exposes the imported instructions, the declared register width, the
        qubit behind each classical bit, and ``to_circuit``.

    Examples:
        >>> import flagquantum as fq
        >>> program = fq.from_openqasm("OPENQASM 2.0;\\ninclude \\"qelib1.inc\\";\\n"
        ...     "qreg q[2];\\ncreg c[2];\\nh q[0];\\ncx q[0], q[1];\\n"
        ...     "measure q[0] -> c[0];\\nmeasure q[1] -> c[1];")
        >>> [instruction.name for instruction in program.instructions]
        ['h', 'cx']
    """

    return import_module(".compiler.openqasm_import", __package__).import_openqasm(
        source
    )


__all__: tuple[str, ...] = ()
