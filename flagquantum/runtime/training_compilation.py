"""Bounded reusable training preparation; quantum math stays in Simulation."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import FunctionType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import torch
from torch.fx import GraphModule

from ..circuit import Circuit
from ..core.ir import ensure_circuit_ir
from ..errors import ExecutionError, FlagQuantumError, ValidationError
from .builder_compilation import can_defer_ir_snapshot, detached_ir_snapshot
from .result import ExecutionResult
from .result_adapters import LiveRuntimeSummary, _module_execution_result

if TYPE_CHECKING:
    from .module import Module

_GradientResult: TypeAlias = tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
]
_GradientCall: TypeAlias = Callable[..., tuple[_GradientResult, bool]]


@dataclass
class _TrainingProgram:
    objective: Callable[[torch.Tensor], torch.Tensor]
    signature: tuple[Any, ...] = ()
    gradient_indices: tuple[int, ...] = ()
    differentiated: Callable[..., _GradientResult] | None = None
    gradient_call: _GradientCall | None = None
    optimizer_call: Callable[[], Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    optimizer_diagnostics: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, *tensors: torch.Tensor) -> tuple[_GradientResult, bool]:
        assert self.differentiated is not None
        return self.differentiated(*tensors), torch.compiler.is_compiling()


def _tensor_signature(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(value.shape),
        value.dtype,
        value.device,
        value.stride(),
        value.requires_grad,
    )


def _configuration(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return ("tensor", id(value), _tensor_signature(value))
    if type(value) in {tuple, list}:
        return tuple(_configuration(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return (type(value).__name__, id(value))


def _objective_capture_configuration(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return ("tensor", _tensor_signature(value), value.is_inference())
    if type(value) is tuple:
        return tuple(_objective_capture_configuration(item) for item in value)
    return _configuration(value)


def _objective_signature(objective: Callable[..., Any]) -> tuple[Any, ...]:
    if not isinstance(objective, FunctionType):
        return (id(objective),)
    capture = inspect.getclosurevars(objective)
    values = {
        **capture.globals,
        **capture.nonlocals,
        **(objective.__kwdefaults__ or {}),
    }
    return (
        id(objective.__code__),
        tuple(
            (name, _objective_capture_configuration(value))
            for name, value in sorted(values.items())
        ),
        tuple(
            _objective_capture_configuration(value)
            for value in (objective.__defaults__ or ())
        ),
        tuple(
            (name, id(getattr(torch, name)))
            for name in objective.__code__.co_names
            if any(value is torch for value in values.values())
            and callable(getattr(torch, name, None))
        ),
    )


def _training_signature(
    module: Module,
    optimizer: torch.optim.Optimizer,
    objective: Callable[..., Any],
    tensors: tuple[torch.Tensor, ...],
) -> tuple[Any, ...]:
    return (
        id(module.circuit_builder),
        id(module._builder_program),
        module.policy,
        module.precision,
        module.training,
        tuple(id(value) for value in module._parameter_tensors()),
        tuple(_tensor_signature(value) for value in tensors),
        id(optimizer),
        id(objective),
        _objective_signature(objective),
        tuple(
            tuple(
                (
                    key,
                    (
                        tuple(id(value) for value in item)
                        if key == "params"
                        else _configuration(item)
                    ),
                )
                for key, item in sorted(group.items())
            )
            for group in optimizer.param_groups
        ),
    )


def _training_exclusion(
    module: Module, optimizer: torch.optim.Optimizer, tensors: tuple[torch.Tensor, ...]
) -> str:
    from .module import Module

    if (
        type(module) is not Module
        or "execute" in module.__dict__
        or "_build" in module.__dict__
    ):
        return "module_override_not_certified"
    source = Module.__init__.__code__.co_filename
    if any(
        getattr(getattr(getattr(module, name), "__func__", None), "__code__", None)
        is None
        or getattr(module, name).__func__.__code__.co_filename != source
        for name in (
            "execute",
            "_build",
            "_execute_impl",
            "_owned_parameters",
            "_forward_local_tensor",
        )
    ):
        return "module_override_not_certified"
    if module.policy.backend != "pytorch":
        return "explicit_backend"
    if module.policy.mode != "statevector" or module._parallel_plan is not None:
        return "representation_or_parallelism_not_certified"
    if module.policy.correctness_debug:
        return "correctness_debug_requires_eager"
    if not torch.is_grad_enabled():
        return "grad_mode_requires_eager"
    if type(optimizer) not in {torch.optim.SGD, torch.optim.Adam}:
        return "optimizer_not_certified"
    if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
        return "module_hooks_require_eager"
    for value in tensors:
        if type(value) not in {torch.Tensor, torch.nn.Parameter}:
            return "tensor_subclass_not_certified"
        if value.device.type != "cpu":
            return "device_not_certified"
        if value.is_inference() or not value.is_leaf:
            return "inference_or_upstream_autograd_requires_eager"
        if getattr(value, "_backward_hooks", None) or getattr(
            value, "_post_accumulate_grad_hooks", None
        ):
            return "tensor_hooks_require_eager"
    owned = tuple(module._parameter_tensors())
    optimized = tuple(
        value for group in optimizer.param_groups for value in group["params"]
    )
    if len(optimized) != len(owned) or {id(value) for value in optimized} != {
        id(value) for value in owned
    }:
        return "external_or_aliased_optimizer_parameters"
    if any(group.get("differentiable", False) for group in optimizer.param_groups):
        return "differentiable_optimizer_requires_eager"
    return ""


def _training_value(module: Module, inputs: torch.Tensor | None) -> torch.Tensor:
    # Reuse the existing value-only Module implementation for unobserved steps.
    # Its consumed-IR snapshot remains readable before the next data delivery.
    try:
        return module._forward_local_tensor(inputs, None)
    except (FlagQuantumError, TypeError):
        raise
    except Exception as error:
        raise ExecutionError("module execution failed") from error


def _optimizer_exclusion(optimizer: torch.optim.Optimizer) -> str:
    from torch.optim import optimizer as optimizer_core

    if type(optimizer) is not torch.optim.SGD:
        return "adam_optimizer_precision_not_certified"
    if "step" in optimizer.__dict__:
        return "optimizer_step_override"
    if (
        optimizer._optimizer_step_pre_hooks
        or optimizer._optimizer_step_post_hooks
        or (
            optimizer_core._global_optimizer_pre_hooks
            or optimizer_core._global_optimizer_post_hooks
        )
    ):
        return "optimizer_hooks_require_eager"
    if any(
        group.get("momentum", 0)
        or group.get("weight_decay", 0)
        or group.get("maximize", False)
        or group.get("fused", False)
        or group.get("foreach", False)
        for group in optimizer.param_groups
    ):
        return "sgd_options_not_certified"
    return ""


def _scalar_objective(
    objective: Callable[[torch.Tensor], torch.Tensor], value: torch.Tensor
) -> torch.Tensor:
    loss = objective(value)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValidationError("fq.train() objective must return a scalar tensor")
    return loss


def _prepare_training_program(
    program: _TrainingProgram,
    module: Module,
    circuit: Circuit,
    tensors: tuple[torch.Tensor, ...],
) -> str:
    template = module._training_template()
    binding = module._builder_program
    if (
        template is None
        or not isinstance(binding, GraphModule)
        or not can_defer_ir_snapshot(binding, template[0])
    ):
        return "binding_graph_not_certified"
    if type(circuit) is not Circuit or circuit.circuit_param["inputs"] is not None:
        return "circuit_constructor_not_certified"
    if module.policy.observable == "hamiltonian":
        return "hamiltonian_capture_not_certified"
    estimated = (
        (2**circuit.n_qubits)
        * circuit.bsz
        * (16 if circuit.dtype == torch.complex128 else 8)
        * max(8, 4 * len(template[0].instructions))
    )
    if estimated > 2 * 1024**3:
        return "capture_memory_budget"
    from ..simulation.statevector.torch_training import capture_training_value

    started = time.perf_counter()
    try:
        angles = module._builder_bindings.values()
        numerical = capture_training_value(
            template[0],
            dict(zip(template[1], angles, strict=True)),
            batch_size=circuit.bsz,
            device=torch.device(circuit.device),
            dtype=circuit.dtype,
            observable_qubits=module.policy.observable_qubits or (0,),
            sum_observables=module.policy.observable == "z_sum",
            return_state=True,
        )

        def loss(
            *values: torch.Tensor,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            bound = binding(*values)
            bound = (bound,) if isinstance(bound, torch.Tensor) else tuple(bound)
            value, state = numerical(*bound)
            return _scalar_objective(program.objective, value), (value, state)

        samples = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in tensors
        )
        indices = tuple(
            index for index, value in enumerate(samples) if value.requires_grad
        )
        if not indices:
            return "no_differentiable_inputs"
        sample_loss, _ = loss(*samples)
        if not sample_loss.requires_grad:
            return "no_differentiable_objective"
        derivatives = torch.autograd.grad(
            sample_loss, tuple(samples[index] for index in indices), allow_unused=True
        )
        # Functional AD returns zeros for unused arguments. Keep them absent so
        # an unused Adam parameter does not acquire a gradient or optimizer slot.
        active = tuple(
            index
            for index, derivative in zip(indices, derivatives, strict=True)
            if derivative is not None
        )
        if not active:
            return "no_differentiable_objective"
        program.gradient_indices = active
        program.differentiated = cast(
            Callable[..., _GradientResult],
            torch.func.grad_and_value(loss, argnums=active, has_aux=True),
        )
    except Exception as error:
        return f"training_capture_not_certified:{type(error).__name__}"
    finally:
        program.diagnostics["numerical_capture_seconds"] = time.perf_counter() - started
    program.diagnostics["estimated_capture_bytes"] = estimated
    return ""


def _prepared_execution(
    module: Module, circuit: Circuit, inputs: torch.Tensor | None, parameters: Any
) -> ExecutionResult:
    ir = ensure_circuit_ir(circuit)
    frozen = detached_ir_snapshot(ir)
    module._last_ir = frozen
    return _module_execution_result(
        module,
        frozen,
        value=None,
        state=None,
        inputs=inputs,
        parameters=parameters,
        qubits=module.policy.observable_qubits or (0,),
        runtime={
            **module._observe_topology(ir),
            "executor": "pytorch_training_tensor_program",
            "backend": "pytorch",
            "mode": "statevector",
            "world_size": 1,
            "local_world_size": 1,
            "node_count": 1,
            "distribution_semantics": "single_device_fast_path",
            "builder_lowering": "torch_fx_bindings",
            "builder_lowering_reason": "",
        },
        compatibility={"fallback_used": False},
    )


def _compilation_diagnostics(
    execution: ExecutionResult, diagnostics: dict[str, Any]
) -> ExecutionResult:
    # Preserve live distributed backward evidence instead of freezing its Mapping.
    if type(execution) is not ExecutionResult:
        return execution
    return replace(
        execution,
        runtime=LiveRuntimeSummary(
            execution.runtime, {"training_compilation": diagnostics}
        ),
    )


def _build_training_circuit(
    module: Module, inputs: torch.Tensor | None, parameters: Any
) -> Circuit:
    try:
        circuit = module._build(inputs, parameters)
        if not isinstance(circuit, Circuit):
            raise TypeError(
                "local fq.Module execution requires a Circuit builder result"
            )
        return circuit
    except (FlagQuantumError, TypeError):
        raise
    except Exception as error:
        raise ExecutionError("module execution failed") from error


def _snapshot_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    return {
        **statistics,
        "graph_break_reasons": tuple(statistics.get("graph_break_reasons", ())),
    }


def _execute_prebuilt(
    module: Module, circuit: Circuit, inputs: torch.Tensor | None, parameters: Any
) -> ExecutionResult:
    try:
        try:
            return module._execute_impl(inputs, parameters, _program=circuit)
        except (FlagQuantumError, TypeError):
            raise
        except Exception as error:
            raise ExecutionError("module execution failed") from error
    finally:
        module._builder_bindings.clear()
