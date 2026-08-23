"""Canonical PyTorch-native quantum module implementation."""

from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Mapping, cast

import torch

from ..circuit import Circuit
from ..core.ir import CircuitIR, ensure_circuit_ir
from ..core.operator_schema import get_operator_schema
from .builder_compilation import (
    BuilderBindings,
    CompiledInstruction,
    SlotParameterMapping,
    detached_ir_snapshot,
)
from .policy import RuntimePolicy
from .result import ExecutionResult, normalize_execution_result
from .result_adapters import LiveRuntimeSummary

if TYPE_CHECKING:
    from ..algorithms import Hamiltonian
    from .backends.mps.production import (
        MPSAcceptanceGates,
        MPSCrossoverMeasurement,
        MPSProductionPlan,
    )
    from .parallel import HybridParallelPlan
    from .training_state import PrecisionPolicy, SeedContract

CircuitBuilder = Callable[..., Circuit | CircuitIR]


class Module(torch.nn.Module):  # type: ignore[misc]
    """A normal ``torch.nn.Module`` whose quantum execution is policy-driven.

    ``circuit`` receives the owned parameter tensor, and optionally the input
    tensor when its signature accepts two positional arguments.
    """

    def __init__(
        self,
        circuit: CircuitBuilder | Circuit,
        n_parameters: int | tuple[int, ...] | None = None,
        *,
        parameters: Mapping[str, int | tuple[int, ...] | torch.Tensor] | None = None,
        policy: RuntimePolicy | None = None,
        init: torch.Tensor | Mapping[str, Any] | str | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        deployment_binding: Mapping[str, Any] | None = None,
        hamiltonian: Hamiltonian | None = None,
        precision: PrecisionPolicy | None = None,
    ) -> None:
        super().__init__()
        if not callable(circuit) and not isinstance(circuit, Circuit):
            raise TypeError("fq.Module circuit must be a callable builder or Circuit")
        if parameters is not None and n_parameters is not None:
            raise ValueError("pass either n_parameters or parameters, not both")
        if isinstance(circuit, Circuit) and parameters is None:
            parameters = {name: () for name in circuit.parameter_names}
        if parameters is None and n_parameters is None:
            raise ValueError(
                "Module requires n_parameters, named parameters, or a "
                "parameterized Circuit"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device or "cpu")
            generator.manual_seed(int(seed))

        def initialize(shape: tuple[int, ...], value: Any) -> torch.Tensor:
            if isinstance(value, str):
                strategy = value.strip().lower()
                target = torch.empty(shape, dtype=dtype, device=device)
                if strategy == "zeros":
                    return target.zero_()
                if strategy == "uniform":
                    return target.uniform_(0.0, 2.0 * torch.pi, generator=generator)
                if strategy == "normal":
                    return target.normal_(0.0, 0.01, generator=generator)
                raise ValueError(
                    "init strategy must be 'zeros', 'uniform', or 'normal', "
                    f"got {value!r}"
                )
            return torch.as_tensor(value, dtype=dtype, device=device).reshape(shape)

        self.named_parameter_groups: torch.nn.ParameterDict | None = None
        if parameters is not None:
            if not parameters:
                raise ValueError("named parameters cannot be empty")
            if init is not None and not isinstance(init, (Mapping, str)):
                raise TypeError(
                    "named parameters require a mapping or initialization strategy"
                )
            initial = dict(init) if isinstance(init, Mapping) else {}
            unknown = sorted(set(initial) - set(parameters))
            if unknown:
                raise KeyError(
                    "unknown named parameter initializer(s): " + ", ".join(unknown)
                )
            groups: dict[str, torch.nn.Parameter] = {}
            for name, specification in parameters.items():
                if not str(name):
                    raise ValueError("named parameter group names cannot be empty")
                if isinstance(specification, torch.Tensor):
                    default = specification.detach().to(device=device, dtype=dtype)
                    shape = tuple(default.shape)
                else:
                    shape = (
                        (int(specification),)
                        if isinstance(specification, int)
                        else tuple(int(item) for item in specification)
                    )
                    default = torch.zeros(shape, dtype=dtype, device=device)
                value = initial.get(name, init if isinstance(init, str) else default)
                tensor = initialize(shape, value)
                groups[str(name)] = torch.nn.Parameter(tensor.clone())
            self.named_parameter_groups = torch.nn.ParameterDict(groups)
            self.parameters_tensor = None
        else:
            assert n_parameters is not None
            shape = (
                (int(n_parameters),)
                if isinstance(n_parameters, int)
                else tuple(int(item) for item in n_parameters)
            )
            if isinstance(init, Mapping):
                raise TypeError("n_parameters requires tensor-like init or a strategy")
            values = initialize(shape, "zeros" if init is None else init)
            self.parameters_tensor = torch.nn.Parameter(values)
        self.circuit_builder = circuit
        self.policy = policy or RuntimePolicy()
        self.deployment_binding = dict(deployment_binding or {})
        self.hamiltonian = hamiltonian
        self._state_process_group: Any | None = None
        self._parallel_plan: HybridParallelPlan | None = None
        self._last_ir: CircuitIR | None = None
        self._compiled_topology_signature: str | None = None
        self._compile_count = 0
        self._program_cache_hits = 0
        self._builder_program: Any | None = None
        self._builder_program_signature: tuple[Any, ...] | None = None
        self._builder_template: Circuit | None = None
        self._builder_bound_circuit: Circuit | None = None
        self._builder_bindings = BuilderBindings()
        self._builder_slots: tuple[tuple[int, str], ...] = ()
        self._builder_compile_count = 0
        self._builder_cache_hits = 0
        self._jax_kernel: Any | None = None
        self._jax_kernel_signature: tuple[Any, ...] | None = None
        self._correctness_debug_hook_handle: Any | None = None
        if precision is None:
            from .training_state import PrecisionPolicy

            precision = PrecisionPolicy(
                parameter_dtype=str(dtype).removeprefix("torch."),
                complex_dtype=("complex128" if dtype == torch.float64 else "complex64"),
                accumulator_dtype=str(dtype).removeprefix("torch."),
            )
        self.precision = precision
        self.precision.apply(self)
        self._sync_correctness_debug_hook()

    def _sync_correctness_debug_hook(self) -> None:
        handle = self._correctness_debug_hook_handle
        if self.policy.correctness_debug and handle is None:
            self._correctness_debug_hook_handle = tuple(
                parameter.register_hook(self._finite_gradient_hook)
                for parameter in self._parameter_tensors()
            )
        elif not self.policy.correctness_debug and handle is not None:
            for item in handle:
                item.remove()
            self._correctness_debug_hook_handle = None

    def set_runtime_policy(self, policy: RuntimePolicy) -> None:
        """Replace runtime policy and synchronize policy-dependent resources."""

        if not isinstance(policy, RuntimePolicy):
            raise TypeError("fq.Module runtime policy must be a RuntimePolicy")
        self.policy = policy
        self._jax_kernel = None
        self._jax_kernel_signature = None
        self._sync_correctness_debug_hook()

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> "Module":
        if hasattr(self, "parameters_tensor"):
            parameter = self._parameter_tensors()[0]
            probe = fn(torch.empty(0, dtype=parameter.dtype, device=parameter.device))
            if probe.dtype not in {torch.float32, torch.float64}:
                raise TypeError(
                    "fq.Module trainable parameters must use float32 or float64"
                )
        super()._apply(fn, recurse=recurse)
        if hasattr(self, "precision"):
            dtype = self._parameter_tensors()[0].dtype
            if dtype not in {torch.float32, torch.float64}:
                raise TypeError(
                    "fq.Module trainable parameters must use float32 or float64"
                )
            name = str(dtype).removeprefix("torch.")
            self.precision = replace(
                self.precision,
                parameter_dtype=name,
                accumulator_dtype=name,
                complex_dtype="complex128" if dtype == torch.float64 else "complex64",
            )
        self._jax_kernel = None
        self._jax_kernel_signature = None
        object.__setattr__(self, "_builder_program", None)
        self._builder_program_signature = None
        self._builder_template = None
        self._builder_bound_circuit = None
        self._builder_bindings.clear()
        self._builder_slots = ()
        return self

    @staticmethod
    def _finite_gradient_hook(gradient: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(gradient).all():
            from .training_state import NonFiniteTrainingError

            raise NonFiniteTrainingError("non-finite training values: gradient")
        return gradient

    def _build(
        self, inputs: torch.Tensor | None, parameters: Any
    ) -> Circuit | CircuitIR:
        if isinstance(self.circuit_builder, Circuit):
            if inputs is not None:
                raise TypeError("a symbolic Circuit template does not accept inputs")
            return self.circuit_builder.bind_parameters(parameters)
        return self._build_from_compiled_builder(inputs, parameters)

    def _builder_tensor_inputs(
        self, parameters: Any, inputs: torch.Tensor | None
    ) -> tuple[tuple[torch.Tensor, ...], tuple[str, ...], bool]:
        if isinstance(parameters, (Mapping, torch.nn.ParameterDict)):
            names = tuple(parameters)
            tensors = tuple(parameters[name] for name in names)
        else:
            names = ()
            tensors = (parameters,)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("compiled circuit builder parameters must be tensors")
        return (
            tensors + (() if inputs is None else (inputs,)),
            names,
            inputs is not None,
        )

    def _build_from_compiled_builder(
        self, inputs: torch.Tensor | None, parameters: Any
    ) -> Circuit | CircuitIR:
        tensor_inputs, names, has_inputs = self._builder_tensor_inputs(
            parameters, inputs
        )
        signature = tuple(
            (tuple(tensor.shape), tensor.dtype, tensor.device)
            for tensor in tensor_inputs
        ) + (names, has_inputs)
        if (
            self._builder_program is None
            or self._builder_program_signature != signature
        ):
            self._compile_builder_program(
                tensor_inputs, names=names, has_inputs=has_inputs, signature=signature
            )
        else:
            self._builder_cache_hits += 1

        assert self._builder_program is not None
        assert self._builder_bound_circuit is not None
        dynamic_values = self._builder_program(*tensor_inputs)
        if isinstance(dynamic_values, torch.Tensor):
            dynamic_values = (dynamic_values,)
        else:
            dynamic_values = tuple(dynamic_values)
        if len(dynamic_values) != len(self._builder_slots):
            raise RuntimeError("compiled builder parameter slot count changed")
        self._builder_bindings.bind(dynamic_values)
        # ``Circuit.state`` is the only value-dependent cache on this reusable
        # container. Instructions and their parameter maps stay immutable.
        self._builder_bound_circuit._state_cache = None
        return self._builder_bound_circuit

    def _compile_builder_program(
        self,
        tensor_inputs: tuple[torch.Tensor, ...],
        *,
        names: tuple[str, ...],
        has_inputs: bool,
        signature: tuple[Any, ...],
    ) -> None:
        from torch.fx.experimental.proxy_tensor import make_fx

        captured: list[Circuit] = []

        def extract(*flat_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            parameter_count = len(flat_inputs) - int(has_inputs)
            raw_parameters = flat_inputs[:parameter_count]
            traced_parameters: Any = (
                {name: raw_parameters[index] for index, name in enumerate(names)}
                if names
                else raw_parameters[0]
            )
            traced_inputs = flat_inputs[-1] if has_inputs else None
            circuit = self._invoke_circuit_builder(traced_parameters, traced_inputs)
            if not isinstance(circuit, Circuit):
                raise TypeError("compiled circuit builder must return a Circuit")
            if any(
                isinstance(instruction.matrix, torch.Tensor)
                and instruction.matrix.requires_grad
                for instruction in circuit.to_ir().instructions
            ):
                raise TypeError("compiled builders do not yet support dynamic matrices")
            captured[:] = [circuit]
            return tuple(
                value
                for instruction in circuit.to_ir().instructions
                for value in instruction.params.values()
                if isinstance(value, torch.Tensor)
            )

        try:
            program = make_fx(extract)(*tensor_inputs)
        except Exception as error:
            raise RuntimeError(
                "Module builder compilation requires static circuit topology; "
                "do not branch on Tensor values when choosing gates or wires"
            ) from error
        program.graph.eliminate_dead_code()
        program.recompile()
        if not captured:
            raise RuntimeError("compiled builder did not capture a Circuit")
        template = captured[-1]
        if template._inputs is not None:
            raise TypeError(
                "compiled builders do not yet support dynamic Circuit inputs; "
                "encode inputs in gate parameters instead"
            )
        slots = tuple(
            (index, key)
            for index, instruction in enumerate(template.to_ir().instructions)
            for key, value in instruction.params.items()
            if isinstance(value, torch.Tensor)
        )
        bound = type(template)(**dict(template.circuit_param))
        bound._parameter_bindings = self._builder_bindings
        slot_index = 0
        compiled_instructions: list[CompiledInstruction] = []
        for instruction in template.to_ir().instructions:
            constants: dict[str, Any] = {}
            parameter_slots: dict[str, int] = {}
            for key, value in instruction.params.items():
                if isinstance(value, torch.Tensor):
                    parameter_slots[key] = slot_index
                    slot_index += 1
                else:
                    constants[key] = value
            compiled_instructions.append(
                CompiledInstruction(
                    name=instruction.name,
                    wires=instruction.wires,
                    params=SlotParameterMapping(
                        constants, parameter_slots, self._builder_bindings
                    ),
                    matrix=instruction.matrix,
                    metadata=instruction.metadata,
                    parameter_slots=tuple(
                        parameter_slots.get(name, -1)
                        for name in (
                            getattr(
                                get_operator_schema(instruction.name), "parameters", ()
                            )
                        )
                    ),
                    parameter_constants=tuple(
                        constants.get(name)
                        for name in (
                            getattr(
                                get_operator_schema(instruction.name), "parameters", ()
                            )
                        )
                    ),
                )
            )
        if slot_index != len(slots):
            raise RuntimeError("compiled builder parameter slot table is inconsistent")
        bound._instructions.extend(compiled_instructions)
        # Runtime compilation artifacts must not become checkpointed child
        # modules; they are rebuilt after dtype/device/signature changes.
        object.__setattr__(self, "_builder_program", program)
        self._builder_program_signature = signature
        self._builder_template = template
        self._builder_bound_circuit = bound
        self._builder_slots = slots
        self._builder_compile_count += 1

    def _invoke_circuit_builder(
        self, parameters: Any, inputs: torch.Tensor | None
    ) -> Circuit | CircuitIR:
        signature = inspect.signature(self.circuit_builder)
        positional = tuple(
            item
            for item in signature.parameters.values()
            if item.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if len(positional) >= 2:
            return self.circuit_builder(parameters, inputs)
        return self.circuit_builder(parameters)

    def _parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        if self.named_parameter_groups is not None:
            return tuple(self.named_parameter_groups.values())
        assert self.parameters_tensor is not None
        return (self.parameters_tensor,)

    def _owned_parameters(self) -> Any:
        return self.named_parameter_groups or self.parameters_tensor

    def _observe_topology(self, ir: CircuitIR) -> dict[str, Any]:
        """Track a value-independent circuit topology for backend program caches."""

        structure = (
            ir.n_wires,
            tuple(
                (
                    instruction.name,
                    instruction.wires,
                    tuple(sorted(instruction.params)),
                    instruction.matrix is not None,
                    tuple(getattr(instruction.matrix, "shape", ())),
                    bool(instruction.metadata.get("is_channel")),
                )
                for instruction in ir.instructions
            ),
            self.policy.mode,
            self.policy.observable,
            self.policy.observable_wires,
        )
        signature = hashlib.sha256(repr(structure).encode("utf-8")).hexdigest()
        if signature == self._compiled_topology_signature:
            self._program_cache_hits += 1
        else:
            self._compiled_topology_signature = signature
            self._compile_count += 1
        return {
            "compiled": True,
            "topology_hash": signature,
            "compile_count": self._compile_count,
            "program_cache_hits": self._program_cache_hits,
            "builder_compiled": self._builder_program is not None,
            "builder_compile_count": self._builder_compile_count,
            "builder_cache_hits": self._builder_cache_hits,
            "static_program_reused": self._builder_bound_circuit is not None,
        }

    @property
    def parameter_groups(self) -> Mapping[str, torch.Tensor]:
        """Return named trainable groups, or an empty mapping for flat modules."""

        return self.named_parameter_groups or {}

    def set_parallel_context(
        self,
        *,
        state_process_group: Any | None,
        plan: HybridParallelPlan,
    ) -> None:
        """Attach runtime-only process groups; they are not checkpoint state."""

        import torch.distributed as dist

        if not dist.is_initialized():
            if plan.world_size != 1:
                raise RuntimeError(
                    "multi-rank parallel plan requires an initialized process group"
                )
        else:
            global_world_size = dist.get_world_size()
            if global_world_size != plan.world_size:
                raise ValueError(
                    "global process-group size does not match plan.world_size: "
                    f"{global_world_size} != {plan.world_size}"
                )
            group_size = dist.get_world_size(state_process_group)
            if group_size != plan.state_parallel_size:
                raise ValueError(
                    "state process-group size does not match "
                    f"plan.state_parallel_size: {group_size} != "
                    f"{plan.state_parallel_size}"
                )
            global_rank = dist.get_rank()
            if state_process_group is None:
                group_ranks = tuple(range(dist.get_world_size()))
            else:
                get_ranks = getattr(dist, "get_process_group_ranks", None)
                if get_ranks is None:
                    raise RuntimeError(
                        "PyTorch must expose get_process_group_ranks to validate "
                        "hybrid parallel topology"
                    )
                group_ranks = tuple(
                    int(rank) for rank in get_ranks(state_process_group)
                )
            expected = next(
                (group for group in plan.state_groups if global_rank in group), None
            )
            if expected is None or tuple(group_ranks) != tuple(expected):
                raise ValueError(
                    "state process-group ranks do not match the plan topology: "
                    f"actual={group_ranks}, expected={expected}"
                )
        self._state_process_group = state_process_group
        self._parallel_plan = plan

    def plan_production_mps(
        self,
        inputs: torch.Tensor | None = None,
        *,
        estimated_workload_bytes: int,
        single_gpu_capacity_bytes: int,
        gates: MPSAcceptanceGates | Mapping[str, Any],
        crossover: tuple[MPSCrossoverMeasurement | Mapping[str, Any], ...] = (),
        available_gpu_count: int = 0,
    ) -> MPSProductionPlan:
        """Plan the module's MPS workload from measured release evidence."""

        from .backends.mps.production import plan_production_mps

        circuit = self._build(inputs, self._owned_parameters())
        ir = ensure_circuit_ir(circuit)
        self._last_ir = detached_ir_snapshot(ir)
        return plan_production_mps(
            ir,
            estimated_workload_bytes=estimated_workload_bytes,
            single_gpu_capacity_bytes=single_gpu_capacity_bytes,
            gates=gates,
            crossover=crossover,
            available_gpu_count=available_gpu_count,
        )

    def execute_production_mps(
        self,
        inputs: torch.Tensor | None = None,
        *,
        estimated_workload_bytes: int,
        single_gpu_capacity_bytes: int,
        gates: MPSAcceptanceGates | Mapping[str, Any],
        crossover: tuple[MPSCrossoverMeasurement | Mapping[str, Any], ...] = (),
        available_gpu_count: int = 0,
        max_bond: int | None = None,
        cutoff: float = 0.0,
    ) -> ExecutionResult:
        """Execute exactly the local or distributed path selected by the planner."""

        plan = self.plan_production_mps(
            inputs,
            estimated_workload_bytes=estimated_workload_bytes,
            single_gpu_capacity_bytes=single_gpu_capacity_bytes,
            gates=gates,
            crossover=crossover,
            available_gpu_count=available_gpu_count,
        )
        circuit = self._build(inputs, self._owned_parameters())
        ir = ensure_circuit_ir(circuit)
        wires = self.policy.observable_wires
        if self.policy.observable != "z" or len(wires) != 1:
            raise NotImplementedError(
                "production MPS integration currently supports one Z observable"
            )
        if plan.world_size == 1:
            from ..simulation.mps import run_mps

            state = run_mps(
                ir,
                device=self._parameter_tensors()[0].device,
                dtype=getattr(torch, self.precision.complex_dtype),
                max_bond=max_bond,
                cutoff=cutoff,
            )
            value = state.expectation_z(wires)[..., 0]
            runtime = {
                **plan.summary(),
                "executor": "pytorch_native_local_mps",
                "full_mps_materialization": False,
            }
        else:
            import torch.distributed as dist

            from .backends.mps.reverse import execute_torch_distributed_mps_reverse

            if not dist.is_initialized():
                raise RuntimeError(
                    "measured distributed MPS plan requires an initialized process group"
                )
            if dist.get_world_size(self._state_process_group) != plan.world_size:
                raise RuntimeError(
                    "active MPS process-group size does not match measured plan"
                )
            reverse = execute_torch_distributed_mps_reverse(
                ir,
                observable={wires[0]: "z"},
                device=self._parameter_tensors()[0].device,
                max_bond=max_bond,
                cutoff=cutoff,
            )
            value = reverse.value.reshape(1)
            runtime = {
                **reverse.summary(),
                "production_plan": plan.summary(),
                "production_promotion_allowed": plan.production_promotion_allowed,
            }
        return ExecutionResult(
            value=value,
            plan=plan,
            metrics={
                "training": self.training,
                "precision": self.precision.to_dict(),
                "mps_acceptance_gates": plan.gates.to_dict(),
            },
            provenance={"ir_hash": ir.content_hash},
            runtime=runtime,
            compatibility={"fallback_used": False, "selected_backend": "pytorch"},
        )

    def execute(
        self,
        inputs: torch.Tensor | None = None,
        parameters: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> ExecutionResult:
        try:
            return self._execute_impl(inputs, parameters)
        finally:
            # Do not retain the current autograd graph between training steps.
            self._builder_bindings.clear()

    def _execute_impl(
        self,
        inputs: torch.Tensor | None = None,
        parameters: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> ExecutionResult:
        selected_parameters = (
            self._owned_parameters() if parameters is None else parameters
        )
        circuit = self._build(inputs, selected_parameters)
        ir = ensure_circuit_ir(circuit)
        self._last_ir = detached_ir_snapshot(ir)
        program_runtime = self._observe_topology(ir)
        requested_backend = self.policy.backend
        selected_backend = requested_backend
        compatibility: dict[str, Any] = {"fallback_used": False}
        wires = self.policy.observable_wires or (0,)
        if requested_backend == "jax":
            try:
                if self.policy.observable == "z" and len(wires) > 1:
                    raise NotImplementedError(
                        "JAX fq.Module does not yet support vector-valued Z "
                        "observables; use the PyTorch local fast path"
                    )
                if isinstance(selected_parameters, Mapping):
                    raise NotImplementedError(
                        "JAX execution of named parameter groups requires an "
                        "explicit flattening policy"
                    )
                accepts_inputs = (
                    len(
                        tuple(
                            item
                            for item in inspect.signature(
                                self.circuit_builder
                            ).parameters.values()
                            if item.kind
                            in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            )
                        )
                    )
                    >= 2
                )
                kernel_signature = (
                    tuple(selected_parameters.shape),
                    selected_parameters.dtype,
                    selected_parameters.device,
                    None if inputs is None else tuple(inputs.shape),
                    None if inputs is None else inputs.dtype,
                    None if inputs is None else inputs.device,
                    self.precision.complex_dtype,
                    self.policy,
                )
                if (
                    self._jax_kernel is None
                    or self._jax_kernel_signature != kernel_signature
                ):
                    from .backends.jax import compile_quantum_kernel

                    if not isinstance(circuit, Circuit):
                        raise TypeError("JAX module builder must return a Circuit")
                    self._jax_kernel = compile_quantum_kernel(
                        self.circuit_builder,
                        selected_parameters.detach(),
                        backend="jax",
                        interface="torch",
                        mode=self.policy.mode,
                        n_wires=circuit.n_wires,
                        observable=self.policy.observable,
                        observable_wires=wires,
                        hamiltonian=self.hamiltonian,
                        compute_dtype=self.precision.complex_dtype,
                        accepts_inputs=accepts_inputs,
                    )
                    self._jax_kernel_signature = kernel_signature
                value = self._jax_kernel(selected_parameters, inputs)
                if value.ndim == 0:
                    value = value.reshape(1)
                runtime = {**self._jax_kernel.summary(), **program_runtime}
                return ExecutionResult(
                    value=value,
                    metrics={
                        "training": self.training,
                        "precision": self.precision.to_dict(),
                    },
                    provenance={"ir_hash": ir.content_hash},
                    runtime=runtime,
                    compatibility={
                        "fallback_used": False,
                        "requested_backend": "jax",
                        "selected_backend": "jax",
                    },
                )
            except Exception as error:
                if not self.policy.allow_backend_fallback:
                    raise
                selected_backend = "pytorch"
                compatibility = {
                    "fallback_used": True,
                    "requested_backend": requested_backend,
                    "selected_backend": selected_backend,
                    "reason": f"jax_kernel_unavailable:{type(error).__name__}",
                }
        distributed = self.policy.mode == "distributed_statevector"
        if distributed:
            import torch.distributed as dist

            from .backends.statevector import (
                execute_torch_distributed_statevector_reverse,
            )

            if self.policy.observable != "z" or len(wires) != 1:
                raise NotImplementedError(
                    "distributed_statevector currently supports one Z observable; "
                    "observable batching must use explicit term groups"
                )
            if not dist.is_initialized() and int(os.getenv("WORLD_SIZE", "1")) > 1:
                raise RuntimeError(
                    "distributed_statevector requires an initialized process group"
                )
            reverse = execute_torch_distributed_statevector_reverse(
                ir,
                observable_wire=wires[0],
                device=self._parameter_tensors()[0].device,
                process_group=self._state_process_group,
            )
            value = reverse.value.reshape(1)
            state = None
            runtime = LiveRuntimeSummary(reverse)
        else:
            if not isinstance(circuit, Circuit):
                raise TypeError(
                    "local fq.Module execution requires a Circuit builder result"
                )
            if self.policy.mode == "statevector":
                state = circuit.state(refresh=True)
                program_runtime = {
                    **program_runtime,
                    **getattr(circuit, "_last_statevector_runtime", {}),
                }
                values = circuit.expectation_z(wires)
                executor = "pytorch_native_module_v1"
            elif self.policy.mode == "mps":
                from ..simulation.mps import run_mps

                state = None
                backend_state = run_mps(
                    circuit,
                    max_bond=self.policy.mps_max_bond,
                    cutoff=self.policy.mps_cutoff,
                )
                values = backend_state.expectation_z(wires)
                program_runtime = {**program_runtime, **backend_state.summary()}
                executor = "pytorch_native_mps"
            elif self.policy.mode == "tensor_network":
                from ..simulation.tensor import run_tensor_network

                state = None
                backend_state = run_tensor_network(circuit, dense_observable_wires=12)
                values = backend_state.expectation_z(wires)
                executor = "pytorch_native_tensor_network"
            else:
                raise RuntimeError(f"unhandled fq.Module mode {self.policy.mode!r}")
            if self.policy.observable == "hamiltonian":
                if self.policy.mode not in {"statevector", "mps"}:
                    raise NotImplementedError(
                        f"{self.policy.mode} Module does not yet support Hamiltonian "
                        "observables without explicit term execution"
                    )
                if self.hamiltonian is None:
                    raise ValueError("hamiltonian observable requires a Hamiltonian")
                value = self.hamiltonian.expectation(
                    circuit if self.policy.mode == "statevector" else backend_state
                )
            else:
                value = (
                    values.sum(dim=-1)
                    if self.policy.observable == "z_sum"
                    else values if len(wires) > 1 else values[..., 0]
                )
            runtime = {
                "executor": executor,
                "backend": selected_backend,
                "mode": self.policy.mode,
                "distribution_semantics": "single_device_fast_path",
                "world_size": 1,
                "local_world_size": 1,
                "node_count": 1,
                **program_runtime,
            }
        input_batch_size = 1 if inputs is None or inputs.ndim == 0 else inputs.shape[0]
        parameter_batch_size = 1
        if isinstance(selected_parameters, torch.Tensor):
            parameter_batch_size = (
                selected_parameters.shape[0] if selected_parameters.ndim > 1 else 1
            )
        observable_count = (
            self.hamiltonian.n_terms
            if self.policy.observable == "hamiltonian" and self.hamiltonian is not None
            else len(wires)
        )
        if self.policy.observable == "hamiltonian" and self.hamiltonian is not None:
            from .parallel import group_observables

            observable_group_count = len(group_observables(self.hamiltonian.terms))
        else:
            observable_group_count = 1
        parallel = (
            self._parallel_plan.summary()
            if self._parallel_plan
            else {
                "parallel_dimensions": {
                    "input_batch": input_batch_size,
                    "parameter_batch": parameter_batch_size,
                    "observable_batch": observable_count,
                    "data_parallel": 1,
                    "state_parallel": reverse.world_size if distributed else 1,
                    "model_parallel": 1,
                },
                "distribution_semantics": (
                    "sharded_across_ranks"
                    if distributed and reverse.world_size > 1
                    else "single_device_fast_path"
                ),
            }
        )
        result = ExecutionResult(
            value=value,
            state=state,
            metrics={
                "training": self.training,
                "input_batch_size": input_batch_size,
                "parameter_batch_size": parameter_batch_size,
                "observable_count": observable_count,
                "observable_group_count": observable_group_count,
                "parallelism": parallel,
                "precision": self.precision.to_dict(),
            },
            provenance={"ir_hash": ir.content_hash},
            runtime=runtime,
            compatibility=compatibility,
        )
        if self.policy.correctness_debug:
            from .training_state import assert_finite_training

            assert_finite_training(self, value=result.value, include_gradients=False)
        return result

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        parameters: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.policy.backend == "pytorch" and self.policy.mode in {
            "statevector",
            "mps",
            "tensor_network",
        }:
            return self._forward_local_tensor(inputs, parameters)
        result = self.execute(inputs, parameters)
        assert result.value is not None
        return result.value

    def _forward_local_tensor(
        self,
        inputs: torch.Tensor | None,
        parameters: torch.Tensor | Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Execute the local training fast path without result metadata."""

        selected_parameters = (
            self._owned_parameters() if parameters is None else parameters
        )
        try:
            circuit = self._build(inputs, selected_parameters)
            if not isinstance(circuit, Circuit):
                raise TypeError(
                    "local fq.Module execution requires a Circuit builder result"
                )
            self._last_ir = detached_ir_snapshot(ensure_circuit_ir(circuit))
            wires = self.policy.observable_wires or (0,)
            backend_state: Any = None
            if self.policy.mode == "statevector":
                circuit.state(refresh=True)
                values = circuit.expectation_z(wires)
            elif self.policy.mode == "mps":
                from ..simulation.mps import run_mps

                backend_state = run_mps(
                    circuit,
                    max_bond=self.policy.mps_max_bond,
                    cutoff=self.policy.mps_cutoff,
                )
                values = backend_state.expectation_z(wires)
            else:
                from ..simulation.tensor import run_tensor_network

                backend_state = run_tensor_network(circuit, dense_observable_wires=12)
                values = backend_state.expectation_z(wires)

            if self.policy.observable == "hamiltonian":
                if self.policy.mode == "tensor_network":
                    raise NotImplementedError(
                        "tensor_network Module does not yet support Hamiltonian "
                        "observables without explicit term execution"
                    )
                if self.hamiltonian is None:
                    raise ValueError("hamiltonian observable requires a Hamiltonian")
                value = self.hamiltonian.expectation(
                    circuit if self.policy.mode == "statevector" else backend_state
                )
            elif self.policy.observable == "z_sum":
                value = values.sum(dim=-1)
            else:
                value = values if len(wires) > 1 else values[..., 0]

            if self.policy.correctness_debug:
                from .training_state import assert_finite_training

                assert_finite_training(self, value=value, include_gradients=False)
            return value
        finally:
            self._builder_bindings.clear()

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "policy": self.policy.__dict__,
            "deployment_binding": self.deployment_binding,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.set_runtime_policy(RuntimePolicy(**dict(state.get("policy", {}))))
        self.deployment_binding = dict(state.get("deployment_binding", {}))

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        optimizer: torch.optim.Optimizer | None,
        seed: SeedContract,
        runtime_plan: Any | None = None,
        step: int = 0,
    ) -> Any:
        from .training_state import save_training_checkpoint

        return save_training_checkpoint(
            path,
            module=self,
            optimizer=optimizer,
            seed=seed,
            precision=self.precision,
            runtime_plan=runtime_plan,
            step=step,
        )

    def load_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        optimizer: torch.optim.Optimizer | None,
    ) -> dict[str, Any]:
        from .training_state import load_training_checkpoint

        return cast(
            dict[str, Any],
            load_training_checkpoint(
                path,
                module=self,
                optimizer=optimizer,
                precision=self.precision,
            ),
        )


__all__ = [
    "ExecutionResult",
    "Module",
    "RuntimePolicy",
    "normalize_execution_result",
]
