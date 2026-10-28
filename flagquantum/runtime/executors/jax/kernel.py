"""Canonical JAX kernels exposed through the PyTorch interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import prod
from typing import Any, Protocol

import torch
from torch.autograd.function import once_differentiable

from ....simulation.jax.mps.kernels import (
    is_zz_z_chain_hamiltonian as _is_zz_z_chain_hamiltonian,
)
from ....simulation.jax.mps.kernels import (
    jax_mps_hamiltonian_expectation as _jax_mps_hamiltonian_expectation,
)
from ....simulation.jax.mps.kernels import (
    jax_mps_z_sum as _jax_mps_z_sum,
)
from ....simulation.jax.mps.kernels import (
    jax_mps_z_values as _jax_mps_z_values,
)
from ....simulation.jax.primitives import (
    _jax_hamiltonian_expectation,
    _jax_real_dtype,
    _jax_z_sum,
    _jax_z_values,
    _set_active_jax_compute_dtype,
    jax_evaluate_declared_batch,
)
from ....simulation.jax.statevector.kernels import (
    jax_statevector_from_circuit as _jax_statevector_from_circuit,
)
from ....simulation.jax.tensor_network.kernels import (
    jax_tensor_network_hamiltonian_expectation as _jax_tensor_network_hamiltonian_expectation,
)
from ....simulation.jax.tensor_network.kernels import (
    jax_tensor_network_z_sum as _jax_tensor_network_z_sum,
)
from ....simulation.jax.tensor_network.kernels import (
    jax_tensor_network_z_values as _jax_tensor_network_z_values,
)

TorchCircuitBuilder = Callable[..., Any]


class _JAXAutogradContext(Protocol):
    needs_parameter_gradient: bool
    needs_input_gradient: bool

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _torch_to_jax(parameters: torch.Tensor) -> Any:
    import jax.dlpack

    # CPU JIT accepts NumPy buffers directly. Going through JAX's DLPack device
    # discovery on each small tensor costs more than these local circuits.
    # Keep DLPack for accelerators and dtypes NumPy cannot represent.
    if parameters.device.type == "cpu" and parameters.dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
    }:
        default_device = getattr(jax.config, "jax_default_device")
        cpu_default = (default_device is None and jax.default_backend() == "cpu") or (
            default_device is not None and default_device.platform == "cpu"
        )
        if cpu_default:
            return parameters.detach().contiguous().numpy()
    return jax.dlpack.from_dlpack(parameters.detach().contiguous())


def _jax_to_torch(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    tensor = torch.utils.dlpack.from_dlpack(value)
    return tensor.to(device=like.device, dtype=like.dtype)


class _TorchJAXQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _JAXAutogradContext,
        params: torch.Tensor,
        input_values: torch.Tensor,
        kernel: JAXQuantumKernel,
    ) -> torch.Tensor:
        jax_params = _torch_to_jax(params)
        ctx.needs_parameter_gradient = params.requires_grad
        torch_grad = None
        if kernel.accepts_inputs:
            jax_inputs = _torch_to_jax(input_values)
            if input_values.requires_grad:
                if params.requires_grad:
                    value, (param_grad, input_grad) = kernel._value_and_grad(
                        jax_params, jax_inputs
                    )
                    torch_grad = _jax_to_torch(param_grad, like=params)
                else:
                    value, input_grad = kernel._input_value_and_grad(
                        jax_params, jax_inputs
                    )
                torch_input_grad = _jax_to_torch(input_grad, like=input_values)
            else:
                value, param_grad = kernel._parameter_value_and_grad(
                    jax_params, jax_inputs
                )
                torch_grad = _jax_to_torch(param_grad, like=params)
                torch_input_grad = None
        else:
            value, param_grad = kernel._value_and_grad(jax_params)
            torch_grad = _jax_to_torch(param_grad, like=params)
            torch_input_grad = None
        ctx.save_for_backward(
            *(grad for grad in (torch_grad, torch_input_grad) if grad is not None)
        )
        ctx.needs_input_gradient = torch_input_grad is not None
        return _jax_to_torch(value, like=params)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: _JAXAutogradContext, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        parameter_vjp = None
        if ctx.needs_parameter_gradient:
            grad = ctx.saved_tensors[0]
            scale_shape = tuple(grad_output.shape) + (1,) * max(
                0, grad.ndim - grad_output.ndim
            )
            parameter_vjp = grad_output.reshape(scale_shape) * grad
        if not ctx.needs_input_gradient:
            return parameter_vjp, None, None
        input_grad = ctx.saved_tensors[int(ctx.needs_parameter_gradient)]
        input_scale_shape = tuple(grad_output.shape) + (1,) * max(
            0, input_grad.ndim - grad_output.ndim
        )
        input_vjp = grad_output.reshape(input_scale_shape) * input_grad
        return parameter_vjp, input_vjp, None


class _JAXVJPAutogradContext(_JAXAutogradContext, Protocol):
    kernel: JAXQuantumKernel


class _TorchJAXVJPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _JAXVJPAutogradContext,
        params: torch.Tensor,
        inputs: torch.Tensor,
        kernel: JAXQuantumKernel,
    ) -> torch.Tensor:
        frozen_params, frozen_inputs = params.detach().clone(), inputs.detach().clone()
        ctx.save_for_backward(frozen_params, frozen_inputs)
        ctx.kernel = kernel
        ctx.needs_parameter_gradient = params.requires_grad
        ctx.needs_input_gradient = inputs.requires_grad
        value = kernel._value(
            _torch_to_jax(frozen_params), _torch_to_jax(frozen_inputs)
        )
        return _jax_to_torch(value, like=params)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: _JAXVJPAutogradContext,
        upstream: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        params, inputs = ctx.saved_tensors
        demand = (ctx.needs_parameter_gradient, ctx.needs_input_gradient)
        result = ctx.kernel._cotangent_vjps[demand](
            _torch_to_jax(params), _torch_to_jax(inputs), _torch_to_jax(upstream)
        )
        if all(demand):
            return (
                _jax_to_torch(result[0], like=params),
                _jax_to_torch(result[1], like=inputs),
                None,
            )
        if demand[0]:
            return _jax_to_torch(result, like=params), None, None
        return None, _jax_to_torch(result, like=inputs), None


@dataclass
class JAXQuantumKernel:
    """JAX quantum kernel exposed as a PyTorch differentiable callable."""

    circuit_builder: TorchCircuitBuilder
    n_qubits: int
    mode: str = "statevector"
    observable: str = "z_sum"
    observable_qubits: tuple[int, ...] | None = None
    hamiltonian_terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...] = ()
    parameter_shape: tuple[int, ...] | None = None
    accepts_inputs: bool = False
    jit: bool = True
    matmul_precision: str | None = "highest"
    compute_dtype: str = "complex64"
    max_bond: int | None = None
    cutoff: float = 0.0
    status: str = "ready"

    def __post_init__(self) -> None:
        if self.compute_dtype not in {"complex64", "complex128"}:
            raise ValueError("compute_dtype must be 'complex64' or 'complex128'.")
        if self.compute_dtype == "complex128":
            import jax

            update_config: Callable[[str, bool], None] = jax.config.update
            update_config("jax_enable_x64", True)
        self._value_and_grad = self._build_value_and_grad()
        self._value = self._compile(self._jax_loss)
        self._declared_batch_size: int | None = None
        self._batch_input_shape: tuple[Any, ...] | None = None
        self._vector_output = False
        self._statevector_diagnostics: dict[str, Any] = {}
        self._cotangent_vjps = {
            (True, False): self._build_cotangent_vjp(0),
            (False, True): self._build_cotangent_vjp(1),
            (True, True): self._build_cotangent_vjp((0, 1)),
        }
        if self.accepts_inputs:
            import jax

            parameter_gradient = jax.value_and_grad(self._jax_loss, argnums=0)
            self._parameter_value_and_grad = (
                jax.jit(parameter_gradient) if self.jit else parameter_gradient
            )
            input_gradient = jax.value_and_grad(self._jax_loss, argnums=1)
            self._input_value_and_grad = (
                jax.jit(input_gradient) if self.jit else input_gradient
            )
        else:
            self._parameter_value_and_grad = self._value_and_grad

    def __call__(
        self, parameters: torch.Tensor, inputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.accepts_inputs and inputs is None:
            raise ValueError("JAX quantum kernel circuit builder requires inputs")
        if not self.accepts_inputs and inputs is not None:
            raise ValueError(
                "JAX quantum kernel circuit builder does not accept inputs"
            )

        input_tensor = inputs if inputs is not None else parameters.new_empty(0)
        if not torch.is_grad_enabled() or not (
            parameters.requires_grad or input_tensor.requires_grad
        ):
            arguments: tuple[Any, ...] = (_torch_to_jax(parameters),)
            if self.accepts_inputs:
                arguments += (_torch_to_jax(input_tensor),)
            return _jax_to_torch(self._value(*arguments), like=parameters)
        if self.accepts_inputs and self._declared_batch_size != 1:
            if self._declared_batch_size is None:
                signature = (tuple(parameters.shape), tuple(input_tensor.shape))
                if signature != self._batch_input_shape:
                    import jax

                    value = jax.eval_shape(
                        self._jax_loss,
                        _torch_to_jax(parameters),
                        _torch_to_jax(input_tensor),
                    )
                    self._vector_output = value.ndim > 0
                    self._batch_input_shape = signature
            if self._vector_output or (self._declared_batch_size or 1) > 1:
                batch_apply: Callable[..., object] = _TorchJAXVJPFunction.apply
                result = batch_apply(parameters, input_tensor, self)
                if not isinstance(result, torch.Tensor):
                    raise TypeError("JAX autograd bridge must return a tensor")
                return result
        apply: Callable[[torch.Tensor, torch.Tensor, JAXQuantumKernel], object] = (
            _TorchJAXQuantumFunction.apply
        )
        result = apply(parameters, input_tensor, self)
        if not isinstance(result, torch.Tensor):
            raise TypeError("JAX autograd bridge must return a tensor")
        return result

    def value_and_grad(
        self, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self(parameters)
        grad = torch.autograd.grad(
            value, parameters, retain_graph=False, create_graph=False
        )[0]
        return value.detach(), grad.detach()

    def summary(self) -> dict[str, Any]:
        hamiltonian_fastpath = None
        if self.mode == "mps" and self.observable == "hamiltonian":
            hamiltonian_fastpath = (
                "local_pauli_zz_chain_padded_scan"
                if _is_zz_z_chain_hamiltonian(
                    self.hamiltonian_terms, int(self.n_qubits)
                )
                else "generic_pauli_string_transfer"
            )
        return {
            "executor": "jax_quantum_kernel",
            "interface": "torch",
            "backend": "jax",
            "distribution_semantics": "rank_local_replicated_kernel",
            "scalability_claim_allowed": False,
            "distributed_integration_role": "rank_local_jax_accelerator",
            "requires_jax_distributed_plan_for_capacity_scaling": True,
            "compatible_distributed_modes": (
                "distributed_statevector",
                "distributed_mps",
                "distributed_tensor_network",
            ),
            "scalability_note": (
                "The JAX quantum kernel runs independently inside each torchrun rank. "
                "It does not shard one statevector/MPS/TN workload across ranks."
            ),
            "mode": self.mode,
            "observable": self.observable,
            "observable_wires": self.observable_qubits,
            "hamiltonian_terms": self.hamiltonian_terms,
            "hamiltonian_fastpath": hamiltonian_fastpath,
            "parameter_shape": self.parameter_shape,
            "statevector_algorithm": dict(self._statevector_diagnostics),
            "declared_batch_size": self._declared_batch_size,
            "gradient_interface": (
                "cotangent_vjp"
                if self.accepts_inputs
                and (self._vector_output or (self._declared_batch_size or 1) > 1)
                else "precomputed_derivative"
            ),
            "jit": self.jit,
            "matmul_precision": self.matmul_precision,
            "compute_dtype": self.compute_dtype,
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "status": self.status,
        }

    def _build_value_and_grad(self) -> Any:
        import jax

        return self._compile(
            jax.value_and_grad(
                self._jax_loss, argnums=(0, 1) if self.accepts_inputs else 0
            )
        )

    def _build_cotangent_vjp(self, argnums: int | tuple[int, int]) -> Any:
        import jax
        import jax.numpy as jnp

        def derivative(parameters: Any, inputs: Any, upstream: Any) -> Any:
            def objective(values: Any, features: Any) -> Any:
                return jnp.sum(self._jax_loss(values, features) * upstream)

            return jax.grad(objective, argnums=argnums)(parameters, inputs)

        return jax.jit(derivative) if self.jit else derivative

    def _compile(self, function: Any) -> Any:
        import jax

        compiled = function
        if self.parameter_shape is not None and not self.accepts_inputs:
            parameter_shape = tuple(int(dim) for dim in self.parameter_shape)

            def with_optional_batch(parameters: Any) -> Any:
                shape = tuple(parameters.shape)
                batch_ndim = len(shape) - len(parameter_shape)
                if batch_ndim < 0 or shape[batch_ndim:] != parameter_shape:
                    raise ValueError(
                        f"parameter shape must end with {parameter_shape}; got {shape}"
                    )
                if batch_ndim == 0:
                    return function(parameters)
                batch_shape = shape[:batch_ndim]
                flat = parameters.reshape((prod(batch_shape),) + parameter_shape)
                result = jax.vmap(function)(flat)
                return jax.tree_util.tree_map(
                    lambda value: value.reshape(batch_shape + value.shape[1:]), result
                )

            compiled = with_optional_batch
        if self.jit:
            compiled = jax.jit(compiled)
        return compiled

    def _jax_loss(self, parameters: Any, inputs: Any | None = None) -> Any:
        import jax.numpy as jnp

        previous_dtype = _set_active_jax_compute_dtype(self.compute_dtype)
        try:
            parameters = jnp.asarray(parameters, dtype=_jax_real_dtype())
            circuit = (
                self.circuit_builder(
                    _JAXParameterProxy(parameters),
                    jnp.asarray(inputs, dtype=_jax_real_dtype()),
                )
                if self.accepts_inputs
                else self.circuit_builder(_JAXParameterProxy(parameters))
            )
            return jax_evaluate_declared_batch(
                circuit, lambda: self._jax_circuit_value(circuit, parameters)
            )
        finally:
            _set_active_jax_compute_dtype(previous_dtype)

    def _jax_circuit_value(self, circuit: Any, parameters: Any) -> Any:
        import jax.numpy as jnp

        if self.mode == "statevector":
            state = _jax_statevector_from_circuit(
                circuit,
                int(self.n_qubits),
                parameters,
                matmul_precision=self.matmul_precision,
                diagnostics=self._statevector_diagnostics,
            )
        elif self.mode == "mps":
            tensors = _jax_mps_from_circuit(
                circuit,
                int(self.n_qubits),
                parameters,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
                matmul_precision=self.matmul_precision,
            )
            if self.observable == "z_sum":
                qubits = self.observable_qubits or tuple(range(int(self.n_qubits)))
                return _jax_mps_z_sum(tensors, qubits, self.matmul_precision)
            if self.observable == "z":
                qubits = self.observable_qubits or tuple(range(int(self.n_qubits)))
                return _jax_mps_z_values(tensors, qubits, self.matmul_precision).sum()
            if self.observable == "hamiltonian":
                return _jax_mps_hamiltonian_expectation(
                    tensors,
                    self.hamiltonian_terms,
                    self.matmul_precision,
                )
            raise ValueError(
                "JAXQuantumKernel observable must be 'z_sum', 'z', or 'hamiltonian'."
            )
        elif self.mode in {"tensor_network", "tn"}:
            if self.observable == "z_sum":
                qubits = self.observable_qubits or tuple(range(int(self.n_qubits)))
                return _jax_tensor_network_z_sum(
                    circuit,
                    int(self.n_qubits),
                    qubits,
                    self.matmul_precision,
                )
            if self.observable == "z":
                qubits = self.observable_qubits or tuple(range(int(self.n_qubits)))
                return _jax_tensor_network_z_values(
                    circuit,
                    int(self.n_qubits),
                    qubits,
                    self.matmul_precision,
                ).sum()
            if self.observable == "hamiltonian":
                return _jax_tensor_network_hamiltonian_expectation(
                    circuit,
                    int(self.n_qubits),
                    self.hamiltonian_terms,
                    self.matmul_precision,
                )
            raise ValueError(
                "JAXQuantumKernel observable must be 'z_sum', 'z', or 'hamiltonian'."
            )
        else:
            raise ValueError(
                "JAXQuantumKernel mode must be 'statevector', 'mps', or 'tensor_network'."
            )
        if self.observable == "z_sum":
            qubits = self.observable_qubits or tuple(range(int(self.n_qubits)))
            return _jax_z_sum(state, int(self.n_qubits), qubits)
        if self.observable == "z":
            qubits = self.observable_qubits or tuple(range(int(self.n_qubits)))
            return jnp.sum(_jax_z_values(state, int(self.n_qubits), qubits))
        if self.observable == "hamiltonian":
            return _jax_hamiltonian_expectation(
                state,
                int(self.n_qubits),
                self.hamiltonian_terms,
                self.matmul_precision,
            )
        raise ValueError(
            "JAXQuantumKernel observable must be 'z_sum', 'z', or 'hamiltonian'."
        )


class _JAXParameterProxy:
    def __init__(self, parameters: Any) -> None:
        self.parameters = parameters

    def __getitem__(self, item: Any) -> Any:
        return self.parameters[item]

    @property
    def shape(self) -> Any:
        return self.parameters.shape

    @property
    def device(self) -> torch.device:
        # Circuit construction needs a Torch device for its lazy container;
        # numerical gate values remain JAX tracers and are never copied there.
        return torch.device("cpu")

    def reshape(self, *shape: Any) -> Any:
        return self.parameters.reshape(*shape)


from .mps.lowering import _jax_mps_from_circuit  # noqa: E402


def compile_quantum_kernel(
    circuit_builder: TorchCircuitBuilder,
    example_parameters: torch.Tensor | None = None,
    *,
    backend: str = "jax",
    interface: str = "torch",
    mode: str = "statevector",
    n_qubits: int | None = None,
    observable: str = "z_sum",
    observable_qubits: Iterable[int] | int | None = None,
    hamiltonian: Any | None = None,
    jit: bool = True,
    matmul_precision: str | None = "highest",
    compute_dtype: str = "complex64",
    max_bond: int | None = None,
    cutoff: float = 0.0,
    accepts_inputs: bool = False,
) -> JAXQuantumKernel:
    """Compile a quantum kernel for use inside a PyTorch training loop."""

    if backend != "jax" or interface != "torch":
        raise ValueError(
            "Only backend='jax', interface='torch' is currently supported."
        )
    if mode == "tn":
        mode = "tensor_network"
    if mode not in {"statevector", "mps", "tensor_network"}:
        raise ValueError("mode must be 'statevector', 'mps', or 'tensor_network'.")
    qubits: tuple[int, ...] | None
    if isinstance(observable_qubits, int):
        qubits = (int(observable_qubits),)
    elif observable_qubits is None:
        qubits = None
    else:
        qubits = tuple(int(qubit) for qubit in observable_qubits)
    hamiltonian_terms = _normalize_hamiltonian_terms(hamiltonian)
    if hamiltonian is not None:
        observable = "hamiltonian"
    if n_qubits is None:
        if example_parameters is None:
            raise ValueError("n_qubits or example_parameters must be provided.")
        if accepts_inputs:
            raise ValueError("n_qubits is required when circuit_builder accepts inputs")
        probe = circuit_builder(example_parameters.detach())
        n_qubits = int(probe.n_qubits)
    parameter_shape = None
    if example_parameters is not None:
        parameter_shape = tuple(int(dim) for dim in example_parameters.shape)
    return JAXQuantumKernel(
        circuit_builder=circuit_builder,
        n_qubits=int(n_qubits),
        mode=mode,
        observable=observable,
        observable_qubits=qubits,
        hamiltonian_terms=hamiltonian_terms,
        parameter_shape=parameter_shape,
        accepts_inputs=accepts_inputs,
        jit=jit,
        matmul_precision=matmul_precision,
        compute_dtype=compute_dtype,
        max_bond=max_bond,
        cutoff=float(cutoff),
    )


def _normalize_hamiltonian_terms(
    hamiltonian: Any | None,
) -> tuple[tuple[float, tuple[tuple[int, str], ...]], ...]:
    if hamiltonian is None:
        return ()
    terms = getattr(hamiltonian, "terms", hamiltonian)
    normalized = []
    for term in terms:
        coefficient = getattr(term, "coefficient", None)
        ops = getattr(term, "ops", None)
        if coefficient is None or ops is None:
            raise ValueError("Hamiltonian terms must expose coefficient and ops.")
        if isinstance(coefficient, torch.Tensor):
            if coefficient.numel() != 1:
                raise ValueError(
                    "Hybrid JAX Hamiltonian coefficients must be scalar values."
                )
            coeff_value = float(torch.real(coefficient.detach()).cpu().reshape(()))
        else:
            coeff_value = float(coefficient)
        normalized.append(
            (
                coeff_value,
                tuple((int(qubit), str(name).lower()) for qubit, name in ops),
            )
        )
    if not normalized:
        raise ValueError("Hamiltonian observable requires at least one term.")
    return tuple(normalized)


class QuantumTorchLayer(torch.nn.Module):
    """PyTorch module backed by a FlagQuantum hybrid quantum kernel."""

    def __init__(
        self,
        circuit_builder: TorchCircuitBuilder,
        n_parameters: int | tuple[int, ...],
        *,
        backend: str = "jax",
        quantum_backend: str | None = None,
        mode: str = "statevector",
        n_qubits: int,
        observable: str = "z_sum",
        observable_qubits: Iterable[int] | int | None = None,
        hamiltonian: Any | None = None,
        jit: bool = True,
        matmul_precision: str | None = "highest",
        compute_dtype: str = "complex64",
        init: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        selected_backend = quantum_backend or backend
        shape = (
            (int(n_parameters),)
            if isinstance(n_parameters, int)
            else tuple(int(v) for v in n_parameters)
        )
        if init is None:
            values = torch.zeros(shape, dtype=dtype, device=device)
        else:
            values = init.detach().clone().to(dtype=dtype, device=device).reshape(shape)
        self.parameters_tensor = torch.nn.Parameter(values)
        self.kernel = compile_quantum_kernel(
            circuit_builder,
            self.parameters_tensor.detach(),
            backend=selected_backend,
            interface="torch",
            mode=mode,
            n_qubits=n_qubits,
            observable=observable,
            observable_qubits=observable_qubits,
            hamiltonian=hamiltonian,
            jit=jit,
            matmul_precision=matmul_precision,
            compute_dtype=compute_dtype,
        )

    def forward(self, parameters: torch.Tensor | None = None) -> torch.Tensor:
        return self.kernel(self.parameters_tensor if parameters is None else parameters)

    def summary(self) -> dict[str, Any]:
        data = self.kernel.summary()
        data.update(
            {
                "module": "QuantumTorchLayer",
                "parameter_shape": tuple(self.parameters_tensor.shape),
            }
        )
        return data


__all__ = ["JAXQuantumKernel", "QuantumTorchLayer", "compile_quantum_kernel"]
