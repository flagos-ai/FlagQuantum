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
    _jax_statevector_from_circuit,
    _jax_z_sum,
    _jax_z_values,
    _set_active_jax_compute_dtype,
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
    accepts_inputs: bool

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _torch_to_jax(parameters: torch.Tensor) -> Any:
    import jax.dlpack

    return jax.dlpack.from_dlpack(parameters.detach().contiguous())


def _jax_to_torch(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    tensor = torch.utils.dlpack.from_dlpack(value)
    return tensor.to(device=like.device, dtype=like.dtype)


@dataclass
class JAXQuantumKernel:
    """JAX quantum kernel exposed as a PyTorch differentiable callable."""

    circuit_builder: TorchCircuitBuilder
    n_wires: int
    mode: str = "statevector"
    observable: str = "z_sum"
    observable_wires: tuple[int, ...] | None = None
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

    def __call__(
        self, parameters: torch.Tensor, inputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        kernel = self

        if self.accepts_inputs and inputs is None:
            raise ValueError("JAX quantum kernel circuit builder requires inputs")
        if not self.accepts_inputs and inputs is not None:
            raise ValueError(
                "JAX quantum kernel circuit builder does not accept inputs"
            )

        class _TorchJAXQuantumFunction(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: _JAXAutogradContext,
                params: torch.Tensor,
                input_values: torch.Tensor,
            ) -> torch.Tensor:
                jax_params = _torch_to_jax(params)
                if kernel.accepts_inputs:
                    jax_inputs = _torch_to_jax(input_values)
                    value, (param_grad, input_grad) = kernel._value_and_grad(
                        jax_params, jax_inputs
                    )
                    torch_input_grad = _jax_to_torch(input_grad, like=input_values)
                else:
                    value, param_grad = kernel._value_and_grad(jax_params)
                    torch_input_grad = torch.zeros_like(input_values)
                torch_grad = _jax_to_torch(param_grad, like=params)
                ctx.save_for_backward(torch_grad, torch_input_grad)
                ctx.accepts_inputs = kernel.accepts_inputs
                return _jax_to_torch(value, like=params)

            @staticmethod
            @once_differentiable
            def backward(
                ctx: _JAXAutogradContext, grad_output: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
                grad, input_grad = ctx.saved_tensors
                scale_shape = tuple(grad_output.shape) + (1,) * max(
                    0, grad.ndim - grad_output.ndim
                )
                parameter_vjp = grad_output.reshape(scale_shape) * grad
                if not ctx.accepts_inputs:
                    return parameter_vjp, None
                input_scale_shape = tuple(grad_output.shape) + (1,) * max(
                    0, input_grad.ndim - grad_output.ndim
                )
                input_vjp = grad_output.reshape(input_scale_shape) * input_grad
                return parameter_vjp, input_vjp

        input_tensor = inputs if inputs is not None else parameters.new_empty(0)
        apply: Callable[[torch.Tensor, torch.Tensor], object] = (
            _TorchJAXQuantumFunction.apply
        )
        result = apply(parameters, input_tensor)
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
                if _is_zz_z_chain_hamiltonian(self.hamiltonian_terms, int(self.n_wires))
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
            "observable_wires": self.observable_wires,
            "hamiltonian_terms": self.hamiltonian_terms,
            "hamiltonian_fastpath": hamiltonian_fastpath,
            "parameter_shape": self.parameter_shape,
            "jit": self.jit,
            "matmul_precision": self.matmul_precision,
            "compute_dtype": self.compute_dtype,
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "status": self.status,
        }

    def _build_value_and_grad(self) -> Any:
        import jax

        base_value_and_grad = jax.value_and_grad(
            self._jax_loss, argnums=(0, 1) if self.accepts_inputs else 0
        )
        value_and_grad = base_value_and_grad
        if self.parameter_shape is not None and not self.accepts_inputs:
            parameter_shape = tuple(int(dim) for dim in self.parameter_shape)

            def value_and_grad_with_optional_batch(parameters: Any) -> Any:
                shape = tuple(parameters.shape)
                batch_ndim = len(shape) - len(parameter_shape)
                if batch_ndim < 0 or shape[batch_ndim:] != parameter_shape:
                    raise ValueError(
                        f"parameter shape must end with {parameter_shape}; got {shape}"
                    )
                if batch_ndim == 0:
                    return base_value_and_grad(parameters)
                batch_shape = shape[:batch_ndim]
                flat = parameters.reshape((prod(batch_shape),) + parameter_shape)
                values, grads = jax.vmap(base_value_and_grad)(flat)
                return values.reshape(batch_shape), grads.reshape(parameters.shape)

            value_and_grad = value_and_grad_with_optional_batch
        if self.jit:
            value_and_grad = jax.jit(value_and_grad)
        return value_and_grad

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
            if self.mode == "statevector":
                state = _jax_statevector_from_circuit(
                    circuit,
                    int(self.n_wires),
                    parameters,
                    matmul_precision=self.matmul_precision,
                )
            elif self.mode == "mps":
                tensors = _jax_mps_from_circuit(
                    circuit,
                    int(self.n_wires),
                    parameters,
                    max_bond=self.max_bond,
                    cutoff=self.cutoff,
                    matmul_precision=self.matmul_precision,
                )
                if self.observable == "z_sum":
                    wires = self.observable_wires or tuple(range(int(self.n_wires)))
                    return _jax_mps_z_sum(tensors, wires, self.matmul_precision)
                if self.observable == "z":
                    wires = self.observable_wires or tuple(range(int(self.n_wires)))
                    return _jax_mps_z_values(
                        tensors, wires, self.matmul_precision
                    ).sum()
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
                    wires = self.observable_wires or tuple(range(int(self.n_wires)))
                    return _jax_tensor_network_z_sum(
                        circuit,
                        int(self.n_wires),
                        wires,
                        self.matmul_precision,
                    )
                if self.observable == "z":
                    wires = self.observable_wires or tuple(range(int(self.n_wires)))
                    return _jax_tensor_network_z_values(
                        circuit,
                        int(self.n_wires),
                        wires,
                        self.matmul_precision,
                    ).sum()
                if self.observable == "hamiltonian":
                    return _jax_tensor_network_hamiltonian_expectation(
                        circuit,
                        int(self.n_wires),
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
                wires = self.observable_wires or tuple(range(int(self.n_wires)))
                return _jax_z_sum(state, int(self.n_wires), wires)
            if self.observable == "z":
                wires = self.observable_wires or tuple(range(int(self.n_wires)))
                return jnp.sum(_jax_z_values(state, int(self.n_wires), wires))
            if self.observable == "hamiltonian":
                return _jax_hamiltonian_expectation(
                    state,
                    int(self.n_wires),
                    self.hamiltonian_terms,
                    self.matmul_precision,
                )
            raise ValueError(
                "JAXQuantumKernel observable must be 'z_sum', 'z', or 'hamiltonian'."
            )
        finally:
            _set_active_jax_compute_dtype(previous_dtype)


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
    n_wires: int | None = None,
    observable: str = "z_sum",
    observable_wires: Iterable[int] | int | None = None,
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
    wires: tuple[int, ...] | None
    if isinstance(observable_wires, int):
        wires = (int(observable_wires),)
    elif observable_wires is None:
        wires = None
    else:
        wires = tuple(int(wire) for wire in observable_wires)
    hamiltonian_terms = _normalize_hamiltonian_terms(hamiltonian)
    if hamiltonian is not None:
        observable = "hamiltonian"
    if n_wires is None:
        if example_parameters is None:
            raise ValueError("n_wires or example_parameters must be provided.")
        if accepts_inputs:
            raise ValueError("n_wires is required when circuit_builder accepts inputs")
        probe = circuit_builder(example_parameters.detach())
        n_wires = int(probe.n_wires)
    parameter_shape = None
    if example_parameters is not None:
        parameter_shape = tuple(int(dim) for dim in example_parameters.shape)
    return JAXQuantumKernel(
        circuit_builder=circuit_builder,
        n_wires=int(n_wires),
        mode=mode,
        observable=observable,
        observable_wires=wires,
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
                tuple((int(wire), str(name).lower()) for wire, name in ops),
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
        n_wires: int,
        observable: str = "z_sum",
        observable_wires: Iterable[int] | int | None = None,
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
            n_wires=n_wires,
            observable=observable,
            observable_wires=observable_wires,
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
