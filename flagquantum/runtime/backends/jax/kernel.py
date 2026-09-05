"""Canonical JAX kernels exposed through the PyTorch interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch

TorchCircuitBuilder = Callable[..., Any]


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

            jax.config.update("jax_enable_x64", True)
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
                ctx: Any, params: torch.Tensor, input_values: torch.Tensor
            ) -> torch.Tensor:
                import jax

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
                del jax
                torch_grad = _jax_to_torch(param_grad, like=params)
                ctx.save_for_backward(torch_grad, torch_input_grad)
                ctx.accepts_inputs = kernel.accepts_inputs
                return _jax_to_torch(value, like=params)

            @staticmethod
            def backward(
                ctx: Any, grad_output: torch.Tensor
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
        return _TorchJAXQuantumFunction.apply(parameters, input_tensor)

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
                if tuple(parameters.shape) == parameter_shape:
                    return base_value_and_grad(parameters)
                batch_shape = parameters.shape[
                    : len(parameters.shape) - len(parameter_shape)
                ]
                flat = parameters.reshape((-1,) + parameter_shape)
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


def _jax_statevector_from_circuit(
    circuit: Any,
    n_wires: int,
    parameters: Any,
    *,
    matmul_precision: str | None = "highest",
) -> Any:
    import jax.numpy as jnp

    state = jnp.zeros((2 ** int(n_wires),), dtype=_jax_complex_dtype())
    state = state.at[0].set(1.0 + 0.0j)
    for instruction in circuit.to_ir():
        state = _apply_jax_instruction(
            state,
            instruction,
            int(n_wires),
            parameters,
            matmul_precision,
        )
    return state


def _jax_instruction_matrix(instruction: Any) -> Any:
    name = instruction.name
    if name == "rx":
        return _jax_rx(_jax_scalar_param(instruction.params["theta"]))
    if name == "ry":
        return _jax_ry(_jax_scalar_param(instruction.params["theta"]))
    if name == "rz":
        return _jax_rz(_jax_scalar_param(instruction.params["theta"]))
    if name in {"phase", "u1"}:
        return _jax_phase(_jax_scalar_param(instruction.params["theta"]))
    if name == "u2":
        return _jax_u2(
            _jax_scalar_param(instruction.params["phi"]),
            _jax_scalar_param(instruction.params["lbd"]),
        )
    if name == "u3":
        return _jax_u3(
            _jax_scalar_param(instruction.params["theta"]),
            _jax_scalar_param(instruction.params["phi"]),
            _jax_scalar_param(instruction.params["lbd"]),
        )
    if name == "h":
        return _jax_h()
    if name == "x":
        return _jax_x()
    if name == "y":
        return _jax_y()
    if name == "z":
        return _jax_z()
    if name == "s":
        return _jax_s()
    if name == "sdg":
        return _jax_sdg()
    if name == "t":
        return _jax_t()
    if name == "tdg":
        return _jax_tdg()
    if name == "sx":
        return _jax_sx()
    if name == "sxdg":
        return _jax_sxdg()
    if name == "cx":
        return _jax_cx()
    if name == "cz":
        return _jax_cz()
    if name == "cy":
        return _jax_cy()
    if name == "swap":
        return _jax_swap()
    if name == "crx":
        return _jax_controlled(_jax_rx(_jax_scalar_param(instruction.params["theta"])))
    if name == "cry":
        return _jax_controlled(_jax_ry(_jax_scalar_param(instruction.params["theta"])))
    if name == "crz":
        return _jax_controlled(_jax_rz(_jax_scalar_param(instruction.params["theta"])))
    if name == "cphase":
        return _jax_cphase(_jax_scalar_param(instruction.params["theta"]))
    if name == "rxx":
        return _jax_rxx(_jax_scalar_param(instruction.params["theta"]))
    if name == "ryy":
        return _jax_ryy(_jax_scalar_param(instruction.params["theta"]))
    if name == "rzz":
        return _jax_rzz(_jax_scalar_param(instruction.params["theta"]))
    raise NotImplementedError(f"JAX quantum kernel does not support gate {name!r} yet.")


def _apply_jax_instruction(
    state: Any,
    instruction: Any,
    n_wires: int,
    parameters: Any,
    matmul_precision: str | None = "highest",
) -> Any:
    wires = tuple(int(wire) for wire in instruction.wires)
    matrix = _jax_instruction_matrix(instruction)
    del parameters
    return _jax_apply_matrix(state, matrix, wires, n_wires, matmul_precision)


from .mps_kernel import (  # noqa: E402
    _is_zz_z_chain_hamiltonian,
    _jax_mps_from_circuit,
    _jax_mps_hamiltonian_expectation,
    _jax_mps_z_sum,
    _jax_mps_z_values,
)


def _jax_tensor_network_statevector_from_circuit(
    circuit: Any,
    n_wires: int,
    parameters: Any,
    *,
    matmul_precision: str | None = "highest",
) -> Any:
    del parameters
    nodes, current_labels, _next_label = _jax_tensor_network_nodes_from_circuit(
        circuit,
        n_wires,
        start_label=0,
        conjugate=False,
    )
    out = _jax_contract_nodes_greedy(nodes, tuple(current_labels), matmul_precision)
    return out.reshape(-1)


def _jax_tensor_network_nodes_from_circuit(
    circuit: Any,
    n_wires: int,
    *,
    start_label: int = 0,
    conjugate: bool = False,
) -> tuple[list[tuple[Any, tuple[int, ...]]], list[int], int]:
    import jax.numpy as jnp

    next_label = int(start_label)
    current_labels = []
    nodes: list[tuple[Any, tuple[int, ...]]] = []
    zero = jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=_jax_complex_dtype())
    for _wire in range(int(n_wires)):
        label = next_label
        next_label += 1
        current_labels.append(label)
        nodes.append((zero, (label,)))
    for instruction in circuit.to_ir():
        if instruction.metadata.get("is_channel"):
            raise NotImplementedError(
                "JAX tensor_network mode currently supports unitary circuit instructions."
            )
        wires = tuple(int(wire) for wire in instruction.wires)
        matrix = _jax_instruction_matrix(instruction).reshape((2,) * (2 * len(wires)))
        if conjugate:
            matrix = jnp.conj(matrix)
        input_labels = tuple(current_labels[wire] for wire in wires)
        output_labels = tuple(range(next_label, next_label + len(wires)))
        next_label += len(wires)
        for wire, label in zip(wires, output_labels):
            current_labels[wire] = label
        nodes.append((matrix, output_labels + input_labels))
    return nodes, current_labels, next_label


def _jax_tensor_network_expectation_product_ops(
    circuit: Any,
    n_wires: int,
    ops: dict[int, Any],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    ket_nodes, ket_labels, next_label = _jax_tensor_network_nodes_from_circuit(
        circuit,
        n_wires,
        start_label=0,
        conjugate=False,
    )
    bra_nodes, bra_labels, _next_label = _jax_tensor_network_nodes_from_circuit(
        circuit,
        n_wires,
        start_label=next_label,
        conjugate=True,
    )
    identity = _jax_pauli_matrix("i")
    op_nodes = []
    for wire in range(int(n_wires)):
        op = ops.get(int(wire), identity)
        op_nodes.append((op, (int(bra_labels[wire]), int(ket_labels[wire]))))
    value = _jax_contract_nodes_greedy(
        bra_nodes + ket_nodes + op_nodes, tuple(), matmul_precision
    )
    return jnp.real(value.reshape(()))


def _jax_tensor_network_z_values(
    circuit: Any,
    n_wires: int,
    wires: Iterable[int],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    z_op = _jax_pauli_matrix("z")
    values = []
    for wire in wires:
        values.append(
            _jax_tensor_network_expectation_product_ops(
                circuit,
                n_wires,
                {int(wire): z_op},
                matmul_precision,
            )
        )
    return jnp.stack(values) if values else jnp.zeros((0,), dtype=_jax_real_dtype())


def _jax_tensor_network_z_sum(
    circuit: Any,
    n_wires: int,
    wires: Iterable[int],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    values = _jax_tensor_network_z_values(circuit, n_wires, wires, matmul_precision)
    return jnp.sum(values)


def _jax_tensor_network_pauli_string_expectation(
    circuit: Any,
    n_wires: int,
    ops: tuple[tuple[int, str], ...],
    matmul_precision: str | None,
) -> Any:
    op_map = {}
    for wire, name in ops:
        normalized = str(name).lower()
        if normalized != "i":
            op_map[int(wire)] = _jax_pauli_matrix(normalized)
    return _jax_tensor_network_expectation_product_ops(
        circuit, n_wires, op_map, matmul_precision
    )


def _jax_tensor_network_hamiltonian_expectation(
    circuit: Any,
    n_wires: int,
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    total = jnp.zeros((), dtype=_jax_real_dtype())
    for coefficient, ops in terms:
        total = total + jnp.asarray(
            coefficient, dtype=_jax_real_dtype()
        ) * _jax_tensor_network_pauli_string_expectation(
            circuit,
            n_wires,
            ops,
            matmul_precision,
        )
    return total


def _jax_contract_nodes_greedy(
    nodes: list[tuple[Any, tuple[int, ...]]],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    active = list(nodes)
    final_outputs = set(output_labels)
    while len(active) > 1:
        counts: dict[int, int] = {}
        for _tensor, labels in active:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        pair = None
        for left_index in range(len(active)):
            for right_index in range(left_index + 1, len(active)):
                shared = set(active[left_index][1]) & set(active[right_index][1])
                if shared:
                    pair = (left_index, right_index)
                    break
            if pair is not None:
                break
        if pair is None:
            pair = (0, 1)
        left_index, right_index = pair
        left_tensor, left_labels = active[left_index]
        right_tensor, right_labels = active[right_index]
        pair_counts = dict(counts)
        for label in left_labels + right_labels:
            pair_counts[label] -= 1
        pair_outputs = tuple(
            label
            for label in tuple(dict.fromkeys(left_labels + right_labels))
            if label in final_outputs or pair_counts.get(label, 0) > 0
        )
        tensor = _jax_einsum_by_labels(
            left_tensor,
            left_labels,
            right_tensor,
            right_labels,
            pair_outputs,
            matmul_precision,
        )
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append((tensor, pair_outputs))
    final_tensor, final_labels = active[0]
    if final_labels != tuple(output_labels):
        final_tensor = _jax_einsum_reorder(
            final_tensor, final_labels, output_labels, matmul_precision
        )
    return final_tensor


_JAX_EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _jax_einsum_by_labels(
    left_tensor: Any,
    left_labels: tuple[int, ...],
    right_tensor: Any,
    right_labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    labels = tuple(dict.fromkeys(left_labels + right_labels + output_labels))
    if len(labels) > len(_JAX_EINSUM_CHARS):
        raise ValueError(
            "JAX tensor_network pair contraction exceeded local einsum label capacity."
        )
    mapping = {label: _JAX_EINSUM_CHARS[index] for index, label in enumerate(labels)}
    equation = (
        "".join(mapping[label] for label in left_labels)
        + ","
        + "".join(mapping[label] for label in right_labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return jnp.einsum(equation, left_tensor, right_tensor, precision=matmul_precision)


def _jax_einsum_reorder(
    tensor: Any,
    labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    all_labels = tuple(dict.fromkeys(labels + output_labels))
    if len(all_labels) > len(_JAX_EINSUM_CHARS):
        raise ValueError(
            "JAX tensor_network final contraction exceeded local einsum label capacity."
        )
    mapping = {
        label: _JAX_EINSUM_CHARS[index] for index, label in enumerate(all_labels)
    }
    equation = (
        "".join(mapping[label] for label in labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return jnp.einsum(equation, tensor, precision=matmul_precision)


from ....simulation.jax_gate_primitives import (  # noqa: E402
    _jax_apply_matrix,
    _jax_complex_dtype,
    _jax_controlled,
    _jax_cphase,
    _jax_cx,
    _jax_cy,
    _jax_cz,
    _jax_h,
    _jax_hamiltonian_expectation,
    _jax_pauli_matrix,
    _jax_phase,
    _jax_real_dtype,
    _jax_rx,
    _jax_rxx,
    _jax_ry,
    _jax_ryy,
    _jax_rz,
    _jax_rzz,
    _jax_s,
    _jax_scalar_param,
    _jax_sdg,
    _jax_swap,
    _jax_sx,
    _jax_sxdg,
    _jax_t,
    _jax_tdg,
    _jax_u2,
    _jax_u3,
    _jax_x,
    _jax_y,
    _jax_z,
    _jax_z_sum,
    _jax_z_values,
    _set_active_jax_compute_dtype,
)


def __getattr__(name: str) -> Any:
    """Preserve private compatibility probes after kernel decomposition."""
    for module_name in (
        "flagquantum.runtime.backends.jax.mps_kernel",
        "flagquantum.simulation.jax_gate_primitives",
    ):
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
