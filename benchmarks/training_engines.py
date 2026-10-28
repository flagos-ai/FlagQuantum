"""Optional native Torch-interface adapters for the fixed training tasks."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, Literal

import numpy as np
import torch

import flagquantum as fq
from benchmarks.training_workloads import Task, fq_state, recipe, terms
from flagquantum import algorithms as fqa

Engine = Literal[
    "flagquantum",
    "flagquantum_torch_module",
    "flagquantum_jax_module",
    "flagquantum_jax_kernel",
    "pennylane_backprop",
    "pennylane_adjoint",
    "tensorcircuit_torch",
    "tensorcircuit_jax",
    "classical",
]
Expectation = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _scalar(value: Any) -> torch.Tensor:
    """Validate the external simulator boundary before using its objective."""
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise TypeError("simulator must return one differentiable Torch scalar")
    result = value.real.reshape(())
    if result.dtype != torch.float64:
        raise ValueError(f"expected a float64 observable, received {result.dtype}")
    if not bool(torch.isfinite(result)):
        raise ValueError("simulator returned a nonfinite expectation")
    return result


def build_expectation(engine: Engine, task: Task) -> Expectation:
    """Construct a named backend without changing its derivative method."""
    qubits = 2 if task == "classifier" else 4
    gates = recipe(task)
    pauli_terms = terms(task)
    if engine in {
        "flagquantum_torch_module",
        "flagquantum_jax_module",
        "flagquantum_jax_kernel",
    }:
        observable = fqa.Hamiltonian(
            [fqa.pauli_term(c, word, wires) for c, word, wires in pauli_terms]
        )

        def builder(parameters: Any, features: Any) -> fq.Circuit:
            circuit = fq.Circuit(qubits, dtype=torch.complex128)
            for gate in gates:
                options = (
                    {"theta": gate.angle(parameters, features)}
                    if gate.parameter is not None or gate.feature is not None
                    else {}
                )
                circuit.gate(gate.name, gate.wires, **options)
            return circuit

        count = 6 if task == "classifier" else 26
        if engine != "flagquantum_jax_kernel":
            module = fq.Module(
                builder,
                n_parameters=count,
                init=torch.zeros(count, dtype=torch.float64),
                dtype=torch.float64,
                policy=fq.RuntimePolicy(
                    execution_options=fq.ExecutionOptions(
                        backend="jax" if engine == "flagquantum_jax_module" else None,
                        mode="statevector",
                        precision="complex128",
                        allow_backend_fallback=False,
                    ),
                    observable="hamiltonian",
                    observable_qubits=tuple(range(qubits)),
                ),
                hamiltonian=observable,
            )

            def module_value(
                parameters: torch.Tensor, features: torch.Tensor
            ) -> torch.Tensor:
                return _scalar(module(features, parameters=parameters))

            return module_value

        from flagquantum.runtime.executors.jax import compile_quantum_kernel

        kernel = compile_quantum_kernel(
            builder,
            torch.zeros(count, dtype=torch.float64),
            n_qubits=qubits,
            observable="hamiltonian",
            hamiltonian=observable,
            compute_dtype="complex128",
            accepts_inputs=True,
            jit=True,
        )

        def kernel_value(
            parameters: torch.Tensor, features: torch.Tensor
        ) -> torch.Tensor:
            return _scalar(kernel(parameters, features))

        return kernel_value
    if engine == "flagquantum":
        observable = fqa.Hamiltonian(
            [
                fqa.pauli_term(coefficient, word, wires)
                for coefficient, word, wires in pauli_terms
            ]
        )

        def native(parameters: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            return _scalar(observable.expectation(fq_state(task, parameters, features)))

        return native
    if engine in {"pennylane_backprop", "pennylane_adjoint"}:
        qml = import_module("pennylane")
        adjoint = engine == "pennylane_adjoint"
        options = {"c_dtype": np.complex128} if adjoint else {}
        device = qml.device(
            "lightning.qubit" if adjoint else "default.qubit", wires=qubits, **options
        )
        operators = []
        for _, word, wires in pauli_terms:
            factors = [
                getattr(qml, "Pauli" + letter)(wires=wire)
                for letter, wire in zip(word, wires, strict=True)
            ]
            operators.append(factors[0] if len(factors) == 1 else qml.prod(*factors))
        observable = qml.Hamiltonian([term[0] for term in pauli_terms], operators)
        names = {
            "x": "PauliX",
            "z": "PauliZ",
            "h": "Hadamard",
            "cx": "CNOT",
            "ry": "RY",
            "rz": "RZ",
            "rxx": "IsingXX",
            "ryy": "IsingYY",
            "rzz": "IsingZZ",
        }

        @qml.qnode(
            device, interface="torch", diff_method="adjoint" if adjoint else "backprop"
        )
        def qnode(parameters: torch.Tensor, features: torch.Tensor) -> Any:
            for gate in gates:
                operation = getattr(qml, names[gate.name])
                if gate.parameter is None and gate.feature is None:
                    operation(wires=gate.wires)
                else:
                    operation(gate.angle(parameters, features), wires=gate.wires)
            return qml.expval(observable)

        def pennylane(parameters: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            return _scalar(qnode(parameters, features))

        return pennylane
    if engine in {"tensorcircuit_torch", "tensorcircuit_jax"}:
        tc = import_module("tensorcircuit")
        tc.set_backend("jax" if engine == "tensorcircuit_jax" else "pytorch")
        tc.set_dtype("complex128")

        if engine == "tensorcircuit_jax":
            jax = import_module("jax")
            jax.config.update("jax_enable_x64", True)

            def jax_value(parameters: Any, features: Any) -> Any:
                circuit = tc.Circuit(qubits)
                for gate in gates:
                    operation = getattr(circuit, gate.name)
                    if gate.parameter is None and gate.feature is None:
                        operation(*gate.wires)
                    else:
                        operation(*gate.wires, theta=gate.angle(parameters, features))
                value = 0.0
                for coefficient, word, wires in pauli_terms:
                    axes = {
                        axis.lower(): tuple(
                            wire
                            for letter, wire in zip(word, wires, strict=True)
                            if letter == axis
                        )
                        for axis in "XYZ"
                    }
                    value = value + coefficient * circuit.expectation_ps(**axes).real
                return value

            compiled = jax.jit(jax.value_and_grad(jax_value))

            class JaxExpectation(torch.autograd.Function):
                """Include the CPU bridge and precomputed JAX VJP in the workload."""

                @staticmethod
                def forward(
                    ctx: Any, parameters: torch.Tensor, features: torch.Tensor
                ) -> torch.Tensor:
                    value, gradient = compiled(
                        parameters.detach().numpy(), features.detach().numpy()
                    )
                    ctx.save_for_backward(
                        torch.from_numpy(np.array(gradient, copy=True))
                    )
                    return torch.tensor(float(value), dtype=torch.float64)

                @staticmethod
                def backward(
                    ctx: Any, upstream: torch.Tensor
                ) -> tuple[torch.Tensor, None]:
                    (gradient,) = ctx.saved_tensors
                    return upstream * gradient, None

            def jax_expectation(
                parameters: torch.Tensor, features: torch.Tensor
            ) -> torch.Tensor:
                return _scalar(JaxExpectation.apply(parameters, features))

            return jax_expectation

        def tensorcircuit(
            parameters: torch.Tensor, features: torch.Tensor
        ) -> torch.Tensor:
            circuit = tc.Circuit(qubits)
            for gate in gates:
                operation = getattr(circuit, gate.name)
                if gate.parameter is None and gate.feature is None:
                    operation(*gate.wires)
                else:
                    operation(*gate.wires, theta=gate.angle(parameters, features))
            values = []
            for coefficient, word, wires in pauli_terms:
                axes = {
                    axis.lower(): tuple(
                        wire
                        for letter, wire in zip(word, wires, strict=True)
                        if letter == axis
                    )
                    for axis in "XYZ"
                }
                values.append(coefficient * _scalar(circuit.expectation_ps(**axes)))
            return torch.stack(values).sum()

        return tensorcircuit
    raise ValueError(f"no quantum expectation for engine {engine}")


def engine_metadata(engine: Engine) -> dict[str, str]:
    """State the measured method, including the separate adjoint family."""
    devices = {
        "flagquantum": "local_statevector",
        "flagquantum_torch_module": "fq.Module / default pytorch / Torch reverse",
        "flagquantum_jax_module": "fq.Module / jax / jit / Torch bridge",
        "flagquantum_jax_kernel": "FlagQuantum JAXQuantumKernel / jit / Torch bridge",
        "pennylane_backprop": "default.qubit",
        "pennylane_adjoint": "lightning.qubit",
        "tensorcircuit_torch": "TensorCircuit-NG Circuit / pytorch",
        "tensorcircuit_jax": "TensorCircuit-NG Circuit / jax / jit / Torch bridge",
        "classical": "quadratic_logistic",
    }
    return {
        "engine": engine,
        "backend": devices[engine],
        "derivative": (
            "adjoint"
            if engine == "pennylane_adjoint"
            else (
                "jax_reverse_with_torch_bridge" if "jax" in engine else "torch_backprop"
            )
        ),
        "device": "cpu",
        "amplitude_dtype": "not_applicable" if engine == "classical" else "complex128",
        "parameter_dtype": "float64",
        "jit": "enabled" if "jax" in engine else "disabled",
        "batching": "serial_per_example" if engine != "classical" else "matrix_vector",
    }
