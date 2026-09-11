"""Maintained small hybrid models used by product acceptance and tutorials."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .algorithms import Hamiltonian, pauli_term
from .circuit import Circuit
from .runtime.contracts import Module, RuntimePolicy


def _deployment_quantum_parameters(module: Module) -> torch.Tensor:
    parameters = module.parameters_tensor
    if parameters is None:
        raise ValueError("hybrid model deployment requires a quantum parameter tensor")
    return parameters.detach().clone()


def _classifier_circuit(parameters: torch.Tensor) -> Circuit:
    return (
        Circuit(2, device=parameters.device)
        .gate("ry", 0, theta=parameters[0])
        .gate("cx", (0, 1))
        .gate("ry", 1, theta=parameters[1])
    )


def _energy_circuit(parameters: torch.Tensor) -> Circuit:
    return (
        Circuit(2, device=parameters.device)
        .gate("ry", 0, theta=parameters[0])
        .gate("cx", (0, 1))
        .gate("ry", 1, theta=parameters[1])
    )


class HybridQuantumClassifier(torch.nn.Module):
    """Two-feature binary classifier with a policy-selected quantum layer."""

    def __init__(
        self,
        *,
        policy: RuntimePolicy | None = None,
        deployment_binding: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.deployment_binding = dict(deployment_binding or {})
        self.encoder = torch.nn.Linear(2, 2)
        self.quantum = Module(
            _classifier_circuit,
            2,
            policy=policy or RuntimePolicy(observable_wires=(1,)),
        )
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def set_runtime_policy(self, policy: RuntimePolicy) -> None:
        self.quantum.set_runtime_policy(policy)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        angles = self.encoder(inputs) + self.quantum.parameters_tensor
        values = [
            self.quantum.execute(parameters=sample).require_value().reshape(())
            for sample in angles
        ]
        return torch.stack(values) + self.bias

    def deployment_parameters(
        self, inputs: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Return the classical binding contract and, when given inputs, bound IRs."""

        payload: dict[str, Any] = {
            "binding": dict(self.deployment_binding),
            "quantum_parameters": _deployment_quantum_parameters(self.quantum),
            "classical_state": {
                name: value.detach().clone()
                for name, value in self.encoder.state_dict().items()
            },
            "binding_requires_inputs": True,
            "classical_postprocessing": {"bias": self.bias.detach().clone()},
        }
        if inputs is None:
            return payload
        angles = self.encoder(inputs) + self.quantum.parameters_tensor
        deployment_angles = angles.detach().cpu()
        payload.update(
            {
                "bound_quantum_parameters": deployment_angles.clone(),
                "bound_ir": tuple(
                    _classifier_circuit(sample).to_ir() for sample in deployment_angles
                ),
                "binding_requires_inputs": False,
            }
        )
        return payload

    def get_extra_state(self) -> dict[str, Any]:
        return {"deployment_binding": dict(self.deployment_binding)}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.deployment_binding = dict(state.get("deployment_binding", {}))


class VariationalEnergyModel(torch.nn.Module):
    """Small variational energy model sharing the same runtime policy surface."""

    def __init__(
        self,
        *,
        policy: RuntimePolicy | None = None,
        deployment_binding: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.deployment_binding = dict(deployment_binding or {})
        hamiltonian = Hamiltonian(
            (
                pauli_term(-0.7, "Z", (0,)),
                pauli_term(-0.4, "Z", (1,)),
                pauli_term(0.2, "XX", (0, 1)),
            )
        )
        selected = policy or RuntimePolicy(
            observable="hamiltonian", observable_wires=(0, 1)
        )
        self.quantum = Module(
            _energy_circuit,
            2,
            policy=selected,
            hamiltonian=hamiltonian,
        )

    def set_runtime_policy(self, policy: RuntimePolicy) -> None:
        self.quantum.set_runtime_policy(policy)

    def forward(self) -> torch.Tensor:
        values = self.quantum()
        if not isinstance(values, torch.Tensor):
            raise TypeError("the quantum energy layer must return a torch.Tensor")
        return values.sum()

    def deployment_parameters(self) -> dict[str, Any]:
        return {
            "binding": dict(self.deployment_binding),
            "quantum_parameters": _deployment_quantum_parameters(self.quantum),
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {"deployment_binding": dict(self.deployment_binding)}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.deployment_binding = dict(state.get("deployment_binding", {}))


__all__ = ["HybridQuantumClassifier", "VariationalEnergyModel"]
