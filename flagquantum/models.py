"""Maintained small hybrid models used by product acceptance and tutorials."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .algorithms import Hamiltonian, pauli_term
from .circuit import Circuit
from .runtime.contracts import Module, RuntimePolicy


def _classifier_circuit(parameters: torch.Tensor) -> Circuit:
    return (
        Circuit(2, device=getattr(parameters, "device", "cpu"))
        .ry(0, parameters[0])
        .cx(0, 1)
        .ry(1, parameters[1])
    )


def _energy_circuit(parameters: torch.Tensor) -> Circuit:
    return (
        Circuit(2, device=getattr(parameters, "device", "cpu"))
        .ry(0, parameters[0])
        .cx(0, 1)
        .ry(1, parameters[1])
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
        self.encoder = torch.nn.Linear(2, 2)
        self.quantum = Module(
            _classifier_circuit,
            2,
            policy=policy or RuntimePolicy(observable_wires=(1,)),
            deployment_binding=deployment_binding,
        )
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def set_runtime_policy(self, policy: RuntimePolicy) -> None:
        self.quantum.set_runtime_policy(policy)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        angles = self.encoder(inputs) + self.quantum.parameters_tensor
        values = [
            self.quantum.execute(parameters=sample).value.reshape(())
            for sample in angles
        ]
        return torch.stack(values) + self.bias

    def deployment_parameters(
        self, inputs: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Return the classical binding contract and, when given inputs, bound IRs."""

        payload: dict[str, Any] = {
            "binding": dict(self.quantum.deployment_binding),
            "quantum_parameters": self.quantum.parameters_tensor.detach().clone(),
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


class VariationalEnergyModel(torch.nn.Module):
    """Small variational energy model sharing the same runtime policy surface."""

    def __init__(
        self,
        *,
        policy: RuntimePolicy | None = None,
        deployment_binding: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
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
            deployment_binding=deployment_binding,
        )

    def set_runtime_policy(self, policy: RuntimePolicy) -> None:
        self.quantum.set_runtime_policy(policy)

    def forward(self) -> torch.Tensor:
        return self.quantum().sum()

    def deployment_parameters(self) -> dict[str, Any]:
        return {
            "binding": dict(self.quantum.deployment_binding),
            "quantum_parameters": self.quantum.parameters_tensor.detach().clone(),
        }


__all__ = ["HybridQuantumClassifier", "VariationalEnergyModel"]
