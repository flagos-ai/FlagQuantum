"""Review-only canonical IR replacement; no production lowering is installed."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import Instruction, ensure_circuit_ir

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "replacement", (False, True), ids=("current", "canonical-ir-fake")
)
@pytest.mark.parametrize("batch", (1, 8))
@pytest.mark.parametrize("demand", ("parameters", "inputs", "both", "no_grad"))
def test_same_public_training_consumer(
    monkeypatch: pytest.MonkeyPatch, replacement: bool, batch: int, demand: str
) -> None:
    captures = []
    original = fq.Module._build_from_compiled_builder
    templates = {}

    def substitute(owner, inputs, parameters):
        circuit = original(owner, inputs, parameters)
        generation = owner._builder_compile_count
        if generation not in templates:
            ir = ensure_circuit_ir(circuit)
            slots = dict(
                (slot, f"slot_{index}")
                for index, slot in enumerate(owner._builder_slots)
            )
            instructions = tuple(
                Instruction(
                    name=item.name,
                    wires=item.wires,
                    params={
                        name: (
                            fq.Parameter(slots[(index, name)])
                            if (index, name) in slots
                            else value
                        )
                        for name, value in item.params.items()
                    },
                    matrix=item.matrix,
                    metadata=item.metadata,
                )
                for index, item in enumerate(ir.instructions)
            )
            templates[generation] = (replace(ir, instructions=instructions), slots)
            captures.append(generation)
        template, slots = templates[generation]
        values = {
            name: circuit._instructions[index].params[key]
            for (index, key), name in slots.items()
        }
        return fq.Circuit.from_ir(template, **circuit.circuit_param).bind_parameters(
            values
        )

    if replacement:
        monkeypatch.setattr(fq.Module, "_build_from_compiled_builder", substitute)

    def build(parameters, features):
        return fq.Circuit(1, bsz=features.shape[0]).ry(
            0, parameters[0].square() + parameters[1] * features[:, 0]
        )

    module = fq.Module(build, 2, init=torch.tensor([0.23, -0.31]), dtype=torch.float64)
    module.parameters_tensor.requires_grad_(demand in ("parameters", "both"))
    reference = (
        module.parameters_tensor.detach()
        .clone()
        .requires_grad_(module.parameters_tensor.requires_grad)
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=0.03)
    reference_optimizer = torch.optim.Adam([reference], lr=0.03)
    for step in range(4):
        features = torch.linspace(-0.3, 0.5, batch, dtype=torch.float64).reshape(
            batch, 1
        )
        features = (features + step * 0.01).requires_grad_(demand in ("inputs", "both"))
        reference_features = (
            features.detach().clone().requires_grad_(features.requires_grad)
        )
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad() if demand == "no_grad" else torch.enable_grad():
            value = module(features)
            expected = (
                reference[0].square() + reference[1] * reference_features[:, 0]
            ).cos()
            torch.testing.assert_close(value, expected, atol=1e-10, rtol=1e-10)
            if demand != "no_grad":
                weights = torch.linspace(-0.7, 1.3, batch, dtype=torch.float64)
                (value * weights).sum().backward()
                (expected * weights).sum().backward()
                if features.requires_grad:
                    torch.testing.assert_close(
                        features.grad, reference_features.grad, atol=1e-10, rtol=1e-10
                    )
                if reference.requires_grad:
                    torch.testing.assert_close(
                        module.parameters_tensor.grad,
                        reference.grad,
                        atol=1e-10,
                        rtol=1e-10,
                    )
                optimizer.step()
                reference_optimizer.step()
                torch.testing.assert_close(
                    module.parameters_tensor, reference, atol=1e-10, rtol=1e-10
                )
    if replacement:
        assert captures == [1]
