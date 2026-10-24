"""Disposable proposal 091 reference; its JSON is a test recipe, never a plan.

Only the four factories in test_ising_sampled_contract.py depend on this module.
Replace those factories with production paths and delete this file at delivery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.core.ir import _decode_value, _encode_value
from flagquantum.core.ir.program.physical import IsingExchange
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import CapabilityError, ExecutionError
from flagquantum.observables import OutputRequest
from flagquantum.operators import SuperOperator
from flagquantum.runtime import sample_terminal
from flagquantum.runtime._ising_plan import _outputs
from flagquantum.runtime.execution_plan_contract import canonical_hash, canonical_json
from flagquantum.runtime.options import ExecutionOptions
from flagquantum.runtime.result import ExecutionResult, MeasurementResult
from flagquantum.simulation.ising_braid import _exchange_unitary

Channel = Literal["1", "psi"]
Action = tuple[int, bool] | None


class ReferenceSource:
    def __init__(self, *, initial_channel: Channel = "1") -> None:
        self.initial = initial_channel
        self.base = IsingBraidProgram(initial_channel=initial_channel)
        self.actions: tuple[Action, ...] = ()
        self.unread = False

    def exchange(self, left: int, *, clockwise: bool = False) -> ReferenceSource:
        self.base.exchange(left, clockwise=clockwise)
        self.actions += ((left, clockwise),)
        return self

    def measure_pair(self) -> ReferenceSource:
        if None in self.actions or self.unread:
            raise ValueError("only one pair observation is admitted")
        self.actions += (None,)
        return self

    def dephase_pair(self) -> ReferenceSource:
        if None in self.actions or self.unread:
            raise ValueError("only one pair observation is admitted")
        self.unread = True
        return self

    def projector(self, channel: Channel) -> torch.Tensor:
        return self.base.projector(channel)

    def to_ir(self) -> tuple[Channel, tuple[Action, ...], bool]:
        # A test snapshot, explicitly not a ProgramRecord with an admitted opcode.
        return self.initial, self.actions, self.unread


@dataclass(frozen=True)
class ReferenceRecipe:
    encoded: str

    @property
    def identity(self) -> str:
        return canonical_hash(json.loads(self.encoded))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(json.loads(self.encoded), sort_keys=True, indent=indent)


def plan_reference(
    source: ReferenceSource,
    *,
    options: ExecutionOptions,
    outputs: tuple[OutputRequest, ...] = (),
) -> ReferenceRecipe:
    if options.shots != 1 or options.seed is None:
        raise CapabilityError("one explicit shot and seed are required")
    if options.memory_limit_bytes is not None and options.memory_limit_bytes < 216:
        raise CapabilityError("sampling numeric subtotal requires 216 bytes")
    # Existing option/device/output admission remains authoritative for this test.
    fq.plan(source.base, options=replace(options, shots=None), outputs=outputs or None)
    if source.actions.count(None) != 1:
        raise CapabilityError("one sampled boundary is required")
    captured = _outputs(outputs) if outputs else ()
    return ReferenceRecipe(
        canonical_json(
            {
                "schema": "test.ising_sampled_recipe",
                "initial": source.initial,
                "actions": source.actions,
                "seed": options.seed,
                "torch_version": str(torch.__version__),
                "outputs": [
                    {"name": item.name, "matrix": _encode_value(item.observable)}
                    for item in captured
                ],
            }
        )
    )


def read_reference(text: str) -> ReferenceRecipe:
    data = json.loads(text)
    if data.get("schema") != "test.ising_sampled_recipe":
        raise ValueError("this reference reads only test recipes, never native plans")
    if data["torch_version"] != str(torch.__version__):
        raise ValueError("environment_incompatible")
    return ReferenceRecipe(canonical_json(data))


def run_reference(
    source: ReferenceSource | ReferenceRecipe,
    *,
    options: ExecutionOptions | None = None,
    outputs: tuple[OutputRequest, ...] = (),
) -> ExecutionResult:
    if isinstance(source, ReferenceSource):
        if options is None:
            raise CapabilityError("explicit sampling options are required")
        recipe = plan_reference(source, options=options, outputs=outputs)
    else:
        if options is not None or outputs:
            raise ValueError("saved recipe overrides are forbidden")
        recipe = source
    data = json.loads(read_reference(recipe.encoded).encoded)
    state = IsingBraidProgram(initial_channel=data["initial"]).projector(
        data["initial"]
    )
    pair = ["a", "b", "c", "d"]
    projectors = [torch.diag(row).to(torch.complex128) for row in torch.eye(2)]
    event: MeasurementResult | None = None
    for position, action in enumerate(data["actions"]):
        if action is not None:
            left, clockwise = action
            unitary = _exchange_unitary(IsingExchange(left, clockwise))
            state = SuperOperator.left_right_multiply(unitary, unitary.mH).apply(state)
            pair[left], pair[left + 1] = pair[left + 1], pair[left]
            continue
        observed = sample_terminal(
            state.unsqueeze(0),
            {label: [p] for label, p in zip(("1", "psi"), projectors, strict=True)},
            shots=1,
            seed=data["seed"],
        )
        assert isinstance(observed.value, torch.Tensor)
        index = int(observed.value[0, 0])
        branches = [SuperOperator.from_kraus([p]).apply(state) for p in projectors]
        weights = [float(torch.trace(branch).real) for branch in branches]
        if index not in (0, 1) or weights[index] <= 0:
            raise ExecutionError("selected branch must have positive weight")
        state = branches[index] / weights[index]
        event = MeasurementResult(
            "mid_circuit_measurement",
            (),
            observed.value.clone(),
            shots=1,
            metadata={
                "stage": "intermediate",
                "outcome_labels": ["1", "psi"],
                "probabilities": [weights],
                "probability_correction": observed.metadata["probability_correction"],
                "source_operation_index": position,
                "pair_anyon_ids": pair[:2],
            },
        )
    assert event is not None
    measurements = []
    values = []
    for index, output in enumerate(data["outputs"]):
        value = torch.trace(state @ _decode_value(output["matrix"])).real.reshape(1)
        values.append(value)
        measurements.append(
            MeasurementResult(
                "expectation_matrix",
                (),
                value,
                metadata={
                    "name": output["name"],
                    "fq_output_index": index,
                    "fq_output_name": output["name"],
                },
            )
        )
    return ExecutionResult(
        state=state.unsqueeze(0),
        value=torch.stack(values, dim=-1) if values else None,
        measurements=(*measurements, event),
        runtime={
            "mode": "density_matrix",
            "shots": 1,
            "seed": data["seed"],
            "mid_circuit_measurements_available": True,
        },
        # Deliberately no fake ExecutionPlan attachment or production evidence.
    )
