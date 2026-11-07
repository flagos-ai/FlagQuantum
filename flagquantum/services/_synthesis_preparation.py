"""Validate named input preparation and construct its initial gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import get_operator_schema
from ..core.parameters import parameter_names_in_value
from ..runtime.dynamic._conditions import instruction_is_dynamic

if TYPE_CHECKING:
    from .synthesis import InputState, UInt


def prepare_inputs(
    inputs: Mapping[str, UInt],
    mapping: Mapping[str, tuple[int, ...]],
    input_state: InputState | None,
    preparation: CircuitIR | None,
    preparation_map: Mapping[str, Sequence[int]] | None,
) -> tuple[tuple[Instruction, ...], str]:
    """Validate explicit preparation before emitting the arithmetic candidate."""
    from .synthesis import InputState

    input_count = sum(spec.width for spec in inputs.values())
    prepared: list[Instruction] = []
    state_name = "basis_default_zero"
    if preparation is not None:
        if (
            not isinstance(preparation, CircuitIR)
            or preparation.n_wires != input_count
            or preparation.shape not in {(1 << input_count,), (1, 1 << input_count)}
            or preparation.measurements
            or preparation.observables
        ):
            raise ValueError(
                "preparation must be single-batch unitary input-only CircuitIR without measurements or observables"
            )
        if preparation_map is None or set(preparation_map) != set(inputs):
            raise ValueError("preparation_map must cover every named quantum input")
        remapping: dict[int, int] = {}
        for name, logical in mapping.items():
            if name == "result":
                continue
            physical = tuple(preparation_map[name])
            if len(physical) != len(logical):
                raise ValueError(f"preparation map width differs for {name!r}")
            for source, destination in zip(physical, logical, strict=True):
                if (
                    isinstance(source, bool)
                    or not isinstance(source, int)
                    or not 0 <= source < input_count
                    or source in remapping
                ):
                    raise ValueError(
                        "preparation maps must be disjoint and cover the input qubits"
                    )
                remapping[source] = destination
        for instruction in preparation.instructions:
            schema = get_operator_schema(instruction.name)
            if (
                schema is None
                or schema.semantic_kind != "unitary"
                or instruction.matrix is not None
                or instruction_is_dynamic(instruction)
            ):
                raise ValueError(
                    "use registered static unitary preparation gates; opaque matrices, measurements and reset are outside this trial"
                )
            if any(parameter_names_in_value(v) for v in instruction.params.values()):
                raise ValueError(
                    "bind symbolic preparation angles before synthesis; compose the arithmetic block for differentiable runtime bindings"
                )
            prepared.append(
                replace(
                    instruction, wires=tuple(remapping[q] for q in instruction.wires)
                )
            )
        state_name = "explicit_preparation"
    else:
        state = input_state or InputState.basis(**dict.fromkeys(inputs, 0))
        if not isinstance(state, InputState) or state.kind not in {"basis", "uniform"}:
            raise TypeError(
                "input_state must be InputState.basis(...) or InputState.uniform()"
            )
        if state.kind == "basis":
            if set(state.values) != set(inputs):
                raise ValueError("basis state must bind every declared input by name")
            for name, number in state.values.items():
                if (
                    isinstance(number, bool)
                    or not isinstance(number, int)
                    or not 0 <= number < 1 << inputs[name].width
                ):
                    raise ValueError(
                        f"basis value for {name!r} must be an integer in [0, {(1 << inputs[name].width) - 1}]"
                    )
                prepared.extend(
                    Instruction("x", (q,))
                    for bit, q in enumerate(reversed(mapping[name]))
                    if number & (1 << bit)
                )
        else:
            if state.values:
                raise ValueError(
                    "uniform preparation covers all input domains and accepts no basis values"
                )
            prepared.extend(Instruction("h", (q,)) for q in range(input_count))
        if input_state is not None:
            state_name = state.kind
    return tuple(prepared), state_name
