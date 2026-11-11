# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 call_cbloq_classically and ClassicalSimState._recurse_impl.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Compose stored pure classical bodies with the original recursive RNG policy."""

from collections.abc import Callable, Mapping

import numpy as np

from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import RegisterType
from .classical_measurement import MeasurementOutcomePhase
from .component_classical_state import ComponentClassicalState
from .finite_field_encoding import FiniteFieldEncoding

# The source helper owns one default handler for its entire process lifetime.
# Pure recursion calls that helper without forwarding the parent's handler.
_DEFAULT_RNG = np.random.default_rng()


def simulate_pure_component_calls(
    record: ProgramRecord,
    call_orders: Mapping[str, tuple[int, ...]],
    values: Mapping[str, object],
    value_action: Callable[[FunctionRecord, dict[str, object]], object],
    phase_action: (
        Callable[
            [FunctionRecord, dict[str, object]],
            complex | MeasurementOutcomePhase | None,
        ]
        | None
    ) = None,
    *,
    entry: str = "main",
    instance_ids: Mapping[str, tuple[int, ...]] | None = None,
    field_encodings: Mapping[RegisterType, FiniteFieldEncoding] | None = None,
    rng: np.random.Generator | None = None,
    fixed_values: Mapping[int, object] | None = None,
) -> dict[str, object]:
    """Evaluate pure stored calls; each nested body uses the shared default RNG.

    An explicit root RNG takes precedence over root fixed values. With neither,
    the root also uses the process default. Nested bodies always use that default,
    even when the root has a fixed or seeded policy, matching the pinned source.
    This differs from phased recursion, which propagates the parent's policy.

    Compiler supplies local schedules and optional persistent instance IDs. The
    existing state owns binding, output commit and phase rejection. No graph
    discovery, decomposition-exception translation or physical lowering occurs.
    """
    orders, ids = dict(call_orders), dict(instance_ids or {})

    def recurse(
        function: FunctionRecord, inputs: dict[str, object]
    ) -> tuple[dict[str, object], None]:
        return evaluate(function.name, inputs, _DEFAULT_RNG, None), None

    def evaluate(
        name: str,
        inputs: Mapping[str, object],
        generator: np.random.Generator | None,
        fixed: Mapping[int, object] | None,
    ) -> dict[str, object]:
        try:
            order = orders[name]
        except KeyError as exc:
            raise NotImplementedError(
                f"no classical call order supplied for {name}"
            ) from exc
        return ComponentClassicalState(
            record,
            order,
            inputs,
            value_action,
            phase_action,
            entry=name,
            instance_ids=ids.get(name),
            field_encodings=field_encodings,
            rng=generator,
            fixed_values=fixed,
            recurse=recurse,
        ).simulate()

    generator = rng if rng is not None or fixed_values is not None else _DEFAULT_RNG
    return evaluate(entry, values, generator, fixed_values)
