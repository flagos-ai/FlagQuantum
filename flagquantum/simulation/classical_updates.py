# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ClassicalSimState._update and phased _update.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Commit numerical outputs before rejecting or applying their phase."""

from collections.abc import Callable, Mapping

from .classical_measurement import MeasurementOutcomePhase, apply_classical_phase


def commit_classical_outputs(
    outputs: Mapping[str, object],
    contribution: object,
    assign_outputs: Callable[[Mapping[str, object]], None],
    *,
    operation_name: str,
) -> None:
    """Assign first; pure classical evaluation rejects every non-None phase.

    Assignments already performed by the owner remain visible on any error.
    In particular, a numerical identity phase is not treated as absent.
    """
    assign_outputs(outputs)
    if contribution is not None:
        raise ValueError(
            f"{operation_name} imparts a phase of {contribution}, "
            "and can't be simulated purely classically. "
            "Use phased numerical evaluation instead."
        )


def commit_phased_classical_outputs(
    outputs: Mapping[str, object],
    contribution: complex | MeasurementOutcomePhase | None,
    assign_outputs: Callable[[Mapping[str, object]], None],
    *,
    phase: complex,
    outcomes: Mapping[tuple[str, tuple[int, ...]], object],
) -> complex:
    """Assign first, then resolve the phase using the owner's updated outputs.

    Returning the new phase lets the owner replace its accumulator only on
    success. Partial output writes are not rolled back after a binding or phase
    failure. Wire lookup, dtype checks and sampling remain with the binder.
    """
    assign_outputs(outputs)
    return apply_classical_phase(phase, contribution, outcomes)
