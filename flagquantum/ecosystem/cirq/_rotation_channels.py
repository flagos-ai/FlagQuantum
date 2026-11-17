#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Adapted from Qualtran v0.7.0 rotation_synthesis/channels/_channel.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/channels/_channel.py

"""Cirq adapters for exact-ring rotation channels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ...synth._rotation_channels import ProjectiveChannel, UnitaryChannel
from ._rotation_export import to_cirq
from .conversion import _cirq

if TYPE_CHECKING:
    import cirq


def export_unitary_channel_cirq(
    channel: UnitaryChannel, fmt: str = "xz", qs: Sequence[cirq.Qid] | None = None
) -> cirq.Circuit:
    """Export a unitary rotation channel on the requested qubit."""
    if channel.twirl:
        raise ValueError("to_cirq is not supported when twirl=True")
    cirq_api = _cirq()
    if qs:
        (q,) = qs
    else:
        q = cirq_api.q(0)
    return cast("cirq.Circuit", cirq_api.Circuit(to_cirq(channel.to_matrix(), fmt, q)))


def export_projective_channel_cirq(
    channel: ProjectiveChannel, fmt: str = "xz", qs: Sequence[cirq.Qid] | None = None
) -> cirq.Circuit:
    """Export the source measured control flow on two qubits."""
    cirq_api = _cirq()
    if qs:
        q0, q1 = qs
    else:
        q0, q1 = cirq_api.LineQubit.range(2)
    correction = channel.correction
    if not isinstance(correction, UnitaryChannel):
        raise ValueError("to_cirq does not support a non unitary correction")
    return cast(
        "cirq.Circuit",
        cirq_api.Circuit(
            cirq_api.CNOT(q0, q1),
            cirq_api.CircuitOperation(
                export_unitary_channel_cirq(channel.rotation, fmt, (q0,)).freeze()
            ),
            cirq_api.CNOT(q0, q1),
            cirq_api.measure(q1, key="m"),
            cirq_api.CircuitOperation(
                export_unitary_channel_cirq(correction, fmt, (q0,)).freeze()
            ).with_classical_controls("m"),
        ),
    )
