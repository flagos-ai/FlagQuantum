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

"""Quirk adapters for exact-ring rotation channels."""

from ..synth._rotation_channels import ProjectiveChannel, UnitaryChannel
from ._quirk_rotation import to_quirk


def export_unitary_channel_quirk(
    channel: UnitaryChannel, fmt: str = "xz", allow_global_phase: bool = True
) -> str:
    """Export the source Quirk URL for a unitary channel."""
    if channel.twirl:
        raise ValueError("to_quirk is not supported when twirl=True")
    gates = to_quirk(channel.to_matrix(), fmt, allow_global_phase)
    cols = "[" + ",".join(f"[{g}]" for g in gates) + "]"
    return 'https://algassert.com/quirk#circuit={"cols":%s}' % cols


def export_projective_channel_quirk(
    channel: ProjectiveChannel, fmt: str = "xz", allow_global_phase: bool = True
) -> str:
    """Export the source Quirk URL for a measured channel."""
    correction = channel.correction
    if not isinstance(correction, UnitaryChannel):
        raise ValueError(
            f"to_quirk is not supported for correction of type {type(correction)}"
        )
    rot = to_quirk(channel.rotation.to_matrix(), fmt, allow_global_phase)
    cor = to_quirk(correction.to_matrix(), fmt, allow_global_phase)
    xgate = '"X"'
    first_row = []
    second_row = []
    # CNOT
    first_row.append('"•"')
    second_row.append(xgate)
    # rotation
    first_row.extend(rot)
    second_row.extend("1" for _ in rot)
    # CNOT
    first_row.append('"•"')
    second_row.append(xgate)
    # measure
    first_row.append("1")
    second_row.append('"Measure"')
    # correction
    first_row.extend(cor)
    second_row.extend('"•"' for _ in cor)
    cols = (
        "["
        + ",".join(f"[{g1},{g2}]" for g1, g2 in zip(first_row, second_row, strict=True))
        + "]"
    )
    return 'https://algassert.com/quirk#circuit={"cols":%s}' % cols
