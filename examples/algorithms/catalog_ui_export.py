# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 complete UI export catalog.

"""Export all native catalog interfaces through their existing domain owners.

Run ``python -m examples.algorithms.catalog_ui_export``. The catalog retains
source metadata collisions and atomic interface scope. Complete decompositions
and per-gate glyphs have separate acceptance evidence.
"""

import json
from pathlib import Path

from examples.algorithms import (
    arithmetic_ui_export,
    block_encoding_ui_export,
    bookkeeping_ui_export,
    chemistry_ui_export,
    field_ui_export,
    modular_ui_export,
    optimization_ui_export,
    other_ui_export,
    register_gate_ui_export,
    rotations_ui_export,
)

EXPORTERS = (
    register_gate_ui_export.export_example,
    chemistry_ui_export.export_example,
    arithmetic_ui_export.export_example,
    modular_ui_export.export_example,
    field_ui_export.export_example,
    rotations_ui_export.export_example,
    block_encoding_ui_export.export_example,
    optimization_ui_export.export_example,
    bookkeeping_ui_export.export_example,
    other_ui_export.export_example,
)


def export_example(destination: Path) -> None:
    """Publish all groups in source section order, retaining existing node files.

    Each domain owns its identities, metadata, documentation and fresh scores.
    Their generated navigation fragments are combined after the domain exports.
    A failed domain leaves the earlier files and its current partial navigation,
    matching the existing sequential developer publication boundary.
    """
    navigation: dict[str, object] = {}
    for export in EXPORTERS:
        export(destination)
        fragment: dict[str, object] = json.loads(
            (destination / "navigation.json").read_text()
        )
        if navigation.keys() & fragment.keys():
            raise ValueError("Catalog UI sections must have a single domain owner")
        navigation.update(fragment)
    (destination / "navigation.json").write_text(json.dumps(navigation, indent=2))


if __name__ == "__main__":
    export_example(Path("catalog_ui_export"))
