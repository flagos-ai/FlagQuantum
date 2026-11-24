# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 block_encoding/linear_combination.py.

"""Pack caller-owned LCU storage around exact native PREPARE and SELECT calls."""

from typing import cast

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt, smax
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..layout.component_partition import partition_component_program


def linear_combination_signature(
    system: SymbolicInt,
    constituent_ancillas: tuple[SymbolicInt, ...],
    constituent_resources: tuple[SymbolicInt, ...],
    *,
    prepare_selection: SymbolicInt,
    prepare_junk: SymbolicInt,
    controlled: bool = False,
) -> Signature:
    """Keep source maximum child widths, packed junk and optional opaque ctrl."""
    if len(constituent_ancillas) < 2 or len(constituent_ancillas) != len(
        constituent_resources
    ):
        raise ValueError("LCU signature needs matching widths for at least two terms")
    if type(controlled) is not bool:
        raise ValueError("LCU signature control flag must be boolean")
    ancilla = cast(SymbolicInt, smax(constituent_ancillas))
    resource = cast(SymbolicInt, smax(constituent_resources))
    return Signature.build_from_dtypes(
        ctrl=opaque_register_type(int(controlled)),
        system=opaque_register_type(system),
        ancilla=opaque_register_type(prepare_selection + ancilla),
        resource=opaque_register_type(resource + prepare_junk),
    )


def linear_combination_component_program(
    signature: Signature,
    *,
    constituent_ancilla: SymbolicInt,
    constituent_resource: SymbolicInt,
    prepare_selection: SymbolicInt,
    prepare_junk: SymbolicInt,
    prepare_pair: tuple[ProgramRecord, ProgramRecord],
    select_pair: tuple[ProgramRecord, ProgramRecord],
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain original partitions, PREPARE/SELECT/unprepare and exact reversals.

    The supplied SELECT pair is the actual positive-controlled pair when ctrl
    is present. No control is propagated into preparation. All packed storage
    is caller-owned, including logical junk; private scratch belongs to children.
    """
    if type(adjoint) is not bool:
        raise ValueError("Linear combination direction must be boolean")
    widths = (
        constituent_ancilla,
        constituent_resource,
        prepare_selection,
        prepare_junk,
    )
    if any(type(width) is not int for width in widths) or any(
        reg.is_symbolic() for reg in signature
    ):
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    pairs: list[tuple[ProgramRecord, ProgramRecord]] = []

    def partition(
        parts: tuple[Register, ...], name: str
    ) -> tuple[ProgramRecord, ProgramRecord]:
        pair = (
            partition_component_program(parts, name, False),
            partition_component_program(parts, name + "_inverse", True),
        )
        pairs.extend((pair, (pair[1], pair[0])))
        return pair

    anc_parts: tuple[Register, ...] = (
        Register("selection", opaque_register_type(prepare_selection)),
    )
    if constituent_ancilla > 0:
        anc_parts += (Register("ancilla", opaque_register_type(constituent_ancilla)),)
    anc_partition, anc_join = partition(anc_parts, "ancilla_partition")
    be_parts: tuple[Register, ...] = (signature.get_left("system"),)
    if constituent_ancilla > 0:
        be_parts += (Register("ancilla", opaque_register_type(constituent_ancilla)),)
    if constituent_resource > 0:
        be_parts += (Register("resource", opaque_register_type(constituent_resource)),)
    be_partition, be_join = partition(be_parts, "block_partition")
    builder = ComponentProgramBuilder(signature)

    def call(
        child: ProgramRecord, values: dict[str, BindingCollection]
    ) -> dict[str, BindingCollection]:
        return builder.call_program(child, values, entry=child.functions[0].name)

    anc = call(anc_partition, {"x": builder.inputs["ancilla"]})
    be = {"system": builder.inputs["system"]}
    if constituent_ancilla > 0:
        be["ancilla"] = anc.pop("ancilla")
    prep_inputs = {"selection": anc.pop("selection")}
    resource_pair: tuple[ProgramRecord, ProgramRecord] | None = None
    if constituent_resource + prepare_junk > 0:
        res_parts: tuple[Register, ...] = ()
        if constituent_resource > 0:
            res_parts += (
                Register("resource", opaque_register_type(constituent_resource)),
            )
        if prepare_junk > 0:
            res_parts += (Register("prepare_junk", opaque_register_type(prepare_junk)),)
        resource_pair = partition(res_parts, "resource_partition")
        resource = call(resource_pair[0], {"x": builder.inputs["resource"]})
        if constituent_resource > 0:
            be["resource"] = resource.pop("resource")
        if prepare_junk > 0:
            prep_inputs["junk"] = resource.pop("prepare_junk")
    prepare, unprepare = prepare_pair
    select, unselect = select_pair
    prepared = call(prepare, prep_inputs)
    selected_inputs = {
        "selection": prepared.pop("selection"),
        "system": call(be_join, be)["x"],
    }
    if "ctrl" in builder.inputs:
        selected_inputs["ctrl"] = builder.inputs["ctrl"]
    selected = call(select, selected_inputs)
    prepared["selection"] = selected.pop("selection")
    unprepared = call(unprepare, prepared)
    be = call(be_partition, {"x": selected.pop("system")})
    outputs = {"system": be.pop("system"), **selected}
    anc = {"selection": unprepared.pop("selection")}
    if constituent_ancilla > 0:
        anc["ancilla"] = be.pop("ancilla")
    outputs["ancilla"] = call(anc_join, anc)["x"]
    if resource_pair is not None:
        resource = {}
        if constituent_resource > 0:
            resource["resource"] = be.pop("resource")
        if prepare_junk > 0:
            resource["prepare_junk"] = unprepared.pop("junk")
        outputs["resource"] = call(resource_pair[1], resource)["x"]
    record = builder.finalize(outputs)
    pairs.extend(((prepare, unprepare), (unprepare, prepare), (select, unselect)))
    return adjoint_component_calls(record, tuple(pairs)) if adjoint else record
