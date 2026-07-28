"""Dynamic topology-routing implementation."""

from dataclasses import replace

from ...compilation.routing import CouplingMap, route_to_topology
from .circuit import DynamicCircuit


def route_dynamic_circuit(
    circuit: DynamicCircuit,
    coupling_map: CouplingMap,
) -> DynamicCircuit:
    """Route each gate with restored layout across every dynamic boundary."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("dynamic routing requires DynamicCircuit")
    routed_ir = route_to_topology(
        circuit.to_ir(),
        coupling_map,
        strategy="restore_after_each_gate",
    )
    boundaries = tuple(
        index
        for index, instruction in enumerate(circuit._instructions)
        if instruction.metadata.get("is_dynamic")
    )
    metadata = dict(routed_ir.metadata)
    routing = dict(metadata["routing"])
    routing["dynamic_boundary_count"] = len(boundaries)
    routing["dynamic_boundary_source_indices"] = boundaries
    routing["dynamic_boundary_mapping_policy"] = "identity_restored_per_gate"
    routing["conditional_routing_semantics"] = "unconditional_swap_sandwich"
    metadata["routing"] = routing
    routed_ir = replace(routed_ir, metadata=metadata)
    routed = DynamicCircuit(
        n_qubits=circuit.n_wires,
        bsz=circuit.bsz,
        device=circuit.device,
        dtype=circuit.dtype,
        inputs=circuit._inputs,
        config=circuit.runtime_config,
    )
    routed._instructions.extend(routed_ir.instructions)
    routed._ir_cache = routed_ir
    return routed


__all__ = ("route_dynamic_circuit",)
