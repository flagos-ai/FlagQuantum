"""Bind the source ModAddK value contract through native modular arithmetic."""

from collections.abc import Iterator
from typing import Any

from ...compiler.registers.arithmetic.bounded_increment import (
    bounded_increment_ir,
    bounded_increment_requirements,
)
from ...compiler.registers.modular.bounded_modular_addition import (
    bounded_modular_add_ir,
    bounded_modular_add_requirements,
)
from ...core.ir import CircuitIR
from ...core.ir.program.signatures import Signature
from ._register_export import export_register_cirq


def _mod_add_operations(
    source: Any, operands: tuple[Any, ...], manager: Any, key_prefix: str, tag: object
) -> Iterator[Any]:
    """Expand the declared total permutation, owning clean work for its lifetime.

    The source does not supply a physical ModAddK decomposition. This uses the
    existing native realization, not a manufactured upstream gate trace. Control
    and target words are big endian; padding residues remain unchanged.
    """
    controls = tuple(source.cvs)
    width = source.bitsize
    if type(width) is not int or type(source.add_val) is not int:
        raise ValueError("ModAddK import requires concrete integer width and constant")
    count = len(controls)
    if len(operands) != count + width:
        raise ValueError("ModAddK operands do not match its declared registers")
    unit_step = source.add_val in (1, -1)
    requirements = (
        bounded_increment_requirements
        if unit_step
        else bounded_modular_add_requirements
    )
    quantum, classical = requirements(width, source.mod, count)
    logical = count + width
    scratch = tuple(manager.qalloc(quantum))
    try:
        target = tuple(range(count, logical))
        workspace = tuple(range(logical, logical + quantum))
        slots = tuple(range(classical))
        if unit_step:
            gates = bounded_increment_ir(
                target,
                source.mod,
                workspace,
                slots,
                controls=tuple(range(count)),
                control_values=controls,
                adjoint=source.add_val == -1,
            )
        else:
            gates = bounded_modular_add_ir(
                target,
                source.add_val,
                source.mod,
                workspace,
                slots,
                controls=tuple(range(count)),
                control_values=controls,
            )
        if not gates:
            return
        signature = Signature.build(ctrl=count, x=width, work=quantum)
        ports = {
            "ctrl": tuple(range(count)),
            "x": tuple(range(count, logical)),
            "work": tuple(range(logical, logical + quantum)),
        }
        arrays = {"ctrl": operands[:count], "x": operands[count:], "work": scratch}
        names = {reg.name for reg in signature}
        exported, _ = export_register_cirq(
            CircuitIR(logical + quantum, gates),
            signature,
            {name: wires for name, wires in ports.items() if name in names},
            input_quregs={
                name: wires for name, wires in arrays.items() if name in names
            },
            feedback_key_prefix=key_prefix,
        )
        yield from (op.with_tags(tag) for op in exported.circuit.all_operations())
    finally:
        manager.qfree(scratch)


def _private_modular_keys(cirq: Any, circuit: Any, tag: object) -> Any:
    """Keep generated keys deterministic and disjoint from every source key."""
    operations = tuple(circuit.all_operations())

    def private(op: Any) -> bool:
        return any(value is tag for value in op.tags)

    foreign = set().union(
        *(cirq.measurement_key_names(op) for op in operations if not private(op))
    )
    generated = set().union(
        *(cirq.measurement_key_names(op.untagged) for op in operations if private(op))
    )
    mapping: dict[str, str] = {}
    index = 0
    for key in sorted(generated):
        while f"fq_private_modular_{index}" in foreign:
            index += 1
        mapping[key] = f"fq_private_modular_{index}"
        index += 1

    def rename(op: Any) -> Any:
        if not private(op):
            return op
        bare = op.untagged
        if cirq.is_measurement(bare) or bare.classical_controls:
            return cirq.with_measurement_key_mapping(bare, mapping)
        return bare

    return cirq.Circuit(
        (rename(op) for op in operations),
        strategy=cirq.InsertStrategy.NEW,
    )
