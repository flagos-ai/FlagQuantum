"""Amazon Braket IQM dynamic dialect implementation."""

import math
from typing import Any, Iterable

import torch

from ....core.ir import Instruction
from .._conditions import instruction_conditions as _instruction_conditions
from ..circuit import DynamicCircuit


def _qasm_angle(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.requires_grad:
            raise ValueError("braket_iqm requires bound scalar gate parameters")
        value = value.detach().cpu().item()
    return repr(float(value))


def iqm_qubit_groups(backend: Any) -> tuple[frozenset[int], ...]:
    metadata = getattr(backend, "metadata", {}) or {}
    raw_groups = metadata.get("dynamic_qubit_groups")
    if raw_groups is None:
        return ()
    return tuple(frozenset(int(wire) for wire in group) for group in raw_groups)


def _native_gate_qasm(instruction: Instruction) -> str:
    wires = ", ".join(f"${wire}" for wire in instruction.wires)
    if instruction.name == "x":
        return f"prx({math.pi!r}, 0.0) {wires};"
    if instruction.name == "rx":
        angle = _qasm_angle(next(iter(instruction.params.values())))
        return f"prx({angle}, 0.0) {wires};"
    if instruction.name in {"rz", "cz"}:
        params = (
            "("
            + ", ".join(_qasm_angle(value) for value in instruction.params.values())
            + ")"
            if instruction.params
            else ""
        )
        return f"{instruction.name}{params} {wires};"
    raise ValueError(f"braket_iqm_unsupported_native_gate:{instruction.name}")


def _conditional_gate_qasm(
    instruction: Instruction,
    conditions: tuple[tuple[int, int], ...],
    groups: tuple[frozenset[int], ...],
    latest_key: dict[int, int],
    measured_wire: dict[int, int],
    target_controller: dict[int, int],
) -> tuple[str, int]:
    if len(conditions) != 1 or conditions[0][1] != 1:
        raise ValueError("braket_iqm_conditions_require_one_bit_equal_to_one")
    bit = conditions[0][0]
    if bit not in latest_key:
        raise ValueError("braket_iqm_feedback_bit_was_read_before_measurement")
    if len(instruction.wires) != 1 or instruction.name not in {"x", "rx"}:
        raise ValueError("braket_iqm_conditional_gate_cannot_lower_to_cc_prx")

    key = latest_key[bit]
    control = measured_wire[key]
    target = instruction.wires[0]
    if not any(control in group and target in group for group in groups):
        raise ValueError("braket_iqm_feedback_pair_outside_dynamic_qubit_group")
    previous = target_controller.setdefault(target, control)
    if previous != control:
        raise ValueError("braket_iqm_target_has_multiple_feedback_controllers")
    angle = (
        math.pi if instruction.name == "x" else next(iter(instruction.params.values()))
    )
    return f"cc_prx({_qasm_angle(angle)}, 0.0, {key}) ${target};", key


def _qasm_document(n_wires: int, body: list[str]) -> str:
    return (
        "\n".join(
            [
                "OPENQASM 3.0;",
                f"bit[{n_wires}] b;",
                "#pragma braket verbatim",
                "box{",
                *(f"    {line}" for line in body),
                "}",
                *(f"b[{wire}] = measure ${wire};" for wire in range(n_wires)),
            ]
        )
        + "\n"
    )


def export_braket_iqm_dynamic_qasm3(
    circuit: DynamicCircuit,
    *,
    qubit_groups: Iterable[Iterable[int]] | None = None,
) -> str:
    """Lower FlagQuantum feedback to IQM ``measure_ff``/``cc_prx`` QASM."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("Braket IQM dynamic export requires DynamicCircuit")
    groups = tuple(
        frozenset(int(wire) for wire in group) for group in (qubit_groups or ())
    )
    if not groups:
        raise ValueError("braket_iqm_dynamic_qubit_groups_are_required")
    latest_key: dict[int, int] = {}
    measured_wire: dict[int, int] = {}
    feed_forward_keys: set[int] = set()
    target_controller: dict[int, int] = {}
    next_key = 0
    body: list[str] = []

    for instruction in circuit._instructions:
        conditions = _instruction_conditions(instruction)
        if instruction.name == "measure":
            if conditions:
                raise ValueError("braket_iqm_conditional_measurement_is_unsupported")
            bit = int(instruction.metadata["classical_bit"])
            key = next_key
            next_key += 1
            latest_key[bit] = key
            measured_wire[key] = instruction.wires[0]
            body.append(f"measure_ff({key}) ${instruction.wires[0]};")
            continue
        if instruction.name == "reset":
            if conditions:
                raise ValueError("braket_iqm_conditional_reset_is_unsupported")
            wire = instruction.wires[0]
            key = next_key
            next_key += 1
            measured_wire[key] = wire
            body.extend(
                (
                    f"measure_ff({key}) ${wire};",
                    f"cc_prx({math.pi!r}, 0.0, {key}) ${wire};",
                )
            )
            feed_forward_keys.add(key)
            target_controller[wire] = wire
            continue
        if conditions:
            line, key = _conditional_gate_qasm(
                instruction,
                conditions,
                groups,
                latest_key,
                measured_wire,
                target_controller,
            )
            body.append(line)
            feed_forward_keys.add(key)
            continue
        body.append(_native_gate_qasm(instruction))
    if set(measured_wire) - feed_forward_keys:
        raise ValueError("braket_iqm_mid_circuit_measurement_requires_feed_forward")
    return _qasm_document(circuit.n_wires, body)


__all__ = ("export_braket_iqm_dynamic_qasm3",)
