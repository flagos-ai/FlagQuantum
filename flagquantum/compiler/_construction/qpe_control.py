"""Shared controlled expansion for the existing ordinary QPE effect profile."""

from __future__ import annotations

import math
from collections.abc import Callable

from .._hybrid.model import Operation, Value
from .register_profile import register_width


def emit_controlled_effect(
    control: Value,
    bit: int,
    op: Operation,
    emit: Callable[[str, tuple[Value, ...], dict[str, object]], None],
) -> None:
    """Preserve the source QPE sequence, including conditional scalar phases."""

    def effect(name: str, values: tuple[Value, ...], **attributes: object) -> None:
        emit(name, values, attributes)

    values, attrs = op.operands[1:], op.attributes
    if "control_state" in attrs:
        raise ValueError(
            "explicit nested control states are unsupported in QPE callbacks"
        )

    def phase(locations: tuple[tuple[Value, int], ...], angle: float) -> None:
        all_locations = ((control, bit), *locations)
        effect(
            "qreg.control_phase",
            tuple(value for value, _ in all_locations),
            indices=tuple(index for _, index in all_locations),
            angle=angle,
        )

    if op.name == "qreg.phase":
        phase((), attrs["angle"])
    elif op.name in {"qreg.qft", "qreg.inverse_qft"}:
        if bit != 0 or register_width(control) != 1:
            raise ValueError("borrowed Fourier composition requires a scalar control")
        effect(
            "qreg.control_qft",
            (control, values[0]),
            **({"inverse": True} if op.name == "qreg.inverse_qft" else {}),
        )
    elif op.name == "qreg.gate":
        target = values[0]
        gate, wire = attrs["gate"], attrs["index"]
        direction = -1 if attrs.get("inverse") else 1
        if gate == "x":
            effect("qreg.control_x", (control, target), indices=(bit, wire))
        elif gate in {"z", "s", "phase", "i"}:
            angle = {"z": math.pi, "s": math.pi / 2, "i": 0.0}.get(
                gate, attrs.get("angle", 0.0)
            )
            phase(((target, wire),), direction * angle)
        elif gate in {"h", "rx", "ry", "rz", "sx"}:
            if gate == "h":
                phase(((target, wire),), math.pi)
                axis, angle = "Y", math.pi / 4
            elif gate == "sx":
                phase((), direction * math.pi / 4)
                axis, angle = "X", direction * math.pi / 4
            else:
                axis, angle = gate[-1].upper(), direction * attrs["angle"] / 2
            if not math.isfinite(angle) or not math.isfinite(2 * angle):
                raise ValueError(
                    "Pauli rotation angle and doubled angle must be finite"
                )
            effect(
                "qreg.pauli_rotation",
                (target, control),
                support=((wire, axis),),
                angle=angle,
                control_index=bit,
            )
        else:
            raise ValueError("unsupported gate in QPE callback")
    elif op.name in {"qreg.control_phase", "qreg.control_x"} and "indices" in attrs:
        indices = (bit, *attrs["indices"])
        effect(
            op.name,
            (control, *values),
            indices=indices,
            **({"angle": attrs["angle"]} if op.name == "qreg.control_phase" else {}),
        )
    elif op.name == "qreg.array_phase_flip":
        phase(tuple((values[0], i) for i in range(register_width(values[0]))), math.pi)
    elif (
        op.name == "qreg.pauli_rotation"
        and "control_index" not in attrs
        or op.name == "qreg.unitary"
        and "control_index" not in attrs
    ):
        effect(op.name, (*values, control), **dict(attrs), control_index=bit)
    elif op.name == "qreg.inplace_prepare_amplitudes" and "control_count" not in attrs:
        locations = (
            tuple(zip(values, attrs["indices"], strict=True))
            if "indices" in attrs
            else tuple(
                (values[0], i) for i in reversed(range(register_width(values[0])))
            )
        )
        selected = ((control, bit), *locations)
        effect(
            op.name,
            tuple(value for value, _ in selected),
            **{**attrs, "control_count": 1, "indices": tuple(i for _, i in selected)},
        )
    elif op.name == "qreg.pair_gate" and attrs["gate"] == "cry":
        control_wire, target_wire = attrs.get("indices", (0, 0))
        angle = attrs["angle"] / 2
        for radians in (angle, -angle):
            effect(
                "qreg.gate",
                (values[1],),
                gate="ry",
                index=target_wire,
                angle=radians,
            )
            effect(
                "qreg.control_x",
                (control, *values),
                indices=(bit, control_wire, target_wire),
            )
    elif op.name == "qreg.pair_gate" and attrs["gate"] in {"cx", "cz"}:
        coordinates = tuple(zip(values, attrs.get("indices", (0, 0)), strict=True))
        if attrs["gate"] == "cz":
            phase(coordinates, math.pi)
        else:
            effect(
                "qreg.control_x",
                (control, *values),
                indices=(bit, *(i for _, i in coordinates)),
            )
    else:
        raise ValueError("unsupported controlled operation in QPE callback")
