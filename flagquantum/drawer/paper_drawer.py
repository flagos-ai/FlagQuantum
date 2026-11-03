"""Render explicit circuit columns for paper figures."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .ir_adapter import to_drawable_circuit

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Gate label, target wire, and (control wire, required bit) pairs.
CircuitGate = tuple[str, int, tuple[tuple[int, int], ...]]


def draw_paper_circuit(
    program: object,
    *,
    inputs: Mapping[int, str] | None = None,
    outputs: Mapping[int, str] | None = None,
    width_mm: float = 86,
) -> "Figure":
    """Draw an IR circuit in source order with publication annotations."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, Rectangle

    drawable = to_drawable_circuit(program)
    wire_count = drawable.n_wires
    gates = _circuit_columns(drawable.op_history)
    measured: dict[int, int] = {}
    wire_range = range(wire_count)
    if wire_count < 1 or not gates or width_mm <= 0:
        raise ValueError("Provide wires, gates, and a positive figure width.")
    for column, (label, target, controls) in enumerate(gates, 1):
        wires = [target, *(wire for wire, _ in controls)]
        if any(wire not in wire_range for wire in wires) or len(set(wires)) != len(wires):
            raise ValueError("Each gate needs distinct, valid target and control wires.")
        if target in measured or any(bit not in (0, 1) for _, bit in controls):
            raise ValueError("Invalid measured target or control bit.")
        if label == "measure":
            if controls:
                raise ValueError("Measurement does not accept controls.")
            measured[target] = column
    for labels in (inputs, outputs):
        if labels and any(wire not in wire_range for wire in labels):
            raise ValueError("Wire labels must reference existing wires.")

    end = len(gates) + 0.65
    ink = "#18232d"
    fig = plt.figure(figsize=(width_mm / 25.4, width_mm / 25.4 *
                              (wire_count + 0.5) / (end + 1.9)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-0.95, end + 0.95), ylim=(wire_count - 0.25, -0.75),
           aspect="equal")
    ax.axis("off")
    for wire in wire_range:
        split = measured.get(wire, end)
        ax.plot((0.48, split), (wire, wire), color=ink, lw=0.8, ls="-", marker="")
        if wire in measured:
            for offset in (-0.052, 0.052):
                ax.plot((split, end), (wire + offset,) * 2, color=ink, lw=0.65, ls="-", marker="")
    for column, (label, target, controls) in enumerate(gates, 1):
        for wire, bit in controls:
            classical = measured.get(wire, column) < column
            offsets = (-0.052, 0.052) if classical else (0,)
            for offset in offsets:
                ax.plot((column + offset,) * 2, (wire, target), color=ink, lw=0.75, ls="-", marker="")
            ax.add_patch(Circle((column, wire), 0.09, facecolor=ink if bit else "white",
                                edgecolor=ink, lw=0.75, zorder=4))
        if label == "X" and controls:
            ax.add_patch(Circle((column, target), 0.21, facecolor="white",
                                edgecolor=ink, lw=0.8, zorder=3))
            ax.plot((column - 0.21, column + 0.21), (target,) * 2,
                    color=ink, lw=0.8, ls="-", marker="", zorder=4)
            ax.plot((column,) * 2, (target - 0.21, target + 0.21),
                    color=ink, lw=0.8, ls="-", marker="", zorder=4)
            continue
        ax.add_patch(Rectangle((column - 0.41, target - 0.31), 0.82, 0.62,
                               facecolor="white", edgecolor=ink, lw=0.8, zorder=3))
        if label == "measure":
            ax.add_patch(Arc((column, target + 0.16), 0.55, 0.34,
                             theta1=180, theta2=360, color=ink, lw=0.8, zorder=4))
            ax.plot((column, column + 0.16), (target + 0.16, target - 0.19),
                    color=ink, lw=0.8, ls="-", marker="", zorder=4)
        else:
            ax.text(column, target, label, ha="center", va="center",
                    fontsize=7, color=ink, zorder=4)
    for labels, x, align in ((inputs, 0.25, "right"), (outputs, end + 0.15, "left")):
        for wire, label in (labels or {}).items():
            ax.text(x, wire, label, ha=align, va="center", fontsize=7, color=ink)
    return fig


def _circuit_columns(operations: Sequence[dict[str, Any]]) -> list[CircuitGate]:
    """Resolve classical bit IDs to the wires that supplied their values."""
    from flagquantum.errors import ValidationError
    from flagquantum.experimental.controlled_pauli import decode_controlled_pauli

    columns: list[CircuitGate] = []
    measured_bits: dict[int, int] = {}
    for op in operations:
        name = op["name"]
        wires = op["qubits"]
        metadata = op["metadata"]
        if op["params"] or metadata.get("condition_clauses"):
            raise ValidationError("Paper circuit drawing does not support this instruction.")
        if name == "measure":
            measured_bits[metadata["classical_bit"]] = wires[0]
        pattern = decode_controlled_pauli(name)
        if metadata.get("quantum_controls"):
            controls = tuple(tuple(pair) for pair in metadata["quantum_controls"])
            label = metadata["target_gate"].upper()
        elif name in {"cx", "cy", "cz", "ccx"}:
            controls = tuple((wire, 1) for wire in wires[:-1])
            label = name[-1].upper()
        elif pattern is not None and op["matrix"] is not None:
            bits, label = pattern
            controls = tuple(zip(wires[:-1], map(int, bits), strict=True))
        elif len(wires) == 1 and op["matrix"] is None:
            controls = ()
            label = {"sd": r"$S^\dagger$", "sdg": r"$S^\dagger$", "measure": "measure"}.get(
                name, name.upper()
            )
        else:
            raise ValidationError(f"Paper circuit drawing does not support {name!r}.")
        try:
            classical = tuple((measured_bits[bit], value)
                              for bit, value in metadata.get("conditions", ()))
        except KeyError as error:
            raise ValidationError("Classical control has no preceding measurement.") from error
        columns.append((label, wires[-1], classical + controls))
    return columns
