"""TikZ layout for circuits with measurement feedback and open controls."""

from collections.abc import Mapping

from .ir_adapter import to_drawable_circuit
from .paper_drawer import _circuit_columns


def source_link(key: str, x: float, y: float, width: float, height: float) -> str:
    """Emit a gallery source anchor at a drawer-computed location."""
    return f"% fq-source {key} {x:g} {y:g} {width:g} {height:g}"


def paper_circuit_tex(
    program: object,
    inputs: Mapping[int, str],
    outputs: Mapping[int, str],
) -> str:
    """Render circuit columns, using narrower spacing around control symbols."""
    circuit = to_drawable_circuit(program)
    columns = _circuit_columns(circuit.op_history)
    positions = []
    previous_width = 0.0
    x = 0.0
    for label, _, controls in columns:
        width = 5.2 if label == "measure" else 4.7 if label.startswith("$") else (
            2.0 if label == "X" and controls else 3.8
        )
        x += (previous_width + width) / 2 + 1.05
        positions.append(x)
        previous_width = width
    end = x + previous_width / 2 + 1.4
    measured = {
        target: x for x, (label, target, _) in zip(positions, columns)
        if label == "measure"
    }

    lines = [
        r"\begin{tikzpicture}[x=1mm,y=-4.6mm,line width=.36pt,",
        r"gate/.style={draw,fill=white,minimum height=3.2mm,inner xsep=.65mm,inner ysep=.25mm},",
        r"doublewire/.style={double=white,double distance=.42mm},",
        r"every node/.style={font=\fontsize{8}{9}\selectfont}]",
    ]
    for wire in range(circuit.n_qubits):
        split = measured.get(wire, end)
        lines.append(rf"\draw (0,{wire}) -- ({split:g},{wire});")
        if wire in measured:
            lines.append(rf"\draw[doublewire] ({split:g},{wire}) -- ({end:g},{wire});")
    for step, (x, (label, target, controls)) in enumerate(zip(positions, columns)):
        lines.append(source_link(f"gate-0-{step}", x, target, 4, 3))
        for wire, bit in controls:
            split = measured.get(wire, end)
            classical = split < x
            kind = "classical" if classical else "quantum" if bit == 0 else "gate"
            lines.append(source_link(f"{kind}-0-{step}", x, wire, 2.2, 2.2))
            if classical:
                lines.append(source_link(
                    f"classical-0-{step}", x, (wire + target) / 2, 2, 2.6
                ))
                left = split + 2.8  # Leave the measurement symbol independently clickable.
                if left < x:
                    lines.append(source_link(
                        f"classical-0-{step}", (left + x) / 2, wire, x - left, 1.6
                    ))
            style = "[doublewire]" if classical else ""
            lines.append(rf"\draw{style} ({x:g},{wire}) -- ({x:g},{target});")
        for wire, bit in controls:
            fill = "black" if bit else "white"
            lines.append(rf"\filldraw[fill={fill}] ({x:g},{wire}) circle[radius=.43mm];")
        if label == "X" and controls:
            lines.extend([
                rf"\filldraw[fill=white] ({x:g},{target}) circle[radius=1mm];",
                rf"\draw ({x:g},{target}) ++(-1mm,0) -- ++(2mm,0);",
                rf"\draw ({x:g},{target}) ++(0,-1mm) -- ++(0,2mm);",
            ])
        elif label == "measure":
            lines.extend([
                rf"\node[gate,minimum width=4.6mm] (meter) at ({x:g},{target}) {{\phantom{{M}}}};",
                r"\draw (meter.center) ++(-1.35mm,-.65mm) to[bend left=55] ++(2.7mm,0);",
                r"\draw (meter.center) ++(-.25mm,-.8mm) -- ++(1.35mm,1.85mm);",
            ])
        else:
            math_label = label if label.startswith("$") else f"${label}$"
            lines.append(rf"\node[gate] at ({x:g},{target}) {{{math_label}}};")
    for labels, x, anchor in ((inputs, -0.7, "east"), (outputs, end + .7, "west")):
        for wire, label in labels.items():
            if wire not in range(circuit.n_qubits):
                raise ValueError("Wire labels must reference existing wires.")
            if anchor == "east":
                lines.append(source_link(f"input-0-{wire}", x - 2, wire, 4, 3))
            lines.append(rf"\node[anchor={anchor},inner sep=.4mm] at ({x:g},{wire}) {{{label}}};")
    return "\n".join([*lines, r"\end{tikzpicture}", ""])
